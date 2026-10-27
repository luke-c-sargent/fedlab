"""Run a Flower federation through the orchestrator instead of a SuperLink on the server VM.

The orchestrator (see `fedlab orchestrator`) runs one SuperLink per request and hands back the endpoint, the CA and
the node keys. Here the `server` node hosts the ServerApp (a SuperExec container) and every client node runs a
SuperNode container; all of them dial out to the orchestrator on 443. The images (`image_cpu`, `image_gpu`) hold only
the toolchain: the node's home directory is mounted at the same path inside the container, so the code, data and
state that the module stages there are found at the paths the module already sets.
"""

from __future__ import annotations

import base64
import json
import shlex
import time
from pathlib import Path
from typing import Callable

from .. import metrics
from ..modules.base import Deployment, Site
from ..orchestrator import Orchestrator
from .federation import FederationError, heartbeat, toml_pairs

NODE, EXEC = "flwr-node", "flwr-exec"
REQ_DIR = "req"  # ~/req on each node: the request's CA and key or secret
STATE_FILE = "orchestrated.json"
COMPASS_IN_IMAGE = "/opt/COMPASS"

INSTALL_DOCKER = (
    "command -v docker >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io; }\n"
    "sudo systemctl enable --now docker"
)
# The AWS deep-learning AMI already has the toolkit; stock Ubuntu (GCP) does not.
INSTALL_NVIDIA_TOOLKIT = (
    "sudo docker info --format '{{json .Runtimes}}' | grep -q nvidia || {\n"
    "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
    "  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list "
    "| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "
    "| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null\n"
    "  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit\n"
    "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\n"
    "}"
)


class OrchestratedFederation:
    def __init__(self, d: Deployment, orch: Orchestrator):
        self.d, self.orch = d, orch
        self.state_file = d.state_dir / STATE_FILE

    # ---- nodes -------------------------------------------------------------------------------
    def image_for(self, site: Site) -> str:
        return self.d.cfg.image_gpu if site.node.gpu else self.d.cfg.image_cpu

    def prepare_node(self, site: Site) -> None:
        """Docker (and the NVIDIA toolkit on GPU nodes), then the image, timed."""
        host, name, image = site.host, site.node.name, self.image_for(site)
        self.d.log(f"stage {name}: docker")
        host.run(INSTALL_DOCKER + ("\n" + INSTALL_NVIDIA_TOOLKIT if site.node.gpu else ""))
        self.d.log(f"stage {name}: pulling {image.rsplit('/', 1)[-1][:40]}")
        started = time.time()
        host.run(f"sudo docker pull -q {shlex.quote(image)}")
        seconds = time.time() - started
        size = host.run(f"sudo docker image inspect -f '{{{{.Size}}}}' {shlex.quote(image)}").strip()
        metrics.pull(self.d, host=name, image=image, seconds=seconds, size_bytes=int(size) if size.isdigit() else None)

    def _docker_run(self, site: Site, mode: str, name: str, args: str, env: dict[str, str], workdir: str | None = None) -> str:
        """`docker run` on the node: its home is mounted at the same path, and the container runs as the node's user."""
        host, image = site.host, self.image_for(site)
        uid, gid = host.run("id -u; id -g").split()
        flags = "".join(f"-e {shlex.quote(f'{k}={v}')} " for k, v in {**env, 'HOME': host.home}.items())
        return host.run(
            f"sudo docker rm -f {name} >/dev/null 2>&1 || true\n"
            f"sudo docker run {mode} --name {name} --user {uid}:{gid} {'--gpus all ' if site.node.gpu else ''}"
            f"-v {host.home}:{host.home} -w {workdir or host.home} {flags}{shlex.quote(image)} {args}"
        )

    def _run_container(self, site: Site, name: str, args: str, env: dict[str, str]) -> None:
        self._docker_run(site, "-d", name, args, env)

    def run_in_image(self, site: Site, args: str, env: dict[str, str], workdir: str | None = None) -> str:
        """Run a command in the node image and return its output (for checks that need the image's Python)."""
        return self._docker_run(site, "--rm", "fedlab-once", args, env, workdir)

    # ---- the request ---------------------------------------------------------------------------
    @property
    def request(self) -> dict:
        if not self.state_file.exists():
            raise FederationError("no orchestrator request for this deployment; run `fedlab deploy` first")
        return json.loads(self.state_file.read_text())

    def start(self, node_config: Callable[[Site], dict], service_env: Callable[[Site], dict[str, str]], timeout: float = 600) -> None:
        d = self.d
        d.log("orchestrated: creating a request (a SuperLink container)")
        req = self.orch.api("POST", "/requests", {"num_nodes": len(d.clients)})
        d.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"id": req["id"], "endpoint": req["endpoint"]}))
        endpoint = req["endpoint"]
        for site in d.sites:
            site.host.run(f"rm -rf ~/{REQ_DIR} && mkdir -m 700 -p ~/{REQ_DIR}")
            site.host.put_text(req["ca_pem"], f"~/{REQ_DIR}/ca.crt")
        for key, site in zip(req["node_keys"], d.clients):
            site.host.put_text(key, f"~/{REQ_DIR}/key")
            site.host.run(f"chmod 600 ~/{REQ_DIR}/key")
            d.log(f"orchestrated: starting the SuperNode on {site.node.name}")
            self._run_container(
                site, NODE,
                f"flower-supernode --root-certificates {REQ_DIR}/ca.crt --superlink {shlex.quote(endpoint)} "
                f"--auth-supernode-private-key {REQ_DIR}/key --node-config {shlex.quote(toml_pairs(node_config(site)))}",
                service_env(site),
            )
        server = d.server
        secret = d.state_dir / "secret"  # raw bytes, so it goes as a file, not through a shell heredoc
        secret.write_bytes(base64.b64decode(req["superexec_secret_b64"]))
        secret.chmod(0o600)
        try:
            server.host.put(secret, f"~/{REQ_DIR}/secret")
        finally:
            secret.unlink(missing_ok=True)
        server.host.run(f"chmod 600 ~/{REQ_DIR}/secret")
        d.log("orchestrated: starting the SuperExec (ServerApp host)")
        self._run_container(
            server, EXEC,
            f"flower-superexec --runtime-api-address {shlex.quote(endpoint)} --root-certificates {REQ_DIR}/ca.crt "
            f"--superexec-auth-secret-file {REQ_DIR}/secret",
            service_env(server),
        )
        self._wait_for_nodes(len(d.clients), timeout)
        d.log(f"orchestrated: {len(d.clients)} SuperNodes online")
        metrics.probe_rtt(d, self.orch.outputs["ip"], 443)

    def _wait_for_nodes(self, count: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            nodes = self.orch.api("GET", f"/requests/{self.request['id']}")["nodes"]
            if sum(n["status"] == "online" for n in nodes) >= count:
                return
            time.sleep(5)
        logs = "\n".join(f"--- {s.node.name}\n{self.logs(s, NODE)[-600:]}" for s in self.d.clients)
        raise FederationError(f"fewer than {count} SuperNodes came online. Last logs:\n{logs}")

    # ---- runs ---------------------------------------------------------------------------------
    def submit(self, app_dir: Path, run_config: dict, on_line: Callable[[str], None] | None = None, timeout: float = 4 * 3600) -> str:
        """Upload the app to the orchestrator, start it, and wait for the run to finish. Returns the run id."""
        name = f"app-{int(time.time())}"  # a fresh directory: the orchestrator runs as root and leaves files in it
        self.orch.upload_app(app_dir, name)
        rid = self.request["id"]
        reply = self.orch.api("POST", f"/requests/{rid}/run", {"app": name, "run_config": toml_pairs(run_config)})
        if not reply.get("success", False):
            raise FederationError(f"the orchestrator would not start the run: {reply}")
        run_id = str(reply["run-id"])
        started = time.monotonic()
        with heartbeat(on_line, "run still in progress"):
            while time.monotonic() - started < timeout:
                runs = {str(r["run-id"]): r for r in self.orch.api("GET", f"/requests/{rid}")["runs"]}
                status = runs.get(run_id, {}).get("status", "unknown")
                if status.startswith("finished"):
                    if status != "finished:completed":
                        raise FederationError(f"run {run_id} ended with status {status!r}: {runs[run_id].get('status-details', '')}")
                    return run_id
                time.sleep(10)
        raise FederationError(f"run {run_id} did not finish within {timeout:.0f}s")

    # ---- logs and cleanup -------------------------------------------------------------------------------
    def logs(self, site: Site, container: str, lines: int = 400) -> str:
        return site.host.run(f"sudo docker logs --tail {int(lines)} {container} 2>&1 || true", check=False)

    def stop(self) -> None:
        """Remove the containers and the request. Best effort: it must not hide the error that led here."""
        for site in self.d.sites:
            site.host.run(f"sudo docker rm -f {NODE} {EXEC} >/dev/null 2>&1 || true", check=False)
        if self.state_file.exists():
            try:
                self.orch.api("DELETE", f"/requests/{self.request['id']}")
            except Exception as e:  # noqa: BLE001
                self.d.log(f"orchestrated: could not delete the request ({e})")
            self.state_file.unlink(missing_ok=True)
