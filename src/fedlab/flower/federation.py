"""Bring up a TLS + node-authenticated Flower federation on a Deployment, and submit runs to it.

Layout: the server node runs the SuperLink (and `flwr run`), every client node runs a SuperNode.
Certificates and node keys are generated fresh on every `start`, so a changed server IP never
leaves a stale certificate behind.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from ..modules.base import Deployment, Site
from ..remote import Host
from . import certs

FLEET_PORT = 9092  # the only SuperLink port other nodes reach
SERVERAPPIO_PORT = 9091
CONTROL_PORT = 9093
CLIENTAPPIO_PORT = 9094
HEALTH_PORT = 9101
CONNECTION = "fedlab"  # name of the SuperLink connection in the server's ~/.flwr/config.toml
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))  # TOML basic strings accept JSON escapes


def toml_pairs(values: dict) -> str:
    return " ".join(f"{k}={toml_value(v)}" for k, v in values.items())


@contextmanager
def heartbeat(emit: Callable[[str], None] | None, message: str, every: float = 60.0):
    """While the block runs, call `emit("<message> (<elapsed>)")` every `every` seconds."""
    done = threading.Event()
    started = time.monotonic()

    def beat() -> None:
        while not done.wait(every):
            elapsed = int(time.monotonic() - started)
            if emit:
                emit(f"{message} ({elapsed // 60}m{elapsed % 60:02d}s)")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1)


class FederationError(RuntimeError):
    pass


class Federation:
    def __init__(self, d: Deployment, env_dir: str = "fl"):
        self.d = d
        self.env_dir = env_dir

    # ---- paths and environment ------------------------------------------------
    def bin(self, host: Host) -> str:
        return f"{host.home}/{self.env_dir}/bin"

    def env(self, host: Host, extra: dict[str, str] | None = None) -> dict[str, str]:
        # Without the venv's bin first on PATH, SuperLink/SuperNode spawn the wrong `flower-superexec`.
        return {"PATH": f"{self.bin(host)}:{SAFE_PATH}", **(extra or {})}

    def _pki(self, host: Host) -> str:
        return f"{host.home}/pki"

    def _port(self, host: Host, base: int, index: int) -> int:
        return base + index if host.shares_machine else base

    # ---- lifecycle ------------------------------------------------------------
    def start(
        self,
        node_config: Callable[[Site], dict],
        extra_env: Callable[[Site], dict[str, str]] = lambda site: {},
        timeout: float = 240,
    ) -> None:
        d, server = self.d, self.d.server
        pki_dir = d.state_dir / "pki"
        shutil.rmtree(pki_dir, ignore_errors=True)
        ca = certs.make_ca(pki_dir)
        certs.issue_server_cert(pki_dir, ca, [server.host.address, "127.0.0.1", "localhost"])
        keys = {c.node.name: certs.make_node_keys(pki_dir, f"node-{c.node.name}") for c in d.clients}

        d.log("federation: stopping old services, uploading certificates")
        self.stop()
        sh = server.host
        pki = self._pki(sh)
        sh.run(f"rm -rf {pki} {sh.home}/superlink && mkdir -p {pki} {sh.home}/superlink {sh.home}/.flwr")
        for name in ("ca.pem", "server.pem", "server.key"):
            sh.put(pki_dir / name, f"{pki}/{name}")
        for key in keys.values():
            sh.put(key.public_path, f"{pki}/{key.public_path.name}")
        sh.put_text(
            f'[superlink]\ndefault = "{CONNECTION}"\n\n[superlink.{CONNECTION}]\n'
            f'address = "127.0.0.1:{CONTROL_PORT}"\nroot-certificates = "{pki}/ca.pem"\n',
            f"{sh.home}/.flwr/config.toml",
        )
        for c in d.clients:
            ch = c.host
            ch.run(f"rm -rf {self._pki(ch)} && mkdir -p {self._pki(ch)}")
            ch.put(pki_dir / "ca.pem", f"{self._pki(ch)}/ca.pem")
            ch.put(keys[c.node.name].private_path, f"{self._pki(ch)}/node-key")

        d.log("federation: starting SuperLink")
        sh.start_service(
            "superlink",
            f"flower-superlink --ssl-ca-certfile {pki}/ca.pem --ssl-certfile {pki}/server.pem "
            f"--ssl-keyfile {pki}/server.key --enable-supernode-auth --database {sh.home}/superlink/state.db "
            f"--disable-runtime-dependency-installation --fleet-api-address 0.0.0.0:{FLEET_PORT} "
            f"--control-api-address 127.0.0.1:{CONTROL_PORT} --serverappio-api-address 127.0.0.1:{SERVERAPPIO_PORT}",
            self.env(sh, extra_env(server)),
        )
        self._wait(lambda: self._flwr("supernode ls", check=False)[0] == 0, timeout, "SuperLink did not come up", "superlink")
        for key in keys.values():
            self._flwr(f"supernode register {pki}/{key.public_path.name}")

        d.log("federation: starting SuperNodes")
        for i, c in enumerate(d.clients):
            ch = c.host
            ch.start_service(
                "supernode",
                f"flower-supernode --root-certificates {self._pki(ch)}/ca.pem "
                f"--superlink {sh.address}:{FLEET_PORT} --auth-supernode-private-key {self._pki(ch)}/node-key "
                f"--node-config {_quote(toml_pairs(node_config(c)))} "
                f"--clientappio-api-address 127.0.0.1:{self._port(ch, CLIENTAPPIO_PORT, i)} "
                f"--health-server-address 127.0.0.1:{self._port(ch, HEALTH_PORT, i)}",
                self.env(ch, extra_env(c)),
            )
        self._wait(
            lambda: self.online_nodes() >= len(d.clients), timeout,
            f"fewer than {len(d.clients)} SuperNodes came online", "supernode",
        )
        d.log(f"federation: {len(d.clients)} SuperNodes online")

    def stop(self) -> None:
        self.d.server.host.stop_service("superlink")
        for c in self.d.clients:
            c.host.stop_service("supernode")

    # ---- flwr CLI on the server -------------------------------------------------
    def _flwr(self, args: str, check: bool = True, on_line: Callable[[str], None] | None = None) -> tuple[int, str]:
        sh = self.d.server.host
        # Unbuffered: through an SSH pipe Python would otherwise hold lines back in an 8 KiB block.
        code, out = sh.execute(f"flwr {args} 2>&1", env={**self.env(sh), "PYTHONUNBUFFERED": "1"}, on_line=on_line)
        out = ANSI.sub("", out)
        if check and code != 0:
            raise FederationError(f"`flwr {args}` failed ({code}):\n{out[-2000:]}")
        return code, out

    def online_nodes(self) -> int:
        _, out = self._flwr("supernode ls", check=False)
        return sum(1 for line in out.splitlines() if re.search(r"\bonline\b", line))

    def _wait(self, ready: Callable[[], bool], timeout: float, message: str, service: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready():
                return
            time.sleep(3)
        host = self.d.server.host if service == "superlink" else self.d.clients[0].host
        raise FederationError(f"{message}. Last {service} log:\n{host.service_log(service, 30)}")

    # ---- runs ---------------------------------------------------------------------
    def submit(self, app_dir: str, run_config: dict, on_line: Callable[[str], None] | None = None) -> str:
        """Run a Flower app (a directory on the server) to completion. Returns the run id."""
        sh = self.d.server.host
        cfg = f" --run-config {_quote(toml_pairs(run_config))}" if run_config else ""
        with heartbeat(on_line, "run still in progress"):
            code, out = self._flwr(f"run {app_dir} {CONNECTION} --stream{cfg}", on_line=on_line)
        match = re.search(r"started run (\d+)", out)
        if not match:
            raise FederationError(f"could not find a run id in `flwr run` output:\n{out[-2000:]}")
        run_id = match.group(1)
        status = self.run_status(run_id)
        if not status.startswith("finished:completed"):
            raise FederationError(f"run {run_id} ended with status {status!r}")
        return run_id

    def run_status(self, run_id: str) -> str:
        _, out = self._flwr(f"list --run-id {run_id}", check=False)
        match = re.search(r"(finished:\w+|running|pending|starting)", out)
        return match.group(1) if match else "unknown"


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)
