"""The orchestrator: a VM that runs one SuperLink per training request. Deployed with Terraform, driven over SSH.

It exists for tests. `up` creates it (with its own address and CA); `down` removes everything; it also terminates
itself after `orchestrator_auto_terminate_hours`. Its API listens on the VM's loopback, so every call goes through SSH.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Callable

from ..config import Node, Settings
from ..remote import SshHost
from ..ssh import Keypair

HERE = Path(__file__).parent
TERRAFORM = HERE / "terraform"
API = "http://127.0.0.1:8080"
READY_FILE = "/var/tmp/orchestrator-ready"
NGINX_LOG = "/var/log/nginx/fedlab.log"


def my_ip_cidr() -> str:
    """This machine's public address as a /32: the only address allowed to SSH to the orchestrator."""
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as r:
        return r.read().decode().strip() + "/32"


class Orchestrator:
    def __init__(self, cfg: Settings, key: Keypair):
        self.cfg, self.key = cfg, key
        self.dir = cfg.state_dir / "orchestrator"
        self.state, self.outputs_file = (self.dir / "terraform.tfstate").resolve(), self.dir / "outputs.json"

    # ---- Terraform ---------------------------------------------------------------------------
    def _terraform(self, *args: str) -> str:
        env = {**os.environ, "TF_DATA_DIR": str((self.dir / ".terraform").resolve()), "TF_IN_AUTOMATION": "1"}
        p = subprocess.run(["terraform", f"-chdir={TERRAFORM}", *args], env=env, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"terraform {args[0]} failed:\n{(p.stdout + p.stderr)[-3000:]}")
        return p.stdout

    def _variables(self) -> dict:
        return {
            "region": self.cfg.orchestrator_region, "name": self.cfg.run_name, "my_ip_cidr": my_ip_cidr(),
            "ssh_public_key": self.key.public_key, "ssh_user": self.cfg.ssh_user, "flwr_version": version("flwr"),
            "instance_type": self.cfg.orchestrator_instance_type,
            "auto_terminate_hours": self.cfg.orchestrator_auto_terminate_hours,
        }

    def up(self, log: Callable[[str], None] = print) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        variables = self.dir / "vars.json"
        variables.write_text(json.dumps(self._variables()))
        log("orchestrator: terraform init")
        self._terraform("init", "-input=false")
        log("orchestrator: terraform apply (creates the VM, address and CA)")
        self._terraform("apply", "-auto-approve", "-input=false", f"-state={self.state}", f"-var-file={variables.resolve()}")
        out = {k: v["value"] for k, v in json.loads(self._terraform("output", "-json", f"-state={self.state}")).items()}
        self.outputs_file.write_text(json.dumps(out, indent=2))
        log(f"orchestrator: waiting for the service at {out['ip']}")
        self.wait_ready()
        return out

    def down(self, log: Callable[[str], None] = print) -> None:
        if not self.state.exists():
            log("orchestrator: nothing to destroy")
            return
        variables = self.dir / "vars.json"
        log("orchestrator: terraform destroy")
        self._terraform("destroy", "-auto-approve", "-input=false", f"-state={self.state}", f"-var-file={variables.resolve()}")
        self.outputs_file.unlink(missing_ok=True)

    # ---- using it ---------------------------------------------------------------------------------
    @property
    def outputs(self) -> dict:
        if not self.outputs_file.exists():
            raise RuntimeError("no orchestrator is up; run `fedlab orchestrator up`")
        return json.loads(self.outputs_file.read_text())

    @property
    def node(self) -> Node:
        """The orchestrator as a node, so its metrics are collected like any host's."""
        return Node("orchestrator", "aws", self.outputs["region"], "orchestrator", self.cfg.orchestrator_instance_type, 30)

    def host(self) -> SshHost:
        return SshHost("orchestrator", self.outputs["ip"], self.cfg.ssh_user, self.key)

    def wait_ready(self, timeout: float = 900) -> None:
        host, deadline = self.host(), time.time() + timeout
        while time.time() < deadline:
            try:
                if host.execute(f"test -f {READY_FILE} && curl -s -o /dev/null {API}/requests/00000000")[0] == 0:
                    return
            except Exception:  # noqa: BLE001  SSH is not up yet
                pass
            time.sleep(10)
        raise TimeoutError("the orchestrator service did not come up")

    def api(self, method: str, path: str, body: dict | None = None) -> dict:
        data = f" -H 'Content-Type: application/json' --data-binary {shlex.quote(json.dumps(body))}" if body is not None else ""
        code, out = self.host().execute(f"curl -sS -X {method} {API}{path}{data}")
        if code:
            raise RuntimeError(f"orchestrator API call failed: {out.strip()[-300:]}")
        reply = json.loads(out)
        if isinstance(reply, dict) and "error" in reply and "id" not in reply:
            raise RuntimeError(f"orchestrator API {method} {path}: {reply['error']} {reply.get('stderr', '')}"[:600])
        return reply

    def upload_app(self, local: Path, name: str) -> None:
        self.host().put(local, f"/opt/orchestrator/apps/{name}")

    def traffic(self, dest: Path) -> dict:
        """Copy nginx's access log and total the bytes the orchestrator moved, per request id."""
        dest.mkdir(parents=True, exist_ok=True)
        host = self.host()
        host.run(f"sudo cp {NGINX_LOG} /tmp/fedlab-nginx.log && sudo chmod 644 /tmp/fedlab-nginx.log")
        host.get("/tmp/fedlab-nginx.log", dest)
        return summarize_traffic((dest / "fedlab-nginx.log").read_text())


def summarize_traffic(log: str) -> dict:
    """Total nginx's per-request log lines: `<time> <host> <method> <uri> <status> <bytes in> <bytes out> <seconds>`.

    `bytes_in` is what nodes sent to the orchestrator (request size), `bytes_out` what it sent back."""
    per: dict = {}
    for line in log.splitlines():
        f = line.split()
        if len(f) != 8 or not f[5].isdigit() or not f[6].isdigit():
            continue
        rid = f[1].split(".")[0] if f[1] != "-" else "-"
        kind = "fleet" if f[3].startswith("/v1/fleet") else "runtime" if f[3].startswith("/v1/runtime") else "other"
        for key in (rid, "total"):
            t = per.setdefault(key, {"requests": 0, "bytes_in": 0, "bytes_out": 0, "by_api": {}})
            t["requests"] += 1
            t["bytes_in"] += int(f[5])
            t["bytes_out"] += int(f[6])
            a = t["by_api"].setdefault(kind, {"requests": 0, "bytes_in": 0, "bytes_out": 0})
            a["requests"] += 1
            a["bytes_in"] += int(f[5])
            a["bytes_out"] += int(f[6])
    return per
