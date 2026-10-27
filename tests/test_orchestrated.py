"""The orchestrated federation against fake hosts and a fake orchestrator: which commands it sends, and in what order."""

import base64
import json

import pytest

from fedlab.config import Node, Settings
from fedlab.flower.federation import FederationError
from fedlab.flower.orchestrated import EXEC, NODE, OrchestratedFederation
from fedlab.modules import Deployment, Site
from fedlab.remote import Host


class FakeHost(Host):
    address = "10.0.0.1"

    def __init__(self, name):
        self.name, self._home, self.commands, self.files = name, "/home/u", [], {}

    def _exec(self, script, on_line):
        self.commands.append(script)
        if "id -u" in script:
            return 0, "1000\n1000\n"
        if "image inspect" in script:
            return 0, "123456\n"
        return 0, ""

    def put(self, local, remote):
        self.files[remote] = local.read_bytes()

    def get(self, remote, local): ...
    def start_service(self, name, command, env): ...
    def stop_service(self, name): ...
    def service_log(self, name, lines=60):
        return ""


class FakeOrchestrator:
    outputs = {"ip": "1.2.3.4"}

    def __init__(self, runs=None):
        self.calls, self.runs, self.uploads = [], runs or [], []

    def api(self, method, path, body=None):
        self.calls.append((method, path, body))
        if (method, path) == ("POST", "/requests"):
            return {"id": "ab12cd34", "endpoint": "ab12cd34.1.2.3.4.sslip.io:443", "ca_pem": "CA\n",
                    "superexec_secret_b64": base64.b64encode(bytes(range(32))).decode(),
                    "node_keys": ["KEY0\n", "KEY1\n"]}
        if method == "GET":
            return {"nodes": [{"status": "online"}, {"status": "online"}], "runs": self.runs}
        if path.endswith("/run"):
            return {"success": True, "run-id": 7}
        return {}

    def upload_app(self, local, name):
        self.uploads.append(name)


def deployment(tmp_path):
    node = lambda n, role, gpu=False, site=None: Node(n, "aws", "x", role, "t", 10, gpu=gpu, site=site)  # noqa: E731
    server = Site(node("srv", "server"), FakeHost("srv"))
    clients = [Site(node("c1", "client", True, "tcga"), FakeHost("c1")), Site(node("c2", "client", True, "gtex"), FakeHost("c2"))]
    return Deployment(Settings(federation="orchestrated"), server, clients, tmp_path / "state", log=lambda m: None)


def test_start_sends_keys_secret_and_containers_to_the_right_hosts(tmp_path, monkeypatch):
    monkeypatch.setattr("fedlab.metrics.probe_rtt", lambda *a, **k: {})
    d, orch = deployment(tmp_path), FakeOrchestrator()
    OrchestratedFederation(d, orch).start(lambda s: {"site": s.node.site}, lambda s: {"PYTHONPATH": "/home/u/pylib"})
    c1, c2, srv = d.clients[0].host, d.clients[1].host, d.server.host
    assert orch.calls[0] == ("POST", "/requests", {"num_nodes": 2})
    assert "KEY0" in "".join(c for c in c1.commands) and "KEY1" in "".join(c2.commands)
    run1 = next(c for c in c1.commands if "docker run" in c)
    assert "flower-supernode" in run1 and "ab12cd34.1.2.3.4.sslip.io:443" in run1 and "--gpus all" in run1
    assert "--user 1000:1000" in run1 and "-v /home/u:/home/u" in run1 and "PYTHONPATH=/home/u/pylib" in run1 and "site=" in run1
    run_srv = next(c for c in srv.commands if "docker run" in c)
    assert "flower-superexec" in run_srv and "--gpus" not in run_srv and "--superexec-auth-secret-file req/secret" in run_srv
    assert srv.files["~/req/secret"] == bytes(range(32))  # raw bytes, unchanged
    assert not (d.state_dir / "secret").exists()  # and not left on the local disk
    assert json.loads((d.state_dir / "orchestrated.json").read_text())["id"] == "ab12cd34"


def test_submit_waits_for_the_run_and_refuses_one_that_did_not_complete(tmp_path):
    d = deployment(tmp_path)
    (d.state_dir).mkdir()
    (d.state_dir / "orchestrated.json").write_text(json.dumps({"id": "ab12cd34", "endpoint": "e"}))
    ok = FakeOrchestrator(runs=[{"run-id": 7, "status": "finished:completed"}])
    assert OrchestratedFederation(d, ok).submit(tmp_path, {"backend": "stub", "num-server-rounds": 2}) == "7"
    post = next(c for c in ok.calls if c[1].endswith("/run"))
    assert post[2]["run_config"] == 'backend="stub" num-server-rounds=2' and post[2]["app"] == ok.uploads[0]
    bad = FakeOrchestrator(runs=[{"run-id": 7, "status": "finished:failed", "status-details": "boom"}])
    with pytest.raises(FederationError, match="finished:failed"):
        OrchestratedFederation(d, bad).submit(tmp_path, {})


def test_stop_removes_containers_and_the_request(tmp_path):
    d, orch = deployment(tmp_path), FakeOrchestrator()
    d.state_dir.mkdir()
    (d.state_dir / "orchestrated.json").write_text(json.dumps({"id": "ab12cd34", "endpoint": "e"}))
    OrchestratedFederation(d, orch).stop()
    assert ("DELETE", "/requests/ab12cd34", None) in orch.calls
    assert all(f"docker rm -f {NODE} {EXEC}" in s.host.commands[-1] for s in d.sites)
    assert not (d.state_dir / "orchestrated.json").exists()


def test_prepare_node_installs_the_toolkit_only_for_gpu_nodes_and_records_the_pull(tmp_path):
    d = deployment(tmp_path)
    fed = OrchestratedFederation(d, FakeOrchestrator())
    fed.prepare_node(d.server)
    fed.prepare_node(d.clients[0])
    assert not any("nvidia-container-toolkit" in c for c in d.server.host.commands)
    assert any("nvidia-container-toolkit" in c for c in d.clients[0].host.commands)
    pulls = [json.loads(line) for line in (d.state_dir / "pulls.jsonl").read_text().splitlines()]
    assert [p["host"] for p in pulls] == ["srv", "c1"] and pulls[0]["image"] == d.cfg.image_cpu and pulls[1]["image"] == d.cfg.image_gpu
    assert pulls[0]["size_bytes"] == 123456


def test_run_in_image_runs_once_in_the_nodes_image_with_the_home_mounted(tmp_path):
    d = deployment(tmp_path)
    OrchestratedFederation(d, FakeOrchestrator()).run_in_image(d.clients[0], "python -m x", {"A": "1"}, "/home/u/app")
    cmd = next(c for c in d.clients[0].host.commands if "docker run" in c)
    assert "docker run --rm" in cmd and "-w /home/u/app" in cmd and "-v /home/u:/home/u" in cmd and "A=1" in cmd and cmd.rstrip().endswith("python -m x")
