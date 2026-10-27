import json

from fedlab.config import Settings
from fedlab.orchestrator import Orchestrator, summarize_traffic
from fedlab.ssh import Keypair

LOG = """2026-10-27T01:00:00+00:00 ab12cd34.1.2.3.4.sslip.io POST /v1/fleet/pull-messages 200 1200 480 0.020
2026-10-27T01:00:01+00:00 ab12cd34.1.2.3.4.sslip.io POST /v1/fleet/push-messages 200 9000000 300 1.500
2026-10-27T01:00:02+00:00 ab12cd34.1.2.3.4.sslip.io POST /v1/runtime/pull-message 200 800 5000000 0.300
2026-10-27T01:00:03+00:00 ee55ee55.1.2.3.4.sslip.io POST /v1/fleet/pull-messages 200 1000 400 0.010
2026-10-27T01:00:04+00:00 - GET / 403 90 150 0.000
not a log line
"""


def test_traffic_is_totalled_per_request_and_per_api():
    t = summarize_traffic(LOG)
    assert t["ab12cd34"]["bytes_in"] == 1200 + 9000000 + 800 and t["ab12cd34"]["bytes_out"] == 480 + 300 + 5000000
    assert t["ab12cd34"]["by_api"]["fleet"]["requests"] == 2 and t["ab12cd34"]["by_api"]["runtime"]["requests"] == 1
    assert t["ee55ee55"]["requests"] == 1 and t["-"]["by_api"]["other"]["requests"] == 1
    assert t["total"]["requests"] == 5


def test_terraform_variables_pin_the_flower_version_and_the_callers_address(tmp_path, monkeypatch):
    monkeypatch.setattr("fedlab.orchestrator.my_ip_cidr", lambda: "9.9.9.9/32")
    orch = Orchestrator(Settings(state_dir=tmp_path, run_name="t1"), Keypair(tmp_path / "k", "ssh-ed25519 AAAA"))
    v = orch._variables()
    assert v["my_ip_cidr"] == "9.9.9.9/32" and v["name"] == "t1" and v["ssh_public_key"] == "ssh-ed25519 AAAA"
    from importlib.metadata import version

    assert v["flwr_version"] == version("flwr") and v["auto_terminate_hours"] == 4
    assert json.dumps(v)  # serializable for -var-file
