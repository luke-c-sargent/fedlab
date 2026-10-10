import json
import stat

from fedlab import ssh
from fedlab.providers import NodeInfo


def test_keypair_generated_once_with_private_perms(cfg):
    k1 = ssh.ensure_keypair(cfg)
    k2 = ssh.ensure_keypair(cfg)
    assert k1.public_key == k2.public_key and k1.public_key.startswith("ssh-ed25519 ")
    assert stat.S_IMODE(k1.private_path.stat().st_mode) == 0o600


def test_ssh_config_and_nodes_json(cfg):
    key = ssh.ensure_keypair(cfg)
    infos = [
        NodeInfo("fedlearn-client-aws-us-east-1", "aws", "us-east-1", "running", "1.2.3.4"),
        NodeInfo("fedlearn-server-aws-eu-west-1", "aws", "eu-west-1", "running", "5.6.7.8"),
        NodeInfo("fedlearn-client-gcp-us-central1-a", "gcp", "us-central1-a", "stopped", None),
    ]
    text = ssh.render_ssh_config(cfg, infos, key)
    assert "Host fedlearn-client-aws-us-east-1" in text and "HostName 1.2.3.4" in text
    assert "us-central1-a" not in text
    data = json.loads(ssh.write_nodes_json(cfg, infos, key).read_text())
    server = data["fedlearn-server-aws-eu-west-1"]
    assert server["role"] == "server" and server["flower"]["fleet_api"] == "5.6.7.8:9092"
    assert server["ports"] == [22, 9092]
    assert data["fedlearn-client-aws-us-east-1"]["role"] == "client"
    assert data["fedlearn-client-aws-us-east-1"]["ports"] == [22]
    assert data["fedlearn-client-aws-us-east-1"]["flower"] is None  # clients don't serve Flower APIs


def test_wait_for_login_retries_until_ssh_works(monkeypatch):
    attempts = []

    def fake_probe(host, user, key, timeout=10, proxy="auto"):
        attempts.append(host)
        return (len(attempts) >= 3, "")

    monkeypatch.setattr(ssh, "probe_ssh", fake_probe)
    monkeypatch.setattr(ssh.time, "sleep", lambda s: None)
    assert ssh.wait_for_login("1.2.3.4", "ubuntu", None, timeout=300, interval=5) is True
    assert attempts == ["1.2.3.4"] * 3


def test_wait_for_login_gives_up_at_the_deadline(monkeypatch):
    clock = iter(range(0, 1000, 100))  # each probe "takes" 100 s
    monkeypatch.setattr(ssh, "probe_ssh", lambda *a, **k: (False, "refused"))
    monkeypatch.setattr(ssh.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ssh.time, "sleep", lambda s: None)
    assert ssh.wait_for_login("1.2.3.4", "ubuntu", None, timeout=250, interval=5) is False


def test_probe_ssh_leaves_the_users_ssh_config_alone(cfg, monkeypatch):
    key = ssh.ensure_keypair(cfg)
    seen = []
    monkeypatch.setattr(ssh.subprocess, "run", lambda cmd, **k: seen.append(cmd) or type("R", (), {"returncode": 0, "stderr": ""})())
    assert ssh.probe_ssh("1.2.3.4", "ubuntu", key)[0] is True
    assert "-F" not in seen[0] and "ProxyJump=none" not in seen[0]  # ambient config applies
    assert "IdentitiesOnly=yes" in seen[0] and "ubuntu@1.2.3.4" in seen[0] and str(key.private_path) in seen[0]


def test_ssh_host_and_rsync_leave_the_users_ssh_config_alone(cfg, monkeypatch):
    from fedlab import remote

    key = ssh.ensure_keypair(cfg)
    host = remote.SshHost("n", "1.2.3.4", "ubuntu", key)
    assert "-F" not in host._ssh and "ProxyJump=none" not in host._ssh
    seen = []
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd, **k: seen.append(cmd) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    host._rsync("a", "b")
    inner = seen[0][seen[0].index("-e") + 1]
    assert inner.startswith("ssh -i ") and "-F" not in inner.split() and "ProxyJump" not in inner
