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
        NodeInfo("fedlearn-aws-us-east-1", "aws", "us-east-1", "running", "1.2.3.4"),
        NodeInfo("fedlearn-aws-eu-west-1", "aws", "eu-west-1", "stopped", None),
    ]
    text = ssh.render_ssh_config(cfg, infos, key)
    assert "Host fedlearn-aws-us-east-1" in text and "HostName 1.2.3.4" in text
    assert "eu-west-1" not in text
    data = json.loads(ssh.write_nodes_json(cfg, infos, key).read_text())
    assert data["fedlearn-aws-us-east-1"]["flower"]["fleet_api"] == "1.2.3.4:9092"
    assert data["fedlearn-aws-eu-west-1"]["flower"] is None
