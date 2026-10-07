import pytest

from fedlab.config import Settings, load_settings


def test_defaults_give_three_named_nodes(cfg):
    nodes = cfg.resolved_nodes()
    assert [n.name for n in nodes] == [
        "fedlearn-aws-us-east-1",
        "fedlearn-aws-eu-west-1",
        "fedlearn-gcp-us-central1-a",
    ]
    assert [n.machine_type for n in nodes] == ["m5.xlarge", "m5.xlarge", "n2-standard-4"]
    assert all(n.disk_gb == 200 for n in nodes)


def test_yaml_and_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(
        "name_prefix: lab\nnodes:\n  - {provider: aws, location: us-west-2, machine_type: m5.2xlarge, disk_gb: 500}\n"
    )
    monkeypatch.setenv("FEDLAB_DISK_GB", "100")
    monkeypatch.setenv("GCP_PROJECT", "proj")
    s = load_settings(tmp_path / "c.yaml")
    (n,) = s.resolved_nodes()
    assert (n.name, n.machine_type, n.disk_gb) == ("lab-aws-us-west-2", "m5.2xlarge", 500)
    assert s.disk_gb == 100 and s.gcp_project == "proj"


def test_ports_always_include_ssh():
    assert Settings(ports=[9092]).open_ports == [22, 9092]


def test_invalid_run_name():
    with pytest.raises(ValueError):
        Settings(run_name="Bad_Name")
