import pytest

from fedlab.config import Settings, load_settings


def test_defaults_give_one_server_and_two_gpu_clients(cfg):
    nodes = cfg.resolved_nodes()
    assert [(n.name, n.role) for n in nodes] == [
        ("fedlearn-server-aws-eu-west-1", "server"),
        ("fedlearn-client-aws-us-east-1", "client"),
        ("fedlearn-client-gcp-us-central1-a", "client"),
    ]
    assert [n.machine_type for n in nodes] == ["r6i.large", "g4dn.4xlarge", "n1-highmem-8"]
    assert [n.gpu for n in nodes] == [False, True, True]
    assert nodes[2].accelerator == "nvidia-tesla-t4"
    assert all(n.disk_gb == 200 for n in nodes)


def test_exactly_one_server_required():
    for nodes in ([], [{"provider": "aws", "location": "us-east-1"}], [{"provider": "aws", "location": "a", "role": "server"}, {"provider": "aws", "location": "b", "role": "server"}]):
        with pytest.raises(ValueError, match="exactly one"):
            Settings(nodes=nodes).resolved_nodes()


def test_yaml_and_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(
        "name_prefix: lab\nnodes:\n  - {provider: aws, location: us-west-2, role: server, machine_type: m5.2xlarge, disk_gb: 500}\n"
    )
    monkeypatch.setenv("FEDLAB_DISK_GB", "100")
    monkeypatch.setenv("GCP_PROJECT", "proj")
    s = load_settings(tmp_path / "c.yaml")
    (n,) = s.resolved_nodes()
    assert (n.name, n.machine_type, n.disk_gb) == ("lab-server-aws-us-west-2", "m5.2xlarge", 500)
    assert s.disk_gb == 100 and s.gcp_project == "proj"


def test_ports_always_include_ssh():
    assert Settings(ports=[8000]).open_ports == [22, 8000]


def test_invalid_run_name():
    with pytest.raises(ValueError):
        Settings(run_name="Bad_Name")


def test_cloud_init_installs_nvidia_driver_only_when_asked(tmp_path):
    import subprocess

    import yaml

    from fedlab import cloud_init

    assert "ubuntu-drivers" not in cloud_init.render("ubuntu")
    text = cloud_init.render("ubuntu", install_nvidia_driver=True)
    doc = yaml.safe_load(text)  # must stay valid cloud-config
    assert "ubuntu-drivers-common" in doc["packages"]  # the GCP image does not ship the tool
    files = {f["path"]: f["content"] for f in doc["write_files"]}
    script = files["/usr/local/sbin/fedlab-gpu-driver.sh"]
    assert "Acquire::Retries" in files["/etc/apt/apt.conf.d/99fedlab"]
    assert script.index("archive.ubuntu.com/ubuntu|g") < script.index("apt-get update") < script.index("ubuntu-drivers install --gpgpu")
    assert script.index("ubuntu-drivers install --gpgpu") < script.index("nvidia-utils-") < script.index("shutdown -r")
    assert "fedlab-driver-failed" in script
    commands = [c[2] for c in doc["runcmd"]]
    assert commands.index("/usr/local/sbin/fedlab-gpu-driver.sh") < commands.index("touch /var/lib/fedlab-ready")
    path = tmp_path / "gpu.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0  # valid shell
    plain = yaml.safe_load(cloud_init.render("ubuntu"))
    assert [f["path"] for f in plain["write_files"]] == ["/etc/apt/apt.conf.d/99fedlab"]  # apt settings on every node
    assert "ubuntu-drivers-common" not in plain["packages"]


def test_fallback_locations_are_for_gcp_and_must_all_differ():
    import pytest
    from pydantic import ValidationError

    from fedlab.config import NodeSpec

    ok = NodeSpec(provider="gcp", location="us-central1-a", fallback_locations=["us-central1-b"])
    assert ok.fallback_locations == ["us-central1-b"]
    with pytest.raises(ValidationError, match="only supported for GCP"):
        NodeSpec(provider="aws", location="us-west-2", fallback_locations=["us-west-1"])
    with pytest.raises(ValidationError, match="must all differ"):
        NodeSpec(provider="gcp", location="us-central1-a", fallback_locations=["us-central1-b", "us-central1-a"])


def test_a_resolved_node_lists_its_zones_and_keeps_its_name(tmp_path):
    from fedlab.config import NodeSpec, Settings

    cfg = Settings(
        state_dir=tmp_path,
        nodes=[
            NodeSpec(role="server", provider="aws", location="eu-west-1"),
            NodeSpec(role="client", site="gtex", provider="gcp", location="us-central1-a", fallback_locations=["us-central1-b"]),
        ],
    )
    gcp = cfg.resolved_nodes()[1]
    assert gcp.zones == ("us-central1-a", "us-central1-b") and gcp.name.endswith("-gcp-us-central1-a")
    assert cfg.resolved_nodes()[0].zones == ("eu-west-1",)
