from typer.testing import CliRunner

from fedlab.cli import app


def test_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("check", "ping", "up", "start", "stop", "status", "ssh", "destroy"):
        assert cmd in r.output


def test_check_passes_against_moto_without_creating_anything(tmp_path, monkeypatch):
    import boto3
    from moto import mock_aws

    from fedlab.providers.aws import AwsProvider

    monkeypatch.setattr(AwsProvider, "_ami", lambda self, ec2, gpu=False: ec2.describe_images()["Images"][0]["ImageId"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("nodes:\n  - {provider: aws, location: us-east-1, role: server}\n")
    with mock_aws():
        r = CliRunner().invoke(app, ["check"])
        assert r.exit_code == 0, r.output
        assert "All checks passed" in r.output
        ec2 = boto3.client("ec2", region_name="us-east-1")
        assert ec2.describe_instances()["Reservations"] == []
        assert not [g for g in ec2.describe_security_groups()["SecurityGroups"] if g["GroupName"].endswith("-sg")]
    assert not (tmp_path / ".fedlab").exists()


def test_check_fails_on_bad_instance_type(tmp_path, monkeypatch):
    from moto import mock_aws

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "nodes:\n  - {provider: aws, location: us-east-1, role: server, machine_type: nope.huge}\n"
    )
    with mock_aws():
        r = CliRunner().invoke(app, ["check"])
    assert r.exit_code == 1 and "FAIL" in r.output


def test_ping_reports_per_node_and_exits_nonzero_on_failure(tmp_path, monkeypatch):
    from fedlab import cli
    from fedlab.providers import NodeInfo

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "nodes:\n  - {provider: aws, location: us-east-1, role: server}\n  - {provider: aws, location: eu-west-1}\n"
    )

    class P:
        def describe(self, n):
            ip = "1.2.3.4" if n.location == "us-east-1" else None
            return NodeInfo(n.name, "aws", n.location, "running" if ip else "stopped", ip)

    monkeypatch.setattr(cli, "get_provider", lambda *a: P())
    monkeypatch.setattr(cli.sshmod, "probe_ssh", lambda host, user, key, **kw: (True, "12 ms"))
    r = CliRunner().invoke(app, ["ping"])
    assert r.exit_code == 1
    assert "ok" in r.output and "1.2.3.4 12 ms" in r.output and "no public IP (state: stopped)" in r.output
