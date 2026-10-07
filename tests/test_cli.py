from typer.testing import CliRunner

from fedlab.cli import app


def test_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("check", "up", "start", "stop", "status", "ssh", "destroy"):
        assert cmd in r.output


def test_check_passes_against_moto_without_creating_anything(tmp_path, monkeypatch):
    import boto3
    from moto import mock_aws

    from fedlab.providers.aws import AwsProvider

    monkeypatch.setattr(AwsProvider, "_ami", lambda self, ec2: ec2.describe_images()["Images"][0]["ImageId"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("nodes:\n  - {provider: aws, location: us-east-1}\n")
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
        "nodes:\n  - {provider: aws, location: us-east-1, machine_type: nope.huge}\n"
    )
    with mock_aws():
        r = CliRunner().invoke(app, ["check"])
    assert r.exit_code == 1 and "FAIL" in r.output
