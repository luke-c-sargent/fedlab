from typer.testing import CliRunner

from fedlab.cli import app


def test_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("check", "ping", "modules", "deploy", "run", "collect", "experiment", "up", "start", "stop", "status", "ssh", "destroy"):
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


def test_modules_lists_the_registry():
    r = CliRunner().invoke(app, ["modules"])
    assert r.exit_code == 0 and "smoke" in r.output and "compass_tcga_gtex" in r.output


def _experiment_env(monkeypatch, tmp_path, fail_at=None):
    """Replace the cloud/module pieces of `experiment` with recorders."""
    from fedlab import cli

    calls = []

    class FakeModule:
        def validate(self, nodes):
            return []

        def stage(self, d):
            calls.append("stage")

        def start(self, d):
            calls.append("start")

        def run(self, d):
            calls.append("run")
            if fail_at == "run":
                raise RuntimeError("training failed")
            return "1"

        def collect(self, d, dest):
            calls.append("collect")
            return []

        def stop(self, d):
            calls.append("stop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("module: smoke\n")
    monkeypatch.setattr(cli, "get_module", lambda cfg: FakeModule())
    monkeypatch.setattr(cli, "_up", lambda ctx: calls.append("up"))
    monkeypatch.setattr(cli, "_deployment", lambda ctx: object())
    monkeypatch.setattr(cli, "_wait_ready", lambda d: calls.append("ready"))
    monkeypatch.setattr(cli, "_destroy", lambda ctx, yes=False, dry_run=False: calls.append("destroy"))
    return calls


def test_experiment_destroys_only_after_collecting(tmp_path, monkeypatch):
    calls = _experiment_env(monkeypatch, tmp_path)
    r = CliRunner().invoke(app, ["experiment", "--yes"])
    assert r.exit_code == 0, r.output
    assert calls == ["up", "ready", "stage", "start", "run", "collect", "stop", "destroy"]


def test_experiment_keeps_the_fleet_when_the_run_fails(tmp_path, monkeypatch):
    calls = _experiment_env(monkeypatch, tmp_path, fail_at="run")
    r = CliRunner().invoke(app, ["experiment", "--yes"])
    assert r.exit_code == 1
    assert "destroy" not in calls and calls.count("collect") == 1  # still fetched the logs
    assert "still running" in r.output


def test_experiment_can_destroy_after_failure_when_asked(tmp_path, monkeypatch):
    calls = _experiment_env(monkeypatch, tmp_path, fail_at="run")
    r = CliRunner().invoke(app, ["experiment", "--yes", "--destroy-on-failure"])
    assert r.exit_code == 1 and calls[-1] == "destroy"


def test_prepare_explains_a_python_without_pandas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "module: compass_tcga_gtex\nmodule_options:\n  compass_repo: /x\n  tcga_tsv: /x/t\n  gtex_tsv: /x/g\n  prep_python: /usr/bin/false\n"
    )
    r = CliRunner().invoke(app, ["prepare"])
    assert r.exit_code == 1 and "cannot import numpy and pandas" in r.output and "prep_python" in r.output


def test_check_reports_which_zones_sell_the_instance_type(tmp_path, monkeypatch):
    from moto import mock_aws

    from fedlab.providers.aws import AwsProvider

    monkeypatch.setattr(AwsProvider, "_ami", lambda self, ec2, gpu=False: ec2.describe_images()["Images"][0]["ImageId"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("nodes:\n  - {provider: aws, location: us-east-1, role: server, machine_type: m5.xlarge}\n")
    with mock_aws():
        r = CliRunner().invoke(app, ["check"])
    assert r.exit_code == 0, r.output
    text = " ".join(r.output.split())  # rich wraps long lines
    assert "machine type m5.xlarge in us-east-1" in text


class _FakeCloud:
    """destroy(): the plan comes from dry runs; a real run logs progress and (optionally) leaves something behind."""

    def __init__(self, leaves=False):
        self.items, self.leaves = ["[aws r] delete instance a"], leaves

    def destroy(self, nodes, dry_run=False, log=None):
        if not dry_run:
            log("[aws r] terminating 1 instance(s)")
            if not self.leaves:
                self.items = []
        return list(self.items)


def _destroy_env(monkeypatch, tmp_path, leaves):
    from fedlab import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("nodes:\n  - {provider: aws, location: us-east-1, role: server}\n")
    cloud = _FakeCloud(leaves)
    monkeypatch.setattr(cli, "get_provider", lambda *a: cloud)


def test_destroy_shows_phases_progress_and_verifies(tmp_path, monkeypatch):
    _destroy_env(monkeypatch, tmp_path, leaves=False)
    r = CliRunner().invoke(app, ["destroy", "--yes"])
    assert r.exit_code == 0, r.output
    assert "[aws r] delete instance a" in r.output  # the plan keeps its [aws region] prefix
    assert "1/2" in r.output and "2/2" in r.output and "terminating 1 instance(s)" in r.output
    assert "nothing tagged for 'fedlearn' remains" in r.output


def test_destroy_fails_loudly_if_something_is_left(tmp_path, monkeypatch):
    _destroy_env(monkeypatch, tmp_path, leaves=True)
    r = CliRunner().invoke(app, ["destroy", "--yes"])
    assert r.exit_code == 1 and "these remain" in r.output and "delete instance a" in r.output
