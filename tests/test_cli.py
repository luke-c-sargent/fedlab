from typer.testing import CliRunner

from fedlab.cli import app


def test_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("up", "start", "stop", "status", "ssh", "destroy"):
        assert cmd in r.output
