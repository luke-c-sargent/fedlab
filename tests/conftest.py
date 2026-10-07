import pytest

from fedlab.config import Settings


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Settings(state_dir=tmp_path / ".fedlab", ssh_config_path=tmp_path / "ssh_fedlab")
