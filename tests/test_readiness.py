import pytest

from fedlab import readiness
from fedlab.remote import Host


class ScriptedHost(Host):
    """execute() returns the next scripted (code, output) for each call; everything else is unused."""

    address = "1.2.3.4"

    def __init__(self, replies):
        self.name, self.replies, self.calls = "node", list(replies), 0

    def _exec(self, script, on_line):
        self.calls += 1
        return self.replies.pop(0)

    put = get = start_service = stop_service = service_log = lambda *a, **k: None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(readiness.time, "sleep", lambda s: clock.update(t=clock["t"] + s))
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock["t"])


DONE = (0, "cloud-init=0\n")


def test_cpu_node_is_ready_once_cloud_init_is_done():
    host = ScriptedHost([(255, ""), DONE])  # first call: SSH not up yet
    readiness.wait_ready(host, gpu=False, log=lambda m: None)
    assert host.calls == 2


def test_gpu_node_waits_for_the_driver_and_logs_sparingly():
    messages = []
    host = ScriptedHost([DONE, (1, ""), DONE, (1, ""), DONE, (0, "GPU 0: Tesla T4\n")])
    readiness.wait_ready(host, gpu=True, log=messages.append)
    assert len(messages) == 1 and "NVIDIA driver" in messages[0]  # the same message is not repeated every 10 s


def test_gpu_node_fails_fast_when_the_driver_install_failed():
    host = ScriptedHost([DONE, (3, "DRIVER_FAILED\n")])
    with pytest.raises(RuntimeError, match="driver install failed"):
        readiness.wait_ready(host, gpu=True, log=lambda m: None)


def test_times_out_with_the_last_reason():
    host = ScriptedHost([DONE, (1, "")] * 100)
    with pytest.raises(TimeoutError, match="NVIDIA driver"):
        readiness.wait_ready(host, gpu=True, timeout=60, log=lambda m: None)
