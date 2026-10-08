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


DONE = (0, "status: done\nlast: finished\n")
RUNNING = (0, "status: running\nlast: Get:7 linux-image-6.8.0\n")


def test_cpu_node_is_ready_once_cloud_init_is_done():
    host = ScriptedHost([(255, ""), RUNNING, DONE])  # SSH not up yet, then still installing, then done
    readiness.wait_ready(host, gpu=False, log=lambda m: None)
    assert host.calls == 3


def test_progress_lines_show_what_the_node_is_doing():
    messages = []
    host = ScriptedHost([RUNNING, RUNNING, RUNNING, DONE])
    readiness.wait_ready(host, gpu=False, log=messages.append)
    assert len(messages) == 1  # repeated identical progress is not repeated
    assert "cloud-init running: Get:7 linux-image-6.8.0" in messages[0]


def test_a_failed_cloud_init_stops_the_wait():
    with pytest.raises(RuntimeError, match="cloud-init failed"):
        readiness.wait_ready(ScriptedHost([(1, "status: error\nlast: boom\n")]), gpu=False, log=lambda m: None)


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
