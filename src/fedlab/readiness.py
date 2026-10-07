"""Wait until a freshly created node has finished first-boot provisioning."""

from __future__ import annotations

import time
from typing import Callable

from .remote import Host


def wait_ready(host: Host, gpu: bool, timeout: float = 1500, log: Callable[[str], None] = print) -> None:
    """Block until cloud-init is done and, for GPU nodes, the NVIDIA driver works (after its one reboot).

    SSH errors are retried, because a GPU node reboots once during provisioning.
    """
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        code, out = host.execute("cloud-init status --wait >/dev/null 2>&1; echo cloud-init=$?")
        done = "cloud-init=0" in out or "cloud-init=2" in out  # 2 = finished with recoverable errors
        if done and not gpu:
            return
        if done:
            code, out = host.execute("nvidia-smi -L && ! test -e /run/systemd/shutdown/scheduled")
            if code == 0:
                return
            last = "waiting for the NVIDIA driver (the node reboots once)"
        else:
            last = out.strip().splitlines()[-1] if out.strip() else "no SSH yet"
        log(f"{host.name}: not ready yet ({last})")
        time.sleep(10)
    raise TimeoutError(f"{host.name} was not ready after {timeout:.0f}s ({last})")
