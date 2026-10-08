"""Wait until a freshly created node has finished first-boot provisioning."""

from __future__ import annotations

import time
from typing import Callable

from .remote import Host


def wait_ready(host: Host, gpu: bool, timeout: float = 1500, log: Callable[[str], None] = print) -> None:
    """Block until cloud-init is done and, for GPU nodes, the NVIDIA driver works (after its one reboot).

    SSH errors are retried, because a GPU node reboots once during provisioning.
    """
    started = time.monotonic()
    deadline = started + timeout
    last, logged, last_logged = "", "", float("-inf")
    while time.monotonic() < deadline:
        code, out = host.execute("cloud-init status --wait >/dev/null 2>&1; echo cloud-init=$?")
        done = "cloud-init=0" in out or "cloud-init=2" in out  # 2 = finished with recoverable errors
        if done and not gpu:
            return
        if done:
            code, out = host.execute(
                "if test -e /var/lib/fedlab-driver-failed; then echo DRIVER_FAILED; exit 3; fi\n"
                # The kernel module is loaded once /proc/driver/nvidia exists (nvidia-smi may be absent).\n                "test -e /proc/driver/nvidia/version && ! test -e /run/systemd/shutdown/scheduled"
            )
            if code == 0:
                return
            if "DRIVER_FAILED" in out:
                raise RuntimeError(
                    f"{host.name}: the NVIDIA driver install failed. "
                    f"See /var/log/cloud-init-output.log on the node (`fedlab ssh {host.name}`)."
                )
            last = "waiting for the NVIDIA driver (the node reboots once)"
        else:
            last = out.strip().splitlines()[-1] if out.strip() else "no SSH yet"
        now = time.monotonic()
        if last != logged or now - last_logged >= 60:  # log changes at once, repeats once a minute
            log(f"{host.name}: not ready yet after {now - started:.0f}s ({last})")
            logged, last_logged = last, now
        time.sleep(10)
    raise TimeoutError(f"{host.name} was not ready after {timeout:.0f}s ({last})")
