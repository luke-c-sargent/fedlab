"""Wait until a freshly created node has finished first-boot provisioning."""

from __future__ import annotations

import re
import time
from typing import Callable

from .remote import Host

BOOT_STATUS = (
    "cloud-init status 2>&1 | head -1\n"
    'echo "last: $(sudo tail -n 1 /var/log/cloud-init-output.log 2>/dev/null | cut -c1-120)"'
)


def boot_state(host: Host) -> tuple[str, str]:
    """(state, last log line): state is `done`, `running`, `error`, or `no SSH yet`. Never blocks for long."""
    code, out = host.execute(BOOT_STATUS)
    match = re.search(r"status: ([\w ]+)", out)
    last = (re.search(r"^last: (.*)$", out, re.M) or [None, ""])[1]
    if not match:
        return "no SSH yet", ""
    status = match.group(1).strip()
    state = "done" if "done" in status else "error" if "error" in status else "running"
    return state, last


def wait_ready(host: Host, gpu: bool, timeout: float = 1500, log: Callable[[str], None] = print) -> None:
    """Block until cloud-init is done and, for GPU nodes, the NVIDIA driver is loaded (after its one reboot).

    SSH errors are retried, because a GPU node reboots once during provisioning. Progress is logged when
    it changes and at least once a minute, with the last line of the node's cloud-init log.
    """
    started = time.monotonic()
    deadline = started + timeout
    last, logged, last_logged = "", "", float("-inf")
    while time.monotonic() < deadline:
        state, tail = boot_state(host)
        if state == "error":
            raise RuntimeError(f"{host.name}: cloud-init failed. See /var/log/cloud-init-output.log on the node.")
        if state == "done" and not gpu:
            return
        if state == "done":
            code, out = host.execute(
                "if test -e /var/lib/fedlab-driver-failed; then echo DRIVER_FAILED; exit 3; fi\n"
                # The kernel module is loaded once /proc/driver/nvidia exists (nvidia-smi may be absent).
                "test -e /proc/driver/nvidia/version && ! test -e /run/systemd/shutdown/scheduled"
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
            last = f"cloud-init {state}" + (f": {tail}" if tail else "")
        now = time.monotonic()
        if last != logged or now - last_logged >= 60:  # log changes at once, repeats once a minute
            log(f"{host.name}: not ready yet after {now - started:.0f}s ({last})")
            logged, last_logged = last, now
        time.sleep(10)
    raise TimeoutError(f"{host.name} was not ready after {timeout:.0f}s ({last})")
