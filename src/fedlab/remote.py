"""Run commands, move files, and manage background services on a fleet node.

`SshHost` drives a real VM (services are systemd units). `LocalHost` fakes a node with a
directory on this machine, so a whole module lifecycle can run in tests without a cloud.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from .ssh import SSH_BASE_OPTS, Keypair

SSH_OPTS = [*SSH_BASE_OPTS, "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15"]


# ssh errors from before the remote command starts. Nothing ran on the node, so a retry is safe.
# (Errors from a session that already started, such as "Broken pipe", are deliberately not listed.)
TRANSIENT_SSH = (
    "timed out during banner exchange", "Connection timed out", "Connection refused",
    "kex_exchange_identification", "Connection reset by", "Connection closed by",
)
SSH_ATTEMPTS = 4


def is_transient_ssh_failure(returncode: int, output: str) -> bool:
    return returncode in (255, 12, 23) and any(pattern in output for pattern in TRANSIENT_SSH)


class RemoteError(RuntimeError):
    def __init__(self, host: str, command: str, returncode: int, output: str):
        tail = "\n".join(output.strip().splitlines()[-15:])
        super().__init__(f"[{host}] exit {returncode}: {command}\n{tail}")
        self.returncode = returncode
        self.output = output


def _script(command: str, env: dict[str, str] | None) -> str:
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in (env or {}).items())
    return f"set -eo pipefail\n{exports}{command}\n"


class Host(ABC):
    name: str
    address: str  # where other nodes (and TLS certificates) reach this host
    shares_machine = False  # True when several hosts run on one machine (ports must differ)
    _home: str | None = None

    @property
    def home(self) -> str:
        if self._home is None:
            self._home = self.run("echo $HOME").strip()
        return self._home

    @abstractmethod
    def _exec(self, script: str, on_line: Callable[[str], None] | None) -> tuple[int, str]:
        """Run a bash script; return (exit code, combined output). Calls `on_line` per line if given."""

    @abstractmethod
    def put(self, local: Path, remote: str) -> None:
        """Copy a file, or the contents of a directory, to `remote` (a path on the host)."""

    @abstractmethod
    def get(self, remote: str, local: Path) -> None:
        """Copy a remote file, or the contents of a remote directory, to `local`."""

    @abstractmethod
    def start_service(self, name: str, command: str, env: dict[str, str]) -> None:
        """(Re)start a background service that survives this connection."""

    @abstractmethod
    def stop_service(self, name: str) -> None: ...

    @abstractmethod
    def service_log(self, name: str, lines: int = 60) -> str: ...

    def execute(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        """Run a shell command (bash, `set -eo pipefail`); return (exit code, combined output)."""
        return self._exec(_script(command, env), on_line)

    def run(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        check: bool = True,
        on_line: Callable[[str], None] | None = None,
    ) -> str:
        """Like `execute`, but raise `RemoteError` on a non-zero exit (when `check`) and return the output."""
        code, output = self.execute(command, env=env, on_line=on_line)
        if check and code != 0:
            raise RemoteError(self.name, command, code, output)
        return output

    def put_text(self, text: str, remote: str) -> None:
        remote = self.abspath(remote)
        self.mkdir(os.path.dirname(remote) or ".")
        self.run(f"cat > {shlex.quote(remote)} <<'FEDLAB_EOF'\n{text}FEDLAB_EOF")

    def abspath(self, path: str) -> str:
        """Expand a leading `~` (quoting in shell commands would stop the shell doing it)."""
        return self.home + path[1:] if path == "~" or path.startswith("~/") else path

    def mkdir(self, path: str) -> None:
        self.run(f"mkdir -p {shlex.quote(self.abspath(path))}")


class SshHost(Host):
    def __init__(self, name: str, address: str, user: str, key: Keypair):
        self.name, self.address, self.user, self.key = name, address, user, key

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.address}"

    @property
    def _ssh(self) -> list[str]:
        return ["ssh", "-i", str(self.key.private_path), *SSH_OPTS, self._target]

    def _exec(self, script: str, on_line: Callable[[str], None] | None) -> tuple[int, str]:
        """Run a script, retrying with backoff when the SSH connection itself could not be made."""
        for attempt in range(1, SSH_ATTEMPTS + 1):
            code, output = self._exec_once(script, on_line)
            if attempt == SSH_ATTEMPTS or not is_transient_ssh_failure(code, output):
                return code, output
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _exec_once(self, script: str, on_line: Callable[[str], None] | None) -> tuple[int, str]:
        proc = subprocess.Popen(
            [*self._ssh, "bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(script)
        proc.stdin.close()
        lines: list[str] = []
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
        return proc.wait(), "".join(lines)

    def _rsync(self, src: str, dst: str) -> None:
        ssh = "ssh -i {} {}".format(shlex.quote(str(self.key.private_path)), " ".join(shlex.quote(o) for o in SSH_OPTS))
        for attempt in range(1, SSH_ATTEMPTS + 1):
            r = subprocess.run(["rsync", "-az", "-e", ssh, src, dst], capture_output=True, text=True)
            output = r.stdout + r.stderr
            if not r.returncode:
                return
            if attempt == SSH_ATTEMPTS or not is_transient_ssh_failure(r.returncode, output):
                raise RemoteError(self.name, f"rsync {src} {dst}", r.returncode, output)
            time.sleep(2**attempt)

    def put(self, local: Path, remote: str) -> None:
        remote = self.abspath(remote)
        if local.is_dir():
            self.mkdir(remote)
            self._rsync(f"{local}/", f"{self._target}:{remote}/")
        else:
            self.mkdir(os.path.dirname(remote) or ".")
            self._rsync(str(local), f"{self._target}:{remote}")

    def get(self, remote: str, local: Path) -> None:
        remote = self.abspath(remote)
        local.mkdir(parents=True, exist_ok=True)
        is_dir = self.run(f"test -d {shlex.quote(remote)} && echo dir || echo file").strip() == "dir"
        self._rsync(f"{self._target}:{remote}{'/' if is_dir else ''}", f"{local}/")

    def start_service(self, name: str, command: str, env: dict[str, str]) -> None:
        # The unit only runs a script, so systemd never interprets `$` or `%` in the command.
        home = self.home
        script = f"{home}/.fedlab/svc/{name}.sh"
        body = "#!/bin/bash\n" + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items()) + f"exec {command}\n"
        self.run(f"mkdir -p {home}/.fedlab/svc && cat > {script} <<'SCRIPT'\n{body}SCRIPT\nchmod +x {script}")
        unit = (
            f"[Unit]\nDescription=fedlab {name}\nAfter=network.target\n\n"
            f"[Service]\nUser={self.user}\nExecStart=/bin/bash {script}\nRestart=no\n"
        )
        path = f"/etc/systemd/system/fedlab-{name}.service"
        self.run(
            f"sudo tee {path} >/dev/null <<'UNIT'\n{unit}UNIT\n"
            f"sudo systemctl daemon-reload && sudo systemctl restart fedlab-{name}"
        )

    def stop_service(self, name: str) -> None:
        self.run(f"sudo systemctl stop fedlab-{name} 2>/dev/null || true", check=False)

    def service_log(self, name: str, lines: int = 60) -> str:
        return self.run(f"sudo journalctl -u fedlab-{name} -n {int(lines)} --no-pager 2>&1 || true", check=False)


class LocalHost(Host):
    """A pretend node: `root` is its home directory, services are plain background processes."""

    address = "127.0.0.1"
    shares_machine = True

    def __init__(self, name: str, root: Path):
        self.name, self.root = name, root.resolve()
        self._procs: dict[str, subprocess.Popen] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".services").mkdir(exist_ok=True)

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH"}}
        env["HOME"] = str(self.root)
        return {**env, **(extra or {})}

    def _path(self, remote: str) -> Path:
        return Path(remote.replace("~", str(self.root), 1)) if remote.startswith("~") else Path(remote)

    def _exec(self, script: str, on_line: Callable[[str], None] | None) -> tuple[int, str]:
        proc = subprocess.Popen(
            ["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=self.root, env=self._env(),
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(script)
        proc.stdin.close()
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
        return proc.wait(), "".join(lines)

    def put(self, local: Path, remote: str) -> None:
        dest = self._path(remote)
        if local.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{local}/", f"{dest}/"], check=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", str(local), str(dest)], check=True)

    def get(self, remote: str, local: Path) -> None:
        src = self._path(remote)
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{src}/" if src.is_dir() else str(src), f"{local}/"], check=True)

    def _pidfile(self, name: str) -> Path:
        return self.root / ".services" / f"{name}.pid"

    def start_service(self, name: str, command: str, env: dict[str, str]) -> None:
        self.stop_service(name)
        log = (self.root / ".services" / f"{name}.log").open("w")
        proc = subprocess.Popen(
            ["bash", "-c", command], env=self._env(env), cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._procs[name] = proc
        self._pidfile(name).write_text(str(proc.pid))

    def stop_service(self, name: str) -> None:
        pidfile = self._pidfile(name)
        if not pidfile.exists():
            return
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        proc = self._procs.pop(name, None)
        if proc is not None:
            try:
                proc.wait(timeout=10)  # reap it so it does not linger as a zombie
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        pidfile.unlink(missing_ok=True)
        # Flower's SuperExec children can outlive the SuperNode's process group; on a real VM systemd
        # stops the whole cgroup, so do the equivalent here for anything running from this node's directory.
        subprocess.run(["pkill", "-f", str(self.root)], capture_output=True)

    def service_log(self, name: str, lines: int = 60) -> str:
        path = self.root / ".services" / f"{name}.log"
        return "\n".join(path.read_text().splitlines()[-lines:]) if path.exists() else ""


def echo(prefix: str) -> Callable[[str], None]:
    """An `on_line` callback that prints remote output with a node prefix."""
    return lambda line: print(f"{prefix} {line}", file=sys.stderr, flush=True)
