from pathlib import Path

import pytest

from fedlab import ssh
from fedlab.remote import LocalHost, RemoteError, SshHost


def test_local_host_runs_commands_moves_files_and_manages_services(tmp_path):
    host = LocalHost("n1", tmp_path / "n1")
    assert host.home == str((tmp_path / "n1").resolve())
    assert host.run("echo hi").strip() == "hi"
    with pytest.raises(RemoteError, match="exit 3"):
        host.run("echo boom >&2; exit 3")
    assert host.execute("exit 4")[0] == 4

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.txt").write_text("a")
    host.put(src, "~/data")
    assert (tmp_path / "n1" / "data" / "sub" / "a.txt").read_text() == "a"
    host.put_text("hello\n", "~/notes/n.txt")
    host.get("~/notes", tmp_path / "back")
    assert (tmp_path / "back" / "n.txt").read_text() == "hello\n"

    host.start_service("sleeper", "sleep 30", {})
    pid = int((tmp_path / "n1" / ".services" / "sleeper.pid").read_text())
    host.stop_service("sleeper")
    with pytest.raises(ProcessLookupError):
        import os, time

        time.sleep(0.2)
        os.kill(pid, 0)


def test_ssh_host_installs_a_systemd_unit_that_runs_a_script(tmp_path, monkeypatch):
    key = ssh.Keypair(Path("/k"), "ssh-ed25519 AAA")
    host = SshHost("n", "1.2.3.4", "ubuntu", key)
    commands = []
    monkeypatch.setattr(host, "run", lambda cmd, **kw: commands.append(cmd) or "/home/ubuntu\n")
    host.start_service("superlink", "flower-superlink --x '$HOME/%h'", {"PATH": "/venv/bin:/usr/bin", "A": "it's"})
    script_cmd, unit_cmd = commands[1], commands[2]
    assert "export PATH=/venv/bin:/usr/bin" in script_cmd and "export A='it'\"'\"'s'" in script_cmd
    assert "exec flower-superlink --x '$HOME/%h'" in script_cmd  # the command never reaches systemd's own parser
    assert "ExecStart=/bin/bash /home/ubuntu/.fedlab/svc/superlink.sh" in unit_cmd
    assert "User=ubuntu" in unit_cmd and "systemctl restart fedlab-superlink" in unit_cmd
