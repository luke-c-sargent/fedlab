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


def _ssh_host():
    return SshHost("n", "1.2.3.4", "ubuntu", ssh.Keypair(Path("/k"), "ssh-ed25519 AAA"))


def test_ssh_retries_a_connection_that_timed_out_before_the_command_started(monkeypatch):
    host = _ssh_host()
    results = iter([(255, "Connection timed out during banner exchange\n"), (255, "Connection reset by 1.2.3.4 port 22\n"), (0, "ok\n")])
    monkeypatch.setattr(host, "_exec_once", lambda script, on_line: next(results))
    sleeps = []
    monkeypatch.setattr("fedlab.remote.time.sleep", sleeps.append)
    assert host.run("true") == "ok\n"
    assert sleeps == [2, 4]  # backoff between attempts


def test_ssh_does_not_retry_real_failures(monkeypatch):
    host = _ssh_host()
    calls = []
    monkeypatch.setattr(host, "_exec_once", lambda s, o: calls.append(1) or (255, "ubuntu@1.2.3.4: Permission denied (publickey).\n"))
    monkeypatch.setattr("fedlab.remote.time.sleep", lambda s: None)
    with pytest.raises(RemoteError, match="Permission denied"):
        host.run("true")
    calls.clear()
    monkeypatch.setattr(host, "_exec_once", lambda s, o: calls.append(1) or (1, "your command failed\n"))  # exit 1 is the command's own
    with pytest.raises(RemoteError):
        host.run("false")
    assert len(calls) == 1


def test_ssh_gives_up_after_four_attempts(monkeypatch):
    host = _ssh_host()
    calls = []
    monkeypatch.setattr(host, "_exec_once", lambda s, o: calls.append(1) or (255, "Connection timed out during banner exchange\n"))
    monkeypatch.setattr("fedlab.remote.time.sleep", lambda s: None)
    with pytest.raises(RemoteError, match="banner exchange"):
        host.run("true")
    assert len(calls) == 4


def test_rsync_retries_transient_connection_failures(monkeypatch):
    from fedlab import remote

    host = _ssh_host()
    outcomes = iter([(255, "", "ssh: Connection timed out during banner exchange\n"), (0, "", "")])
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd, **k: type("R", (), dict(zip(("returncode", "stdout", "stderr"), next(outcomes))))())
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    host._rsync("a", "b")  # no exception: the second attempt worked


def test_installer_package_listing_is_summarized():
    from fedlab.modules.flower_module import quiet_packages

    out = []
    feed = quiet_packages(out.append)
    for line in ["Resolved 3 packages in 1s", " + numpy==1.26.4", " + torch==2.0.1", " ~ pip==24.0", "Installed 3 packages in 2s", ""]:
        feed(line)
    assert out == ["Resolved 3 packages in 1s", "3 package change(s) listed", "Installed 3 packages in 2s"]


def test_heartbeat_reports_while_a_long_step_runs():
    import time as _time

    from fedlab.flower.federation import heartbeat

    beats = []
    with heartbeat(beats.append, "still working", every=0.05):
        _time.sleep(0.2)
    assert len(beats) >= 2 and beats[0].startswith("still working (0m0")
    count = len(beats)
    _time.sleep(0.15)
    assert len(beats) == count  # it stops with the block
