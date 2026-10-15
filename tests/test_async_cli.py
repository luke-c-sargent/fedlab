import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from fedlab_async.cli import app
from fedlab_async.envelope import peek, read
from fedlab_async.run import RunDir
from test_async_strategy import MomentumFedAvg, in_memory


@pytest.fixture
def cli(capsys):
    def invoke(*args):
        capsys.readouterr()
        try:
            app(args=[str(a) for a in args])
            code = 0
        except SystemExit as exit_:
            code = exit_.code or 0
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


def write(path: Path, value) -> Path:
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.fixture
def plain_run(tmp_path, cli):
    config = write(tmp_path / "run.yaml", {"program": "cli_programs:build", "options": {"rounds": 8}})
    code, out, _ = cli("init", tmp_path / "run", "--config", config)
    assert code == 0 and "created run" in out
    sites = {n: write(tmp_path / f"site{n}.yaml", {"node_id": n}) for n in (1, 2, 3)}
    return tmp_path / "run", sites


def client_args(run, site, tmp_path):
    return ("client", "--exchange", run / "exchange", "--state-dir", tmp_path / f"state{site['n']}", "--site-file", site["file"],
            "--client", "test_async_strategy:client_fn")


def drive_one_step(cli, run, sites, tmp_path, explicit_for=()):
    code, out, _ = cli("emit", run)
    assert code == 0
    files = [Path(line) for line in out.splitlines()]
    for node, site in sites.items():
        args = client_args(run, {"n": node, "file": site}, tmp_path)
        if node in explicit_for:
            args += tuple(f for f in files if f"-n{node}." in f.name)
        code, out, _ = cli(*args)
        assert code == 0 and "wrote" in out
    return cli("collect", run)


def test_a_plain_run_through_the_cli_matches_an_in_memory_run(tmp_path, cli, plain_run):
    run, sites = plain_run
    rounds = 0
    while "finished" not in cli("status", run)[1]:
        code, out, _ = drive_one_step(cli, run, sites, tmp_path, explicit_for=(1,) if rounds % 2 else ())
        assert code == 0 and "advanced" in out
        rounds += 1
    state = pickle.loads((run / "private" / "state.pkl").read_bytes())
    reference = in_memory(8)
    np.testing.assert_allclose(state.program.strategy.w, reference.w)
    assert rounds == len(reference.rounds_seen)


def test_status_follows_the_run(tmp_path, cli, plain_run):
    run, sites = plain_run
    assert "round 1, train: not emitted yet" in cli("status", run)[1]
    cli("emit", run)
    out = cli("status", run)[1]
    assert "requests are out" in out and "waiting on: [1, 2, 3]" in out
    cli(*client_args(run, {"n": 2, "file": sites[2]}, tmp_path))
    out = cli("status", run)[1]
    assert "waiting on: [1, 3]" in out and "response files present: [2]" in out


def test_collect_says_who_is_missing_and_exits_with_2(tmp_path, cli, plain_run):
    run, sites = plain_run
    cli("emit", run)
    cli(*client_args(run, {"n": 1, "file": sites[1]}, tmp_path))
    code, out, _ = cli("collect", run)
    assert code == 2 and "still waiting on node(s) [2, 3]" in out
    code, out, _ = cli("collect", tmp_path / "nowhere")
    assert code == 1


def test_collect_before_emit_says_so(tmp_path, cli, plain_run):
    code, out, _ = cli("collect", plain_run[0])
    assert code == 2 and "emit first" in out


def test_init_will_not_replace_a_run_without_force(tmp_path, cli, plain_run):
    run, _ = plain_run
    config = tmp_path / "run.yaml"
    code, _, err = cli("init", run, "--config", config)
    assert code == 1 and "already holds a run" in err and "Traceback" not in err
    before = RunDir(run).server().status().run_id
    assert cli("init", run, "--config", config, "--force")[0] == 0
    assert RunDir(run).server().status().run_id != before


def test_errors_are_one_short_line(tmp_path, cli):
    code, _, err = cli("status", tmp_path / "nowhere")
    assert code == 1 and "not a run" in err and "Traceback" not in err and err.count("\n") == 2
    write(tmp_path / "bad.yaml", {"options": {}})
    code, _, err = cli("init", tmp_path / "r", "--config", tmp_path / "bad.yaml")
    assert code == 1 and "missing `program" in err


def test_client_needs_exactly_one_kind_and_something_to_do(tmp_path, cli, plain_run):
    run, sites = plain_run
    common = ("client", "--exchange", run / "exchange", "--state-dir", tmp_path / "s", "--site-file", sites[1])
    assert "exactly one of --app" in cli(*common)[2]
    out = cli(*common, "--client", "test_async_strategy:client_fn")[1]
    assert "nothing to do" in out  # nothing emitted yet
    cli("emit", run)
    assert "wrote" in cli(*common, "--client", "test_async_strategy:client_fn")[1]
    assert "nothing to do" in cli(*common, "--client", "test_async_strategy:client_fn")[1]  # now answered


def test_client_factory_gets_the_site_files_node_and_run_settings(tmp_path, cli, plain_run):
    run, _ = plain_run
    site = write(tmp_path / "site.yaml", {"node_id": 1, "node": {"x": 1}, "run": {"y": 2}})
    cli("emit", run)
    args = ("client", "--exchange", run / "exchange", "--state-dir", tmp_path / "s", "--site-file", site)
    code, out, _ = cli(*args, "--client-factory", "cli_programs:client_factory")
    assert code == 0 and "wrote" in out
    assert cli(*args, "--client", "test_async_strategy:client_fn", "--client-factory", "cli_programs:client_factory")[0] == 1


def test_a_client_refuses_an_older_request_with_a_clear_message(tmp_path, cli, plain_run):
    run, sites = plain_run
    first = Path(cli("emit", run)[1].splitlines()[0])  # node 1's round 1 request
    drive_one_step(cli, run, sites, tmp_path)  # rounds 1 done
    cli("emit", run)
    args = client_args(run, {"n": 1, "file": sites[1]}, tmp_path)
    assert "wrote" in cli(*args)[1]  # round 2 answered
    code, _, err = cli(*args, first)
    assert code == 1 and "older request cannot be run again" in err


def test_reissue_restores_a_lost_request_with_the_same_nonce(tmp_path, cli, plain_run):
    run, sites = plain_run
    files = [Path(line) for line in cli("emit", run)[1].splitlines()]
    original = files[0].read_bytes()
    assert "nothing to restore" in cli("reissue", run)[1]
    files[0].unlink()  # a transport that moved the file away
    code, out, _ = cli("reissue", run)
    assert code == 0 and "restored" in out and files[0].read_bytes() == original
    files[1].write_bytes(files[1].read_bytes()[:-5])  # damaged in transit
    assert "restored" in cli("reissue", run, "--node", 2)[1] and read(files[1])[0].node_id == 2
    assert "nothing to restore" in cli("reissue", run, "--node", 3)[1]
    assert "restored" in cli("reissue", run, "--force", "--node", 3)[1]
    for node, site in sites.items():
        assert cli(*client_args(run, {"n": node, "file": site}, tmp_path))[0] == 0
    assert cli("collect", run)[0] == 0
    assert not list((run / "private" / "state.pkl.requests").iterdir())  # copies are dropped once absorbed


def test_reissue_without_a_stored_copy_is_an_error(tmp_path, cli, plain_run):
    run, _ = plain_run
    files = [Path(line) for line in cli("emit", run)[1].splitlines()]
    files[0].unlink()
    (run / "private" / "state.pkl.requests" / files[0].name).unlink()
    code, _, err = cli("reissue", run)
    assert code == 1 and "no stored copy" in err


def test_inspect_shows_the_header_and_detects_damage(tmp_path, cli, plain_run):
    run, _ = plain_run
    request = Path(cli("emit", run)[1].splitlines()[0])
    code, out, _ = cli("inspect", request)
    header = peek(request)
    assert code == 0 and f"run_id: {header.run_id}" in out and "kind: request" in out and "hash ok" in out
    request.write_bytes(request.read_bytes()[:-3])
    code, out, _ = cli("inspect", request)
    assert code == 1 and "DAMAGED" in out


# ---- COMPASS through the CLI ---------------------------------------------------------------------

REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser() / "compass_hpc_foundation_model_train"


@pytest.mark.skipif(not REPO.is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")
def test_a_compass_run_through_the_cli(tmp_path, cli):
    from compass_cli_driver import audit_lines, drive

    run = drive(lambda *args: cli(*args), tmp_path, REPO, backend="stub", rounds=2, patience=5)
    result = json.loads((run / "results" / "stub_result.json").read_text())
    assert result["rounds_completed"] == 2
    assert len(audit_lines(run)) == (1 + 2 * 2) * 2 * 2  # a request and a response per node per step
