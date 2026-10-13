"""A file-based COMPASS run, as separate processes of the real `fedlab-async` commands.

The first test always runs (given the federated-learning-model checkout): it drives the subprocess commands with
the numpy stub backend, so the mechanism is checked without torch. The second is the real thing: real COMPASS
training on synthetic data, on CPU, with the commands run by the Python of a prepared environment.

Opt-in, like tests/test_e2e_compass_real.py (read its docstring for the environment), plus typer and pyyaml in that
environment (`pip install typer pyyaml`), because the CLI needs them and the module's requirements.txt does not list them:

    FEDLAB_E2E_REAL=1 \\
    FEDLAB_REAL_ENV=/path/to/venv          # the module's requirements.txt installed (python 3.11) + typer pyyaml
    FEDLAB_COMPASS_CLONE=/path/to/COMPASS  # github.com/mims-harvard/COMPASS at the pinned commit
    FEDLAB_REPO=/path/to/federated-learning-model
    [FEDLAB_COMPARE_ROUNDS=/path/to/rounds.json]   # rounds.json of a live run on the same data, to compare losses
    uv run pytest tests/test_e2e_compass_real_async.py -s

Do not run these at the same time as tests/test_e2e_compass_real.py, or anything else that uses FEDLAB_REAL_ENV. The
live test's staging step runs `uv pip install` into that shared environment, from the server and both clients at once,
and each run re-downloads and reinstalls torch. Another process that imports torch meanwhile fails with errors such as
"No module named 'torch._subclasses.meta_utils'". Run them one after the other.

To produce the file for FEDLAB_COMPARE_ROUNDS, run the live test with `--basetemp=/some/dir` and use
`/some/dir/.../out/run-*/rounds.json`. The validation losses should agree: both runs use the same seed, data and
code. They are compared with a relative tolerance of FEDLAB_COMPARE_RTOL (default 1e-4); if CPU training is not
bit-reproducible on your machine, loosen it.
"""

import json
import os
import sys
from pathlib import Path

import pytest

import fedlab_async
import fedlab_compass
from compass_cli_driver import audit_lines, drive, subprocess_runner

FEDERATED_REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser()
CODE_ROOT = FEDERATED_REPO / "compass_hpc_foundation_model_train"
SRC = Path(fedlab_async.__file__).parent.parent
APP = Path(fedlab_compass.__file__).parent.parent
needs_repo = pytest.mark.skipif(not CODE_ROOT.is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")
needs_real = pytest.mark.skipif(not os.environ.get("FEDLAB_E2E_REAL"), reason="set FEDLAB_E2E_REAL=1 (see module docstring)")


@needs_repo
def test_subprocess_commands_run_a_stub_compass_run(tmp_path):
    run = drive(subprocess_runner(sys.executable, SRC, APP), tmp_path, CODE_ROOT, backend="stub", rounds=2)
    assert json.loads((run / "results" / "stub_result.json").read_text())["rounds_completed"] == 2
    assert len(audit_lines(run)) == (1 + 2 * 2) * 2 * 2


@needs_repo
@needs_real
def test_real_compass_training_as_files(tmp_path):
    env, clone = Path(os.environ["FEDLAB_REAL_ENV"]), Path(os.environ["FEDLAB_COMPASS_CLONE"])
    run = drive(
        subprocess_runner(str(env / "bin" / "python"), SRC, APP), tmp_path, CODE_ROOT, backend="compass", rounds=3,
        compass_source=clone,
        run_settings={"device": "cpu", "local-epochs": 1, "micro-batch-size": 16, "num-workers": 0, "cpu-threads": 8},  # the module defaults, as in the live test
    )
    out = run / "results"
    assert {p.name for p in out.iterdir()} >= {"best_model.pth", "history.tsv", "pretrainer_federated_tcga_gtex.pt", "rounds.json"}
    rounds = json.loads((out / "rounds.json").read_text())
    assert len(rounds) == 3
    losses = [r["validation_loss"] for r in rounds]
    assert all(loss == loss and loss > 0 for loss in losses)  # finite
    assert losses[-1] < losses[0]  # it learns
    assert len(audit_lines(run)) == (1 + 2 * 3) * 2 * 2  # prepare, then train and evaluate per round

    reference = os.environ.get("FEDLAB_COMPARE_ROUNDS")
    if reference:  # the same experiment run live must give the same validation losses
        live = [r["validation_loss"] for r in json.loads(Path(reference).read_text())]
        assert losses == pytest.approx(live, rel=float(os.environ.get("FEDLAB_COMPARE_RTOL", "1e-4")))


@needs_repo
@needs_real
def test_real_compass_training_moved_by_the_fedlab_module(tmp_path):
    """`module: compass_tcga_gtex_async` on fake nodes: fedlab provisions the files and moves them, the training is real."""
    from test_async_fleet import fleet

    env, clone = Path(os.environ["FEDLAB_REAL_ENV"]), Path(os.environ["FEDLAB_COMPASS_CLONE"])
    module, d, sites, logs = fleet(tmp_path, venv=env, backend="compass", rounds=3, patience=10, device="cpu")
    for site in sites:  # reuse the prebuilt COMPASS checkout instead of fetching it per node
        site.host.root.joinpath("compass_src").symlink_to(clone)
    module.stage(d)
    module.start(d)
    run_id = module.run(d)
    module.collect(d, tmp_path / "out")
    module.stop(d)

    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} >= {"best_model.pth", "history.tsv", "pretrainer_federated_tcga_gtex.pt", "rounds.json", "state.pkl.audit.jsonl"}
    rounds = json.loads((out / "rounds.json").read_text())
    losses = [r["validation_loss"] for r in rounds]
    assert len(rounds) == 3 and all(loss == loss and loss > 0 for loss in losses) and losses[-1] < losses[0]
    assert len((out / "state.pkl.audit.jsonl").read_text().splitlines()) == (1 + 2 * 3) * 2 * 2
    assert any("step 7" in m for m in logs) and len(run_id) == 32

    reference = os.environ.get("FEDLAB_COMPARE_ROUNDS")
    if reference:
        live = [r["validation_loss"] for r in json.loads(Path(reference).read_text())]
        assert losses == pytest.approx(live, rel=float(os.environ.get("FEDLAB_COMPARE_RTOL", "1e-4")))
