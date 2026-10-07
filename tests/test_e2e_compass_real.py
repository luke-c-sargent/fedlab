"""The real COMPASS training path (torch, COMPASS source, prep scripts) on synthetic data, on CPU.

Opt-in: it needs a prepared Python 3.11 environment and two git checkouts, and takes ~10 minutes.

    FEDLAB_E2E_REAL=1 \\
    FEDLAB_REAL_ENV=/path/to/venv          # the module's requirements.txt installed (python 3.11)
    FEDLAB_COMPASS_CLONE=/path/to/COMPASS  # github.com/mims-harvard/COMPASS at the pinned commit
    FEDLAB_REPO=/path/to/federated-learning-model
    uv run pytest tests/test_e2e_compass_real.py -s
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from fedlab.config import Node, Settings
from fedlab.modules import Deployment, Site, get_module
from fedlab.remote import LocalHost

pytestmark = pytest.mark.skipif(not os.environ.get("FEDLAB_E2E_REAL"), reason="set FEDLAB_E2E_REAL=1 (see module docstring)")
TESTS = Path(__file__).parent


def test_real_training_round_trip(tmp_path):
    env, clone, repo = (Path(os.environ[k]) for k in ("FEDLAB_REAL_ENV", "FEDLAB_COMPASS_CLONE", "FEDLAB_REPO"))
    python = env / "bin" / "python"
    data = tmp_path / "data"
    data.mkdir()
    subprocess.run(
        [str(python), "-c", f"import sys; sys.path.insert(0, {str(TESTS)!r}); import synth_compass; from pathlib import Path; synth_compass.write_tsvs(Path({str(data)!r}))"],
        check=True,
    )
    cfg = Settings(
        module="compass_tcga_gtex", state_dir=tmp_path / ".fedlab",
        module_options={
            "compass_repo": str(repo), "tcga_tsv": str(data / "tcga.tsv"), "gtex_tsv": str(data / "gtex.tsv"),
            "prepared_root": str(tmp_path / "prepared"), "prep_python": str(python),
            "device": "cpu", "rounds": 3, "patience": 10,
        },
    )
    module = get_module(cfg)
    module.prepare(print)

    def node(name, role, site=None):
        return Node(name, "aws", "local", role, "t", 10, site=site)

    sites = [
        Site(node("srv", "server"), LocalHost("srv", tmp_path / "srv")),
        Site(node("c1", "client", "tcga"), LocalHost("c1", tmp_path / "c1")),
        Site(node("c2", "client", "gtex"), LocalHost("c2", tmp_path / "c2")),
    ]
    assert module.validate([s.node for s in sites]) == []
    for s in sites:  # reuse the prebuilt environment and COMPASS checkout instead of installing/cloning per node
        s.host.root.joinpath("fl").symlink_to(env)
        s.host.root.joinpath("compass_src").symlink_to(clone)
    d = Deployment(cfg, sites[0], sites[1:], tmp_path / "state", log=print)
    try:
        module.stage(d)
        module.start(d)
        run_id = module.run(d)
        module.collect(d, tmp_path / "out")
    finally:
        module.stop(d)

    out = tmp_path / "out" / f"run-{run_id}"
    assert {p.name for p in out.iterdir()} >= {"best_model.pth", "history.tsv", "pretrainer_federated_tcga_gtex.pt", "rounds.json"}
    rounds = json.loads((out / "rounds.json").read_text())
    assert len(rounds) == 3
    losses = [r["validation_loss"] for r in rounds]
    assert all(loss == loss and loss > 0 for loss in losses)  # finite
    assert losses[-1] < losses[0]  # it learns
