"""`module: compass_tcga_gtex_async`: fedlab moves the files of a file-based COMPASS run between fake nodes.

The nodes are LocalHosts (directories on this machine, commands run in them), so everything the SSH versions would do
happens for real except the network. The environment is not installed: each node's ~/fl points at this project's
virtualenv. Skipped if the federated-learning-model checkout is not found (set FEDLAB_REPO).
"""

import json
import os
from pathlib import Path

import pytest

import synth_compass
from fedlab.config import Node, Settings
from fedlab.modules import Deployment, Site, get_module
from fedlab.remote import LocalHost

REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser()
pytestmark = pytest.mark.skipif(not (REPO / "compass_hpc_foundation_model_train").is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")
PROJECT_VENV = Path(__file__).resolve().parents[1] / ".venv"


def fleet(tmp_path, venv=PROJECT_VENV, **options):
    data = tmp_path / "data"
    data.mkdir()
    tcga, gtex = synth_compass.write_tsvs(data)
    cfg = Settings(
        module="compass_tcga_gtex_async", state_dir=tmp_path / ".fedlab",
        module_options={"compass_repo": str(REPO), "tcga_tsv": str(tcga), "gtex_tsv": str(gtex), "backend": "stub", "rounds": 2,
                        "patience": 5, "install_environment": False, **options},
    )
    node = lambda name, role, site=None: Node(name, "aws", "local", role, "t", 10, site=site)  # noqa: E731
    sites = [
        Site(node("srv", "server"), LocalHost("srv", tmp_path / "srv")),
        Site(node("c1", "client", "tcga"), LocalHost("c1", tmp_path / "c1")),
        Site(node("c2", "client", "gtex"), LocalHost("c2", tmp_path / "c2")),
    ]
    for site in sites:
        site.host.root.joinpath("fl").symlink_to(venv)
    logs: list[str] = []
    d = Deployment(cfg, sites[0], sites[1:], tmp_path / "state", log=logs.append)
    return get_module(cfg), d, sites, logs


def test_validate_names_what_is_missing(tmp_path):
    module, d, sites, _ = fleet(tmp_path)
    assert module.validate([s.node for s in sites]) == []
    bad = get_module(Settings(module="compass_tcga_gtex_async", module_options={"backend": "stub"}))
    problems = bad.validate([s.node for s in sites])
    assert any("compass_repo is required" in p for p in problems) and any("tcga_tsv is required" in p for p in problems)
    assert any("exactly the sites" in p for p in module.validate([s.node for s in sites[:2]]))


def test_a_whole_run_through_the_module_lifecycle(tmp_path):
    module, d, sites, logs = fleet(tmp_path)
    module.stage(d)
    module.start(d)
    run_id = module.run(d)
    files = module.collect(d, tmp_path / "out")
    module.stop(d)

    assert len(run_id) == 32
    result = json.loads((tmp_path / "out" / "stub_result.json").read_text())
    assert result["rounds_completed"] == 2 and result["optimizer_steps"] == [1, 2]
    audit = [json.loads(line) for line in (tmp_path / "out" / "state.pkl.audit.jsonl").read_text().splitlines()]
    assert len(audit) == (1 + 2 * 2) * 2 * 2 and {e["run_id"] for e in audit} == {run_id}
    assert {p.name for p in files} >= {"stub_result.json", "state.pkl.audit.jsonl"}
    # each site prepared its own data from the raw file that was staged for it, and got the shared files back
    for name in ("c1", "c2"):
        root = tmp_path / name / "site" / "site_root"
        assert (root / "global_scaler.npz").exists() and (root / "federated_manifest.json").exists()
    assert (tmp_path / "c1" / "site" / "site_root" / "clients" / "tcga" / "manifest.json").exists()
    assert not (tmp_path / "c1" / "site" / "site_root" / "clients" / "gtex").exists()  # a site holds only its own data
    assert any("step 5" in m for m in logs) and not any("step 6" in m for m in logs)


def test_a_failing_site_stops_the_run_with_its_own_output(tmp_path):
    module, d, sites, _ = fleet(tmp_path)
    module.stage(d)
    (tmp_path / "c2" / "data" / "gtex.tsv").write_text("not a data file\n")  # damaged after staging
    module.start(d)
    with pytest.raises(RuntimeError) as error:
        module.run(d)
    assert "client failed on c2" in str(error.value) and "not the assigned file" in str(error.value)  # the site's own output
    assert not list((tmp_path / "srv" / "run" / "exchange" / "responses").glob("*-n2.res"))  # nothing from the failed site reached the server
