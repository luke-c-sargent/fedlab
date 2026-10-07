"""Full module lifecycle on LocalHost nodes (real Flower SuperLink + SuperNodes with TLS and auth).

Slow and needs network (uv installs Flower), so it only runs with FEDLAB_E2E=1.
"""

import json
import os

import pytest

from fedlab.config import Node, Settings
from fedlab.modules import Deployment, Site, get_module
from fedlab.remote import LocalHost

pytestmark = pytest.mark.skipif(not os.environ.get("FEDLAB_E2E"), reason="set FEDLAB_E2E=1 to run")


def _node(name, role, site=None):
    return Node(name, "aws", "local", role, "t", 10, site=site)


def _deployment(tmp_path, cfg, client_sites):
    sites = [Site(_node("srv", "server"), LocalHost("srv", tmp_path / "srv"))] + [
        Site(_node(f"c{i}", "client", s), LocalHost(f"c{i}", tmp_path / f"c{i}")) for i, s in enumerate(client_sites, 1)
    ]
    return Deployment(cfg, sites[0], sites[1:], tmp_path / "state", log=print)


def test_compass_module_with_stub_backend(tmp_path):
    """The real COMPASS app's orchestration over real TLS/auth Flower, with the numpy stand-in trainer."""
    cfg = Settings(module="compass_tcga_gtex", module_options={"backend": "stub", "rounds": 8, "patience": 3}, state_dir=tmp_path / ".fedlab")
    d = _deployment(tmp_path, cfg, ["tcga", "gtex"])
    module = get_module(cfg)
    try:
        module.stage(d)
        module.start(d)
        run_id = module.run(d)
        module.collect(d, tmp_path / "out")
    finally:
        module.stop(d)
    result = json.loads((tmp_path / "out" / f"run-{run_id}" / "stub_result.json").read_text())
    assert result["optimizer_steps"] == list(range(1, result["rounds_completed"] + 1))  # Adam-state stand-in persisted
    assert 1 <= result["rounds_completed"] <= 8


def test_smoke_module_lifecycle(tmp_path):
    cfg = Settings(module="smoke", module_options={"rounds": 3}, state_dir=tmp_path / ".fedlab")
    d = _deployment(tmp_path, cfg, ["a", "b"])
    module = get_module(cfg)
    try:
        module.stage(d)
        module.start(d)
        run_id = module.run(d)
        files = module.collect(d, tmp_path / "out")
    finally:
        module.stop(d)
    result = json.loads((tmp_path / "out" / f"run-{run_id}" / "result.json").read_text())
    assert result["arrays"] == [[3.0, 3.0, 3.0]]  # +1 per round, averaged over two clients
    assert result["train"]["3"]["calls"] == 3.0  # context.state survived across rounds
    assert any(p.name.endswith(".superlink.log") for p in files)
