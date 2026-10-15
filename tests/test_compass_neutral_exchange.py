"""A COMPASS file run on the neutral contract: sites prepare their own data, the server builds the scaler, then training.

Preparation, aggregation, shared-file installation and the data checks are the real ones, on synthetic data; training
is the numpy stub. Skipped if the federated-learning-model checkout is not found: set FEDLAB_REPO.
"""

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import pytest

import synth_compass
from fedlab_async.adapters.strategy import StrategyServer, run_strategy_client
from fedlab_async.assets import Asset, AssetError, Manifest
from fedlab_async.envelope import read
from fedlab_async.layout import Exchange
from fedlab_compass import identity
from fedlab_compass.neutral_client import make_client
from fedlab_compass.site import load_fc
from fedlab_compass.strategy import CompassRun
from fedlab_core import decode

REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser() / "compass_hpc_foundation_model_train"
SITES = {1: "tcga", 2: "gtex"}
RUN_CONFIG = {"backend": "stub", "seed": 42}
GOLDEN = Path(__file__).parent / "golden"

pytestmark = pytest.mark.skipif(not REPO.is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")


@pytest.fixture(scope="module")
def remote(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote")
    synth_compass.write_tsvs(root)
    return root


def assignment(path: Path) -> Manifest:
    raw = path.read_bytes()
    return Manifest((Asset("source", path.as_uri(), hashlib.sha256(raw).hexdigest(), len(raw)),))


@pytest.fixture(autouse=True)
def _no_state_dir(monkeypatch):
    monkeypatch.delenv("FEDLAB_STATE_DIR", raising=False)


def run_setup(tmp_path, remote, assignments=None, sites=SITES, rounds=3, patience=50):
    return dict(
        sites=sites, assignments=assignments or {1: assignment(remote / "tcga.tsv"), 2: assignment(remote / "gtex.tsv")},
        code_root=REPO, expected_identity=identity.code_identity(REPO), run_config=RUN_CONFIG, rounds=rounds, patience=patience,
        out_dir=tmp_path / "out", server_site_root=tmp_path / "server_site",
    )


class World:
    def __init__(self, tmp_path, remote, **kwargs):
        self.tmp, self.ex, self.state = tmp_path, Exchange(tmp_path / "exchange"), tmp_path / "private" / "state.pkl"
        self.code_root = REPO
        StrategyServer.create(self.state, self.ex, CompassRun(**run_setup(tmp_path, remote, **kwargs)))

    def server(self):
        return StrategyServer(self.state, self.ex)

    def node_config(self, node, code_root=None):
        return {"site": SITES[node], "workdir": str(self.tmp / f"site-{node}"), "cache": str(self.tmp / f"cache-{node}"),
                "code_root": str(code_root or self.code_root)}

    def run_client(self, request, code_root=None):
        node = int(request.stem.rsplit("-n", 1)[1])
        fn = make_client(self.node_config(node, code_root), RUN_CONFIG)
        return run_strategy_client(request, fn, self.ex, self.tmp / f"clients-{node}")

    def run_to_end(self):
        while not self.server().done:
            for request in self.server().emit():
                self.run_client(request)
            assert self.server().collect().advanced


def test_full_run_prepares_aggregates_and_trains(tmp_path, remote):
    world = World(tmp_path, remote)
    assert world.server().status().phase == "prepare"
    world.run_to_end()

    result = json.loads((tmp_path / "out" / "stub_result.json").read_text())
    assert result["rounds_completed"] == 3 and result["optimizer_steps"] == [1, 2, 3]
    assert (tmp_path / "out" / "stub_result.json").read_text() == (GOLDEN / "stub_3_50.json").read_text()  # as the original gave

    shared = {n: (tmp_path / "server_site" / n).read_bytes() for n in ("global_scaler.npz", "federated_manifest.json", "gene_order.txt")}
    for node in SITES:
        root = tmp_path / f"site-{node}" / "site_root"
        assert {n: (root / n).read_bytes() for n in shared} == shared  # server and sites ended with the same shared files
        assert list(root.glob(".data-verified-*"))  # the site's own data was checked against the federated manifest
    fc = load_fc(REPO)  # what the sites ended up with passes the original full validation
    combined = tmp_path / "combined" / "clients"
    combined.mkdir(parents=True)
    for node, site in SITES.items():
        (combined / site).symlink_to(tmp_path / f"site-{node}" / "site_root" / "clients" / site)
    fc.validate_federated_prepared_manifest(
        tmp_path / "server_site" / "federated_manifest.json", clients_root=combined,
        global_scaler=tmp_path / "server_site" / "global_scaler.npz", config_path=REPO / "centralized_test" / "config" / "paper_pretraining.json",
    )
    audit = [json.loads(line) for line in world.state.with_name("state.pkl.audit.jsonl").read_text().splitlines()]
    assert sorted((e["round"], e["phase"], e["event"], e["node_id"]) for e in audit if e["round"] == 0) == [
        (0, "prepare", "request", 1), (0, "prepare", "request", 2), (0, "prepare", "response", 1), (0, "prepare", "response", 2)]


def test_training_requests_carry_the_shared_files_and_the_code_identity(tmp_path, remote):
    world = World(tmp_path, remote)
    for request in world.server().emit():
        world.run_client(request)
    assert world.server().collect().advanced
    world.server().emit()
    request = decode(read(world.ex.request(1, "train", 1))[1])
    assert request.blobs["scaler_npz"] == (tmp_path / "server_site" / "global_scaler.npz").read_bytes()
    assert json.loads(request.meta["code_identity"]) == identity.code_identity(REPO)
    assert request.meta["config"]["server_round"] == 1 and request.meta["config"]["server-round"] == 1


def test_the_server_waits_for_both_sites_to_prepare(tmp_path, remote):
    world = World(tmp_path, remote)
    first, second = world.server().emit()
    world.run_client(second)  # gtex first
    result = world.server().collect()
    assert not result.advanced and result.missing == [1] and world.server().status().phase == "prepare"
    assert not (tmp_path / "server_site").exists()  # nothing is built from a partial set
    world.run_client(first)
    assert world.server().collect().advanced and world.server().status().phase == "train"


def test_a_site_with_different_code_refuses_to_prepare(tmp_path, remote):
    other = tmp_path / "other_code"
    for tree in identity.TREES:
        shutil.copytree(REPO / tree, other / tree, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (other / "federated_test" / "scripts" / "federated_common.py").write_text(
        (REPO / "federated_test" / "scripts" / "federated_common.py").read_text() + "\n# local edit\n"
    )
    world = World(tmp_path, remote)
    requests = world.server().emit()
    with pytest.raises(identity.CodeMismatch, match="federated_test/ differs"):
        world.run_client(requests[0], code_root=other)
    assert not world.ex.response(0, "prepare", 1).exists()


def test_a_remote_file_that_is_not_the_assigned_one_is_refused(tmp_path, remote):
    mine = tmp_path / "remote"
    shutil.copytree(remote, mine)  # this test changes a file, so it works on its own copy
    world = World(tmp_path, mine)
    requests = world.server().emit()
    with (mine / "gtex.tsv").open("ab") as handle:  # the data at the assigned URI changes after the server hashed it
        handle.write(b"\n")
    with pytest.raises(AssetError, match="not the assigned file"):
        world.run_client(requests[1])
    assert not world.ex.response(0, "prepare", 2).exists()


def test_the_server_refuses_data_prepared_from_a_file_other_than_the_one_it_assigned(tmp_path, remote):
    world = World(tmp_path, remote)
    for request in world.server().emit():
        world.run_client(request)
    state = pickle.loads(world.state.read_bytes())  # the server believes it assigned a different file to node 1
    run = state.program.strategy
    original = run.assignments[1].files[0]
    run.assignments[1] = Manifest((Asset("source", original.uri, "0" * 64, original.bytes),))
    world.state.write_bytes(pickle.dumps(state))
    with pytest.raises(ValueError, match="other than the assigned"):
        world.server().collect()
    assert world.server().status().phase == "prepare" and not (tmp_path / "server_site").exists()


def test_run_refuses_inconsistent_setup(tmp_path, remote):
    setup = run_setup(tmp_path, remote)
    with pytest.raises(ValueError, match="exactly one assignment"):
        CompassRun(**{**setup, "assignments": {1: assignment(remote / "tcga.tsv")}})
    with pytest.raises(ValueError, match="site must be one of"):
        CompassRun(**{**setup, "sites": {1: "lung"}, "assignments": {1: assignment(remote / "tcga.tsv")}})
    no_source = Manifest((Asset("data", "file:///x", "0" * 64, 1),))
    with pytest.raises(ValueError, match="named 'source'"):
        CompassRun(**{**setup, "sites": {1: "tcga"}, "assignments": {1: no_source}})
