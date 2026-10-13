"""A COMPASS run as files: sites prepare their own data, the server builds the scaler, then training rounds.

Preparation, aggregation, shared-file installation and the data checks are the real ones, on synthetic data.
Training uses the numpy stub backend (the real one needs torch), so this checks the plumbing around it.
Skipped if the federated-learning-model checkout is not found: set FEDLAB_REPO.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from flwr.common.serde import message_from_proto
from flwr.proto.message_pb2 import Message as ProtoMessage

import synth_compass
from fedlab_async.adapters.flower_client import run_client
from fedlab_async.assets import Asset, AssetError, Manifest
from fedlab_async.envelope import read
from fedlab_async.layout import Exchange
from fedlab_compass import async_client, identity, orchestrator, stub
from fedlab_compass.async_run import CompassServer, EnvFactory
from fedlab_compass.site import load_fc

REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser() / "compass_hpc_foundation_model_train"
pytestmark = pytest.mark.skipif(not REPO.is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")
SITES = {1: "tcga", 2: "gtex"}
RUN_CONFIG = {"backend": "stub", "seed": 42}


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


class World:
    def __init__(self, tmp_path, remote, code_root=REPO, rounds=3, patience=50):
        self.tmp, self.ex = tmp_path, Exchange(tmp_path / "exchange")
        self.state = tmp_path / "private" / "state.pkl"
        self.code_root = code_root
        assignments = {1: assignment(remote / "tcga.tsv"), 2: assignment(remote / "gtex.tsv")}
        CompassServer.create(
            self.state, self.ex, sites=SITES, assignments=assignments, code_root=REPO, expected_identity=identity.code_identity(REPO),
            run_config=RUN_CONFIG, rounds=rounds, patience=patience, out_dir=tmp_path / "out", server_site_root=tmp_path / "server_site",
        )

    def server(self):
        return CompassServer(self.state, self.ex)

    def node_config(self, node, code_root=None):
        return {"site": SITES[node], "workdir": str(self.tmp / f"site-{node}"), "cache": str(self.tmp / f"cache-{node}"),
                "code_root": str(code_root or self.code_root)}

    def run_client(self, request, code_root=None):
        node = int(request.stem.rsplit("-n", 1)[1])
        return run_client(request, async_client.app, self.ex, self.tmp / f"clients-{node}", self.node_config(node, code_root), RUN_CONFIG)

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
    # the same training as a live run: the stub does not depend on the data, so this checks the plumbing around it
    from test_async_compass import LiveGrid, live  # noqa: PLC0415

    history, _ = live(stub.StubServer(), 3, 50, tmp_path / "live")
    assert (tmp_path / "live" / "stub_result.json").read_text() == (tmp_path / "out" / "stub_result.json").read_text()

    # the server and both sites ended with the same shared files, and the original validators accept them
    shared = {n: (tmp_path / "server_site" / n).read_bytes() for n in ("global_scaler.npz", "federated_manifest.json", "gene_order.txt")}
    for node in SITES:
        root = tmp_path / f"site-{node}" / "site_root"
        assert {n: (root / n).read_bytes() for n in shared} == shared
        assert list(root.glob(".data-verified-*"))  # the site's own data was checked against the federated manifest
    # what the sites actually ended up with passes the original full validation, read from the sites' own directories
    fc = load_fc(REPO)
    combined = tmp_path / "combined" / "clients"
    combined.mkdir(parents=True)
    for node, site in SITES.items():
        (combined / site).symlink_to(tmp_path / f"site-{node}" / "site_root" / "clients" / site)
    fc.validate_federated_prepared_manifest(
        tmp_path / "server_site" / "federated_manifest.json", clients_root=combined,
        global_scaler=tmp_path / "server_site" / "global_scaler.npz", config_path=REPO / "centralized_test" / "config" / "paper_pretraining.json",
    )
    # exchange 0 is in the audit log, with both sites' reports
    audit = [json.loads(line) for line in world.state.with_name("state.pkl.audit.jsonl").read_text().splitlines()]
    assert sorted((e["round"], e["phase"], e["event"], e["node_id"]) for e in audit if e["round"] == 0) == [
        (0, "prepare", "request", 1), (0, "prepare", "request", 2), (0, "prepare", "response", 1), (0, "prepare", "response", 2)]


def test_training_requests_carry_the_shared_files_and_the_code_identity(tmp_path, remote):
    world = World(tmp_path, remote)
    for request in world.server().emit():
        world.run_client(request)
    assert world.server().collect().advanced
    world.server().emit()
    _, payload = read(world.ex.request(1, "train", 1))
    config = dict(message_from_proto(ProtoMessage.FromString(payload)).content["config"])
    assert config["scaler_npz"] == (tmp_path / "server_site" / "global_scaler.npz").read_bytes()
    assert json.loads(config["code_identity"]) == identity.code_identity(REPO)
    assert config["server_round"] == 1 and config["server-round"] == 1  # the backend's and the strategy's own entries


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


def test_resuming_from_disk_between_every_step_gives_the_same_run(tmp_path, remote):
    world = World(tmp_path, remote, rounds=2)
    while not world.server().done:
        requests = world.server().emit()
        for request in reversed(requests):
            world.run_client(request)
        assert world.server().collect().advanced
    assert json.loads((tmp_path / "out" / "stub_result.json").read_text())["rounds_completed"] == 2


def test_env_factory_sets_what_the_real_backend_reads(tmp_path, monkeypatch):
    for key in ("FEDLAB_COMPASS_CODE", "FEDLAB_COMPASS_SITE_ROOT", "COMPASS_SOURCE_DIR"):
        monkeypatch.delenv(key, raising=False)
    built = EnvFactory(lambda: "backend", tmp_path / "code", tmp_path / "compass", tmp_path / "site")()
    assert built == "backend"
    assert os.environ["FEDLAB_COMPASS_CODE"] == str(tmp_path / "code")
    assert os.environ["FEDLAB_COMPASS_SITE_ROOT"] == str(tmp_path / "site")
    assert os.environ["COMPASS_SOURCE_DIR"] == str(tmp_path / "compass")


def test_program_refuses_inconsistent_setup(tmp_path, remote):
    ex = Exchange(tmp_path / "ex")
    base = dict(code_root=REPO, expected_identity={}, run_config=RUN_CONFIG, rounds=1, patience=1, out_dir=tmp_path, server_site_root=tmp_path)
    with pytest.raises(ValueError, match="exactly one assignment"):
        CompassServer.create(tmp_path / "a.pkl", ex, sites=SITES, assignments={1: assignment(remote / "tcga.tsv")}, **base)
    with pytest.raises(ValueError, match="site must be one of"):
        CompassServer.create(tmp_path / "b.pkl", ex, sites={1: "lung"}, assignments={1: assignment(remote / "tcga.tsv")}, **base)
    no_source = Manifest((Asset("data", "file:///x", "0" * 64, 1),))
    with pytest.raises(ValueError, match="named 'source'"):
        CompassServer.create(tmp_path / "c.pkl", ex, sites={1: "tcga"}, assignments={1: no_source}, **base)


def test_the_server_refuses_data_prepared_from_a_file_other_than_the_one_it_assigned(tmp_path, remote):
    import pickle

    world = World(tmp_path, remote)
    for request in world.server().emit():
        world.run_client(request)
    state = pickle.loads(world.state.read_bytes())  # the server believes it assigned a different file to node 1
    original = state.program.assignments[1].files[0]
    state.program.assignments[1] = Manifest((Asset("source", original.uri, "0" * 64, original.bytes),))
    world.state.write_bytes(pickle.dumps(state))
    with pytest.raises(ValueError, match="other than the assigned"):
        world.server().collect()
    assert world.server().status().phase == "prepare" and not (tmp_path / "server_site").exists()


def test_the_real_backend_branch_sets_the_environment_around_training_and_restores_it(tmp_path, monkeypatch):
    """The stub skips this branch. The real backends read these variables, so check they are set while the action runs."""
    from flwr.app import ConfigRecord, Context, Message, RecordDict

    for key in ("FEDLAB_COMPASS_CODE", "FEDLAB_COMPASS_SITE_ROOT", "COMPASS_SOURCE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COMPASS_SOURCE_DIR", "/original")
    seen = {}

    def action(msg, context):
        seen.update({k: os.environ.get(k) for k in ("FEDLAB_COMPASS_CODE", "FEDLAB_COMPASS_SITE_ROOT", "COMPASS_SOURCE_DIR")})
        return "reply"

    node = {"site": "tcga", "workdir": str(tmp_path / "w"), "code_root": str(tmp_path / "code"), "compass_source": str(tmp_path / "compass")}
    msg = Message(RecordDict({"config": ConfigRecord({})}), dst_node_id=1, message_type="train")
    context = Context(1, 1, node, RecordDict(), {"backend": "compass", "seed": 42})
    assert async_client._run(action, msg, context) == "reply"
    assert seen == {"FEDLAB_COMPASS_CODE": str(tmp_path / "code"), "FEDLAB_COMPASS_SITE_ROOT": str(tmp_path / "w" / "site_root"),
                    "COMPASS_SOURCE_DIR": str(tmp_path / "compass")}
    assert "FEDLAB_COMPASS_CODE" not in os.environ and os.environ["COMPASS_SOURCE_DIR"] == "/original"  # restored
