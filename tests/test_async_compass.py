"""The COMPASS orchestration (weighted FedAvg, early stopping, best model, finalize) run as files."""

import json
import pickle
import random
import time

import numpy as np
import pytest
from flwr.app import ArrayRecord, Context, RecordDict

from fedlab_async.adapters.flower_client import run_client
from fedlab_async.adapters.flower_server import FlowerServer
from fedlab_async.layout import Exchange
from fedlab_compass import apps_common, client_app, orchestrator, stub
from flwr.serverapp.strategy import FedAvg

SITES = {1: "tcga", 2: "gtex"}
RUN_CONFIG = {"backend": "stub"}


class Rising(stub.StubServer):
    """Validation is best in round 1 and worsens after. The counter is instance state, so it must survive pickling."""

    def __init__(self):
        self.calls = 0

    def combine_validation(self, by_site):
        self.calls += 1
        return stub.Validation(1.0 if self.calls == 1 else 1.0 + self.calls, {})


class LiveGrid:
    def __init__(self):
        self.contexts = {n: Context(1, n, {"site": s}, RecordDict(), RUN_CONFIG) for n, s in SITES.items()}

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, timeout=None):
        return [client_app.app(m, self.contexts[m.metadata.dst_node_id]) for m in messages]


def live(backend, rounds, patience, out):
    return orchestrator.run_federation(LiveGrid(), backend, rounds, patience, out)


def server_for(tmp_path, backend, rounds, patience, factory=None):
    federation = orchestrator.Federation(backend, rounds, patience, tmp_path / "out", backend_factory=factory)
    ex = Exchange(tmp_path / "exchange")
    server = FlowerServer.create(
        tmp_path / "private" / "state.pkl", ex, orchestrator.make_strategy(), ArrayRecord((backend or factory()).initial_arrays()),
        list(SITES), rounds, hooks=federation,
    )
    return server, ex


def drive(server, ex, tmp_path, stop_after_phase=None):
    """One client process per request, one server process per step."""
    steps = 0
    while not server.done:
        for request in server.emit():
            node = int(request.stem.rsplit("-n", 1)[1])
            run_client(request, client_app.app, ex, tmp_path / "clients", {"site": SITES[node]}, RUN_CONFIG)
        assert server.collect().advanced
        server = FlowerServer(server.state_path, ex)  # fresh process: nothing carried over in memory
        steps += 1
    return server


def comparable(history):
    return [(r.round, r.train_loss, r.validation_loss, r.extra["train"], r.extra["validation"]) for r in history]


def test_full_run_matches_live(tmp_path):
    random.seed(0)
    history, stopper = live(stub.StubServer(), 6, 50, tmp_path / "live")
    random.seed(0)
    server, ex = server_for(tmp_path, stub.StubServer(), 6, 50)
    server = drive(server, ex, tmp_path)
    federation = server.program.hooks
    assert comparable(federation.history) == comparable(history)
    assert len(federation.history) == 6
    assert (tmp_path / "out" / "stub_result.json").read_text() == (tmp_path / "live" / "stub_result.json").read_text()
    steps = json.loads((tmp_path / "out" / "stub_result.json").read_text())["optimizer_steps"]
    assert steps == [1, 2, 3, 4, 5, 6]  # client optimizer state carried across six separate processes


def test_early_stopping_matches_live(tmp_path):
    history, stopper = live(Rising(), 50, 3, tmp_path / "live")
    server, ex = server_for(tmp_path, Rising(), 50, 3)
    server = drive(server, ex, tmp_path)
    federation = server.program.hooks
    assert len(federation.history) == len(history) == 4  # best round + 3 stale rounds, then stop
    assert federation.stopper.best.round == stopper.best.round == 1
    assert server.program.round == 5 and server.done
    assert (tmp_path / "out" / "stub_result.json").read_text() == (tmp_path / "live" / "stub_result.json").read_text()


def test_weighted_33_to_1(tmp_path):
    server, ex = server_for(tmp_path, stub.StubServer(), 1, 5)
    drive(server, ex, tmp_path)
    best = json.loads((tmp_path / "out" / "stub_result.json").read_text())["best_arrays"][0]
    # from zeros: tcga moves to 0.5, gtex to -0.5; weighted 33:1
    np.testing.assert_allclose(best, [(33 * 0.5 - 0.5) / 34] * 4, rtol=1e-5)


def test_a_failed_client_check_leaves_state_retryable(tmp_path):
    server, ex = server_for(tmp_path, stub.StubServer(), 2, 5)
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": "tcga"}, RUN_CONFIG)  # both claim tcga
    with pytest.raises(RuntimeError, match="site"):
        server.collect()
    assert server.program.phase == "train" and server.program.round == 1 and server.state.pending  # unchanged
    for request in server.emit():  # a corrected client run overwrites the bad response
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients2", {"site": SITES[node]}, RUN_CONFIG)
    assert server.collect().advanced


CALLS = []


class Recording(FedAvg):
    def aggregate_train(self, server_round, replies):
        CALLS.append(server_round)
        return super().aggregate_train(server_round, replies)


def test_replies_are_checked_before_the_strategy_aggregates(tmp_path):
    ex = Exchange(tmp_path / "exchange")
    federation = orchestrator.Federation(stub.StubServer(), 2, 5, tmp_path / "out")
    strategy = Recording(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2, weighted_by_key="num-examples")
    server = FlowerServer.create(tmp_path / "s.pkl", ex, strategy, ArrayRecord([np.zeros(4, dtype=np.float32)]), [1, 2], 2, hooks=federation)
    for request in server.emit():
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": "tcga"}, RUN_CONFIG)  # both claim tcga
    CALLS.clear()
    with pytest.raises(RuntimeError, match="site"):
        server.collect()
    assert CALLS == []


class Unpicklable(stub.StubServer):
    def __init__(self):
        self.torch_like = np  # a module, like the real backend's torch / federated_common handles


def make_unpicklable():
    return Unpicklable()


def test_backend_factory_replaces_an_unpicklable_backend(tmp_path):
    with pytest.raises(TypeError):
        pickle.dumps(orchestrator.Federation(Unpicklable(), 2, 5, tmp_path))
    federation = orchestrator.Federation(None, 2, 5, tmp_path / "out", backend_factory=make_unpicklable)
    assert isinstance(pickle.loads(pickle.dumps(federation)).backend, Unpicklable)

    random.seed(0)
    history, _ = live(stub.StubServer(), 4, 50, tmp_path / "live")
    server, ex = server_for(tmp_path, None, 4, 50, factory=make_unpicklable)
    server = drive(server, ex, tmp_path)
    assert comparable(server.program.hooks.history) == comparable(history)


def test_app_level_factory_is_picklable_and_builds_the_backend():
    factory = pickle.loads(pickle.dumps(apps_common.server_backend_factory({"backend": "stub"})))
    assert isinstance(factory(), stub.StubServer)


def test_timing_separates_waiting_from_server_work(tmp_path):
    server, ex = server_for(tmp_path, stub.StubServer(), 1, 5)
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": SITES[node]}, RUN_CONFIG)
    time.sleep(0.4)  # a slow transfer: nobody is working
    server.collect()
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": SITES[node]}, RUN_CONFIG)
    server.collect()
    extra = server.program.hooks.history[0].extra
    assert extra["elapsed_seconds"] >= 0.4 and extra["wall_seconds"] == extra["elapsed_seconds"]
    assert extra["server_seconds"] < 0.2


def test_a_wrong_manifest_in_a_file_based_run_is_refused_and_the_step_can_be_retried(tmp_path, monkeypatch):
    from test_compass_app import GOOD, Expecting, reporting

    reporting(monkeypatch, {"tcga": GOOD, "gtex": "c" * 64})
    server, ex = server_for(tmp_path, Expecting(), 2, 5)
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": SITES[node]}, RUN_CONFIG)
    with pytest.raises(RuntimeError, match="gtex used a different federated dataset manifest"):
        server.collect()
    assert server.program.phase == "train" and server.state.pending  # nothing was advanced
    reporting(monkeypatch, {"tcga": GOOD, "gtex": GOOD})  # the site is fixed and runs its request again
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_client(request, client_app.app, ex, tmp_path / "clients", {"site": SITES[node]}, RUN_CONFIG)
    assert server.collect().advanced
