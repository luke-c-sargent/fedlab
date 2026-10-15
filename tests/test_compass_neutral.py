"""COMPASS training as a neutral Strategy, on every runner, with the numpy stub backend.

The stub is numpy only, so this checks the logic and plumbing around the training, not the training. The files in
`tests/golden/` are what the original Flower-based implementation produced for the same runs (captured before it was
removed): equal files mean the port did not change a single number.
"""

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pytest
from flwr.app import Context, RecordDict

from fedlab_async.adapters.flower_server import _RecordingGrid
from fedlab_async.adapters.strategy import StrategyServer, run_strategy_client
from fedlab_async.layout import Exchange
from fedlab_compass import apps_common, stub
from fedlab_compass.contract import DATA_BINDING
from fedlab_compass.neutral_client import make_client
from fedlab_compass.strategy import CompassStrategy, named
from fedlab_core import Payload, run_in_process
from fedlab_core.flower.runner import make_client_app, serve

SITES = {1: "tcga", 2: "gtex"}
RUN_CONFIG = {"backend": "stub"}
GOLDEN = Path(__file__).parent / "golden"


class Rising(stub.StubServer):
    """Validation is best in round 1 and worsens after. The counter is instance state, so it must survive pickling."""

    def __init__(self):
        self.calls = 0

    def combine_validation(self, by_site):
        self.calls += 1
        return stub.Validation(1.0 if self.calls == 1 else 1.0 + self.calls, {})


@pytest.fixture(autouse=True)
def _no_state_dir(monkeypatch):
    monkeypatch.delenv("FEDLAB_STATE_DIR", raising=False)


def clients(sites=SITES):
    return {n: make_client({"site": s}, RUN_CONFIG) for n, s in sites.items()}


def in_process(strategy, sites=SITES):
    by_node = clients(sites)
    return run_in_process(strategy, lambda phase, request, state, node: by_node[node](phase, request, state, node))


def through_files(tmp_path, strategy):
    """One server process per step and one client process per request."""
    ex, state, by_node = Exchange(tmp_path / "exchange"), tmp_path / "private" / "state.pkl", clients()
    StrategyServer.create(state, ex, strategy)
    while not StrategyServer(state, ex).done:
        for request in StrategyServer(state, ex).emit():
            node = int(request.stem.rsplit("-n", 1)[1])
            run_strategy_client(request, by_node[node], ex, tmp_path / "clients")
        assert StrategyServer(state, ex).collect().advanced
    return StrategyServer(state, ex).strategy


class LiveGrid(_RecordingGrid):
    """Stand-in for the SuperLink: delivers each message to the ClientApp of the SuperNode it is for."""

    def __init__(self, app):
        super().__init__([1001, 2002], 1)
        self.app, self.ids = app, {1001: 1, 2002: 2}

    def send_and_receive(self, messages, *, timeout=None):
        node_config = lambda flower_id: {"node-id": self.ids[flower_id], "site": SITES[self.ids[flower_id]], "client-factory": "x"}
        return [
            self.app(m, Context(7, m.metadata.dst_node_id, node_config(m.metadata.dst_node_id), RecordDict(), RUN_CONFIG)) for m in messages
        ]


def through_flower(tmp_path, strategy, monkeypatch):
    monkeypatch.setenv("FEDLAB_STATE_DIR", str(tmp_path / "state"))
    return serve(strategy, LiveGrid(make_client_app(lambda context: make_client(dict(context.node_config), dict(context.run_config)))))


def comparable(history):
    return [(r.round, r.train_loss, r.validation_loss, r.extra["train"], r.extra["validation"]) for r in history]


def make(tmp_path, name, rounds, patience, backend=None):
    return CompassStrategy(SITES, rounds, patience, tmp_path / name, backend=backend or stub.StubServer())


def test_full_run_matches_the_original_implementation_on_every_runner(tmp_path, monkeypatch):
    runs = {
        "process": in_process(make(tmp_path, "process", 6, 50)),
        "files": through_files(tmp_path / "f", make(tmp_path / "f", "files", 6, 50)),
        "flower": through_flower(tmp_path, make(tmp_path, "flower", 6, 50), monkeypatch),
    }
    reference = (GOLDEN / "stub_6_50.json").read_text()
    for name, strategy in runs.items():
        assert len(strategy.history) == 6, name
        assert (strategy.out_dir / "stub_result.json").read_text() == reference, name
    assert json.loads(reference)["optimizer_steps"] == [1, 2, 3, 4, 5, 6]  # optimizer state carried between exchanges


def test_early_stopping_matches_the_original_implementation(tmp_path):
    strategy = through_files(tmp_path, make(tmp_path, "new", 50, 3, Rising()))  # Rising keeps state: it is pickled with the run
    assert len(strategy.history) == 4 and strategy.stopper.best.round == 1  # best round + 3 stale rounds, then stop
    assert strategy.round == 5 and strategy.done
    assert (tmp_path / "new" / "stub_result.json").read_text() == (GOLDEN / "rising_50_3.json").read_text()


def test_weighted_33_to_1(tmp_path):
    in_process(make(tmp_path, "out", 1, 5))
    best = json.loads((tmp_path / "out" / "stub_result.json").read_text())["best_arrays"][0]
    np.testing.assert_allclose(best, [(33 * 0.5 - 0.5) / 34] * 4, rtol=1e-5)  # tcga moves to 0.5, gtex to -0.5


def test_two_nodes_claiming_one_site_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="invalid or duplicate client site"):
        in_process(make(tmp_path, "out", 1, 5), sites={1: "tcga", 2: "tcga"})


def test_strategy_needs_exactly_the_two_sites(tmp_path):
    with pytest.raises(ValueError, match="exactly the sites"):
        CompassStrategy({1: "tcga", 2: "tcga"}, 1, 5, tmp_path, backend=stub.StubServer())


def test_a_dataset_manifest_mismatch_is_refused(tmp_path):
    class Binding(stub.StubServer):
        def expected_data_binding(self):
            return "a" * 64

    with pytest.raises(RuntimeError, match="did not report which federated dataset manifest"):
        in_process(make(tmp_path, "out", 1, 5, Binding()))


def test_timing_separates_waiting_from_server_work(tmp_path):
    strategy, by_node = make(tmp_path, "out", 1, 5), clients()
    plan = strategy.plan()
    replies = {n: by_node[n]("train", r, Payload(), n)[0] for n, r in plan.requests.items()}
    time.sleep(0.4)  # a slow transfer: nobody is working
    strategy.absorb(1, "train", replies)
    plan = strategy.plan()
    replies = {n: by_node[n]("evaluate", r, Payload(), n)[0] for n, r in plan.requests.items()}
    strategy.absorb(1, "evaluate", replies)
    extra = strategy.history[0].extra
    assert extra["elapsed_seconds"] >= 0.4 and extra["wall_seconds"] == extra["elapsed_seconds"]
    assert extra["server_seconds"] < 0.2


def test_a_failed_check_leaves_the_run_retryable(tmp_path):
    ex, state = Exchange(tmp_path / "exchange"), tmp_path / "private" / "state.pkl"
    server = StrategyServer.create(state, ex, make(tmp_path, "out", 2, 5))
    both_tcga = clients({1: "tcga", 2: "tcga"})
    for request in server.emit():
        node = int(request.stem.rsplit("-n", 1)[1])
        run_strategy_client(request, both_tcga[node], ex, tmp_path / "clients")
    with pytest.raises(RuntimeError, match="site"):
        server.collect()
    assert server.strategy.phase == "train" and server.strategy.round == 1 and server.state.pending  # unchanged
    good = clients()
    for request in server.emit():  # a corrected client run overwrites the bad response
        node = int(request.stem.rsplit("-n", 1)[1])
        run_strategy_client(request, good[node], ex, tmp_path / "clients2")
    assert server.collect().advanced


class Unpicklable(stub.StubServer):
    def __init__(self):
        self.torch_like = np  # a module, like the real backend's torch / federated_common handles


def make_unpicklable():
    return Unpicklable()


def test_backend_factory_replaces_an_unpicklable_backend(tmp_path):
    with pytest.raises(TypeError):
        pickle.dumps(make(tmp_path, "a", 2, 5, Unpicklable()))
    strategy = CompassStrategy(SITES, 4, 50, tmp_path / "out", backend_factory=make_unpicklable)
    assert isinstance(pickle.loads(pickle.dumps(strategy)).backend, Unpicklable)
    # a factory-built backend runs end to end, pickled between every step
    result = json.loads((through_files(tmp_path, strategy).out_dir / "stub_result.json").read_text())
    assert result["rounds_completed"] == 4 and result["optimizer_steps"] == [1, 2, 3, 4]


def test_app_level_factory_is_picklable_and_builds_the_backend():
    factory = pickle.loads(pickle.dumps(apps_common.server_backend_factory({"backend": "stub"})))
    assert isinstance(factory(), stub.StubServer)


def test_a_silent_site_aborts_the_round(tmp_path, monkeypatch):
    grid = LiveGrid(make_client_app(lambda context: make_client(dict(context.node_config), dict(context.run_config))))
    answer = grid.send_and_receive
    grid.send_and_receive = lambda messages, *, timeout=None: list(answer(messages, timeout=timeout))[:1]  # only one site answers
    monkeypatch.setenv("FEDLAB_STATE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="no reply from Flower node"):
        serve(make(tmp_path, "out", 1, 5), grid)


# ---- the data-binding field: every client reports the federated manifest it used; the server compares ----

GOOD = "a" * 64


class Expecting(stub.StubServer):
    def expected_data_binding(self):
        return GOOD


def reporting(monkeypatch, by_site, fit_only=False):
    """Clients that add the field (site -> value; None leaves it out), as the real client does."""

    class Reporting(stub.StubClient):
        def fit(self, arrays, config, optimizer):
            result = super().fit(arrays, config, optimizer)
            if by_site[self.site] is not None:
                result.metrics[DATA_BINDING] = by_site[self.site]
            return result

        def evaluate(self, arrays, config):
            metrics = super().evaluate(arrays, config)
            if by_site[self.site] is not None and not fit_only:
                metrics[DATA_BINDING] = by_site[self.site]
            return metrics

    from fedlab_compass import neutral_client

    monkeypatch.setattr(neutral_client, "client_backend", lambda run_config, node_config: Reporting(str(node_config["site"])))


def test_clients_that_report_the_expected_manifest_pass(tmp_path, monkeypatch):
    reporting(monkeypatch, {"tcga": GOOD, "gtex": GOOD})
    assert len(in_process(make(tmp_path, "out", 2, 5, Expecting())).history) == 2


def test_a_client_on_a_different_manifest_stops_the_round_and_is_named(tmp_path, monkeypatch):
    reporting(monkeypatch, {"tcga": GOOD, "gtex": "b" * 64})
    with pytest.raises(RuntimeError, match=r"round 1 train: gtex used a different federated dataset manifest \(bbbbbbbbbbbb"):
        in_process(make(tmp_path, "out", 2, 5, Expecting()))


def test_a_client_that_reports_nothing_is_refused(tmp_path, monkeypatch):
    reporting(monkeypatch, {"tcga": None, "gtex": GOOD})
    with pytest.raises(RuntimeError, match="tcga did not report which federated dataset manifest it used"):
        in_process(make(tmp_path, "out", 2, 5, Expecting()))


def test_evaluate_replies_are_checked_too(tmp_path, monkeypatch):
    reporting(monkeypatch, {"tcga": GOOD, "gtex": GOOD}, fit_only=True)  # evaluate leaves the field out
    with pytest.raises(RuntimeError, match="round 1 evaluate: gtex did not report"):  # sites are checked in sorted order
        in_process(make(tmp_path, "out", 2, 5, Expecting()))


def test_without_an_expectation_nothing_is_checked(tmp_path, monkeypatch):
    reporting(monkeypatch, {"tcga": "garbage", "gtex": None})  # the stub backend has no manifest
    assert len(in_process(make(tmp_path, "out", 1, 5)).history) == 1


ENV_KEYS = ("FEDLAB_COMPASS_CODE", "FEDLAB_COMPASS_SITE_ROOT", "COMPASS_SOURCE_DIR")


def test_env_factory_sets_what_the_real_backend_reads(tmp_path, monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    built = apps_common.EnvFactory(lambda: "backend", tmp_path / "code", tmp_path / "compass", tmp_path / "site")()
    assert built == "backend"
    assert os.environ["FEDLAB_COMPASS_CODE"] == str(tmp_path / "code")
    assert os.environ["FEDLAB_COMPASS_SITE_ROOT"] == str(tmp_path / "site")
    assert os.environ["COMPASS_SOURCE_DIR"] == str(tmp_path / "compass")


def test_the_real_backend_branch_sets_the_environment_around_training_and_restores_it(tmp_path, monkeypatch):
    """The stub skips this branch. The real backends read these variables, so check they are set while the client runs."""
    from fedlab_compass import neutral_client

    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COMPASS_SOURCE_DIR", "/original")
    seen = {}

    class Recording(stub.StubClient):
        def evaluate(self, arrays, config):
            seen.update({k: os.environ.get(k) for k in ENV_KEYS})
            return super().evaluate(arrays, config)

    monkeypatch.setattr(neutral_client, "client_backend", lambda run_config, node_config: Recording("tcga"))
    node = {"site": "tcga", "workdir": str(tmp_path / "w"), "code_root": str(tmp_path / "code"), "compass_source": str(tmp_path / "compass")}
    client = make_client(node, {"backend": "compass", "seed": 42})
    client("evaluate", Payload(named([np.zeros(4, np.float32)]), {"config": {}}), Payload(), 1)
    assert seen == {"FEDLAB_COMPASS_CODE": str(tmp_path / "code"), "FEDLAB_COMPASS_SITE_ROOT": str(tmp_path / "w" / "site_root"),
                    "COMPASS_SOURCE_DIR": str(tmp_path / "compass")}
    assert "FEDLAB_COMPASS_CODE" not in os.environ and os.environ["COMPASS_SOURCE_DIR"] == "/original"  # restored
