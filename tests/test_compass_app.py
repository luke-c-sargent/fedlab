"""The COMPASS Flower app's orchestration, run through Flower's own strategy and ClientApp with a fake grid.

Uses the numpy stub backend, so no torch, data or cloud is needed.
"""

import json

import numpy as np
import pytest
from flwr.app import Context, RecordDict

from fedlab_compass import client_app, logic, orchestrator, stub
from fedlab_compass.contract import OptimizerState


class FakeGrid:
    """Two SuperNodes (tcga, gtex). `send_and_receive` hands each message to the real ClientApp."""

    def __init__(self, backend="stub"):
        run_config = {"backend": backend}
        self.contexts = {
            1: Context(run_id=1, node_id=1, node_config={"site": "tcga"}, state=RecordDict(), run_config=run_config),
            2: Context(run_id=1, node_id=2, node_config={"site": "gtex"}, state=RecordDict(), run_config=run_config),
        }
        self.rounds_seen = 0

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            replies.append(client_app.app(message=message, context=self.contexts[message.metadata.dst_node_id]))
        return replies


def test_runs_all_rounds_and_keeps_optimizer_state(tmp_path):
    history, stopper = orchestrator.run_federation(FakeGrid(), stub.StubServer(), rounds=6, patience=50, out_dir=tmp_path)
    assert len(history) == 6
    # the stub counts optimizer steps in context.state: one more each round proves state survives
    result = json.loads((tmp_path / "stub_result.json").read_text())
    assert result["optimizer_steps"] == [1, 2, 3, 4, 5, 6]
    assert result["rounds_completed"] == 6


def test_optimizer_state_can_live_in_a_node_local_file(tmp_path, monkeypatch):
    """Adam state is too big for `context.state`, so a node-local directory takes over when configured."""
    monkeypatch.setenv("FEDLAB_STATE_DIR", str(tmp_path / "state"))
    grid = FakeGrid()
    orchestrator.run_federation(grid, stub.StubServer(), rounds=4, patience=50, out_dir=tmp_path / "out")
    steps = json.loads((tmp_path / "out" / "stub_result.json").read_text())["optimizer_steps"]
    assert steps == [1, 2, 3, 4]
    assert len(list((tmp_path / "state").glob("optimizer-run1-node*.npz"))) == 2  # one file per node
    assert all("optimizer_meta" not in ctx.state for ctx in grid.contexts.values())  # nothing went through gRPC state


def test_a_new_run_does_not_reuse_old_optimizer_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDLAB_STATE_DIR", str(tmp_path / "state"))
    first = FakeGrid()
    orchestrator.run_federation(first, stub.StubServer(), rounds=2, patience=50, out_dir=tmp_path / "a")
    second = FakeGrid()
    for ctx in second.contexts.values():
        ctx.run_id = 2
    orchestrator.run_federation(second, stub.StubServer(), rounds=2, patience=50, out_dir=tmp_path / "b")
    assert json.loads((tmp_path / "b" / "stub_result.json").read_text())["optimizer_steps"] == [1, 2]
    assert not list((tmp_path / "state").glob("optimizer-run1-*"))  # superseded files are removed


def test_aggregation_is_weighted_33_to_1(tmp_path):
    orchestrator.run_federation(FakeGrid(), stub.StubServer(), rounds=1, patience=5, out_dir=tmp_path)
    # one round from zero: tcga moves to 0.5, gtex to -0.5; weights 33:1 -> (33*0.5 - 0.5) / 34
    best = json.loads((tmp_path / "stub_result.json").read_text())["best_arrays"][0]
    assert best == pytest.approx([(33 * 0.5 - 0.5) / 34] * 4, rel=1e-5)


def test_early_stopping_stops_and_keeps_the_best_round(tmp_path):
    class Rising(stub.StubServer):
        calls = 0

        def combine_validation(self, by_site):
            Rising.calls += 1
            loss = 1.0 if Rising.calls == 1 else 1.0 + Rising.calls  # best is round 1, then it only gets worse
            return stub.Validation(loss, {})

    history, stopper = orchestrator.run_federation(FakeGrid(), Rising(), rounds=50, patience=3, out_dir=tmp_path)
    assert len(history) == 4  # round 1 (best) + 3 stale rounds
    assert stopper.best.round == 1
    assert json.loads((tmp_path / "stub_result.json").read_text())["best_round"] == 1


def test_early_stopper_patience_counts_only_non_improvements():
    stopper = logic.EarlyStopper(patience=2)
    arrays = [np.zeros(1)]
    assert [stopper.update(r, loss, arrays, {}) for r, loss in enumerate([3.0, 2.0, 2.5, 1.0, 1.5, 1.6], 1)] == [
        False, False, False, False, False, True,
    ]
    assert stopper.best.round == 4


def test_a_failed_client_aborts_the_round(tmp_path):
    class Broken(FakeGrid):
        def send_and_receive(self, messages, timeout=None):
            return list(super().send_and_receive(messages, timeout))[:1]  # only one client answers

    with pytest.raises(RuntimeError, match="expected replies from"):
        orchestrator.run_federation(Broken(), stub.StubServer(), rounds=1, patience=1, out_dir=tmp_path)


def test_optimizer_state_round_trips_through_packing():
    state = {
        "state": {0: {"step": np.float32(7), "exp_avg": np.arange(6, dtype=np.float32).reshape(2, 3), "exp_avg_sq": np.ones(4)}},
        "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), "params": [0]}],
    }
    packed = logic.pack_optimizer_state(state)
    restored = logic.unpack_optimizer_state(OptimizerState(packed.arrays, packed.meta))
    assert restored["state"][0]["exp_avg"].shape == (2, 3)
    assert restored["state"][0]["step"].shape == ()
    assert float(restored["state"][0]["step"]) == 7
    np.testing.assert_array_equal(restored["state"][0]["exp_avg"], state["state"][0]["exp_avg"])
    assert restored["param_groups"][0]["betas"] == [0.9, 0.999]  # tuples become lists; Adam accepts both
