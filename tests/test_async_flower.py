import random

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp.strategy import FedAvg

from fedlab_async.adapters.flower_client import run_client
from fedlab_async.adapters.flower_server import FlowerServer, _RecordingGrid
from fedlab_async.layout import Exchange

WEIGHTS = {1: 3, 2: 1, 3: 2}  # node id -> num-examples

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    calls = int(context.state["calls"]["n"]) + 1 if "calls" in context.state else 1
    context.state["calls"] = ConfigRecord({"n": calls})  # persists between rounds
    shift = context.node_id * 0.5 + calls * 0.01
    arrays = [a + shift for a in msg.content["arrays"].to_numpy_ndarrays()]
    metrics = {"num-examples": WEIGHTS[context.node_id], "calls": calls}
    return Message(RecordDict({"arrays": ArrayRecord(arrays), "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    loss = float(msg.content["arrays"].to_numpy_ndarrays()[0].mean()) * context.node_id
    return Message(RecordDict({"metrics": MetricRecord({"num-examples": WEIGHTS[context.node_id], "loss": loss})}), reply_to=msg)


class LiveGrid(_RecordingGrid):
    """In-process stand-in for the SuperLink: runs the ClientApp directly. The reference for equivalence."""

    def __init__(self, node_ids=(1, 2)):
        super().__init__(list(node_ids), 1)
        self.states = {n: RecordDict() for n in node_ids}

    def send_and_receive(self, messages, *, timeout=None):
        replies = []
        for m in messages:
            n = m.metadata.dst_node_id
            replies.append(app(m, Context(1, n, {}, self.states[n], {})))
        return replies


def strategy(fraction=1.0):
    return FedAvg(fraction_train=fraction, fraction_evaluate=fraction, min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2)


def run_through_files(tmp_path, rounds, node_ids=(1, 2), fraction=1.0, between=lambda: None):
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    FlowerServer.create(state, ex, strategy(fraction), ArrayRecord([np.zeros(3)]), list(node_ids), rounds)
    server = FlowerServer(state, ex)
    while not server.done:
        server = FlowerServer(state, ex)  # a new process for every step
        for request in server.emit():
            run_client(request, app, ex, tmp_path / "clients")
        between()
        server = FlowerServer(state, ex)
        assert server.collect().advanced
    return FlowerServer(state, ex)


def test_matches_live_run(tmp_path):
    random.seed(0)
    live = strategy().start(LiveGrid(), ArrayRecord([np.zeros(3)]), num_rounds=4)
    random.seed(0)
    files = run_through_files(tmp_path, 4).result
    np.testing.assert_allclose(files.arrays.to_numpy_ndarrays()[0], live.arrays.to_numpy_ndarrays()[0])
    for r in range(1, 5):
        assert dict(files.train_metrics_clientapp[r]) == pytest.approx(dict(live.train_metrics_clientapp[r]))
        assert dict(files.evaluate_metrics_clientapp[r]) == pytest.approx(dict(live.evaluate_metrics_clientapp[r]))
    assert files.train_metrics_clientapp[4]["calls"] == 4  # client state survived 4 separate invocations


def test_result_is_not_trivial(tmp_path):
    arrays = run_through_files(tmp_path, 4).result.arrays.to_numpy_ndarrays()[0]
    assert np.all(arrays > 1.0)  # moved a long way from zeros; equality above is not vacuous


def fresh(tmp_path, rounds=2):
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    return FlowerServer.create(state, ex, strategy(), ArrayRecord([np.zeros(3)]), [1, 2], rounds), ex, state


def answer(server, ex, tmp_path, only=None):
    for request in server.emit():
        if only is None or request.stem.endswith(f"n{only}"):
            run_client(request, app, ex, tmp_path / "clients")


def test_waits_for_all_clients(tmp_path):
    server, ex, _ = fresh(tmp_path)
    answer(server, ex, tmp_path, only=1)
    result = server.collect()
    assert not result.advanced and result.missing == [2]
    assert server.program.phase == "train" and server.program.round == 1
    answer(server, ex, tmp_path, only=2)
    assert server.collect().advanced and server.program.phase == "evaluate"


def test_resume_between_every_step(tmp_path):
    server, ex, state = fresh(tmp_path, rounds=3)
    random.seed(1)
    while not server.done:
        server = FlowerServer(state, ex)  # a brand-new process every time
        answer(server, ex, tmp_path)
        server = FlowerServer(state, ex)
        assert server.collect().advanced
    reference = run_through_files(tmp_path / "ref", 3).result.arrays.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(server.result.arrays.to_numpy_ndarrays()[0], reference)


def test_emit_is_idempotent(tmp_path):
    server, ex, _ = fresh(tmp_path)
    first = server.emit()
    nonces = dict(server.state.pending)
    assert server.emit() == first and server.state.pending == nonces


def test_rejects_response_from_another_run(tmp_path):
    a, ex_a, _ = fresh(tmp_path / "a")
    b, ex_b, _ = fresh(tmp_path / "b")
    answer(a, ex_a, tmp_path / "a")
    answer(b, ex_b, tmp_path / "b")
    foreign = ex_b.response(1, "train", 1).read_bytes()
    ex_a.response(1, "train", 1).write_bytes(foreign)
    result = a.collect()
    assert not result.advanced and result.missing == [1]
    assert "run_id" in next(iter(result.rejected.values()))
    assert list((ex_a.root / "rejected").glob("*.res"))  # quarantined, not deleted


def test_rejects_stale_round_and_rerun_nonce(tmp_path):
    server, ex, _ = fresh(tmp_path)
    answer(server, ex, tmp_path)
    old = {n: ex.response(1, "train", n).read_bytes() for n in (1, 2)}
    assert server.collect().advanced
    answer(server, ex, tmp_path)
    assert server.collect().advanced  # evaluate phase done, now round 2 train
    answer(server, ex, tmp_path)
    ex.response(2, "train", 1).write_bytes(old[1])  # last round's reply, renamed into this round
    result = server.collect()
    assert not result.advanced and result.missing == [1]
    assert any(k in v for v in result.rejected.values() for k in ("round", "nonce"))


def test_rejects_truncated_file(tmp_path):
    server, ex, _ = fresh(tmp_path)
    answer(server, ex, tmp_path)
    path = ex.response(1, "train", 2)
    path.write_bytes(path.read_bytes()[:-10])
    result = server.collect()
    assert result.missing == [2] and "hash" in next(iter(result.rejected.values()))
    answer(server, ex, tmp_path, only=2)  # the client simply runs again
    assert server.collect().advanced


def test_validate_reply_hook_rejects(tmp_path):
    ex = Exchange(tmp_path / "exchange")

    def reject(reply):
        raise ValueError("bad shape")

    server = FlowerServer.create(tmp_path / "s.pkl", ex, strategy(), ArrayRecord([np.zeros(3)]), [1, 2], 1, validate_reply=reject)
    answer(server, ex, tmp_path)
    result = server.collect()
    assert not result.advanced and len(result.rejected) == 2


def test_random_sampling_survives_restarts_and_outside_rng_use(tmp_path):
    ids = (1, 2, 3)
    random.seed(7)
    live = strategy(0.67).start(LiveGrid(ids), ArrayRecord([np.zeros(3)]), num_rounds=5)
    random.seed(7)
    # something else uses the global RNG between every step; the run must not notice
    files = run_through_files(tmp_path, 5, ids, 0.67, between=lambda: [random.random() for _ in range(3)]).result
    np.testing.assert_allclose(files.arrays.to_numpy_ndarrays()[0], live.arrays.to_numpy_ndarrays()[0])
    sampled = {tuple(sorted(p.name for p in (tmp_path / "exchange" / "requests").glob(f"r000{r}-train-*"))) for r in range(1, 6)}
    assert len(sampled) > 1  # different nodes were sampled in different rounds, so the test is not vacuous


def test_steps_leave_the_callers_rng_alone(tmp_path):
    server, ex, _ = fresh(tmp_path)
    random.seed(5)
    before = random.getstate()
    answer(server, ex, tmp_path)
    server.collect()
    assert random.getstate() == before


# ---- deferred use: requests wait, clients run whenever and in any order, collect happens later ----

import fcntl  # noqa: E402

flaky = ClientApp()
FLAKY = {"fail": True}


@flaky.train()
def flaky_train(msg: Message, context: Context) -> Message:
    calls = int(context.state["calls"]["n"]) + 1 if "calls" in context.state else 1
    context.state["calls"] = ConfigRecord({"n": calls})
    if FLAKY["fail"]:
        raise RuntimeError("client crashed after touching its state")
    return Message(
        RecordDict({"arrays": ArrayRecord([np.full(3, float(calls))]), "metrics": MetricRecord({"num-examples": 1, "calls": calls})}),
        reply_to=msg,
    )


def requests_for(server, node):
    return [p for p in server.emit() if p.stem.endswith(f"n{node}")][0]


def test_deferred_run_in_any_order_matches_straight_run(tmp_path):
    random.seed(3)
    reference = run_through_files(tmp_path / "ref", 3, (1, 2, 3)).result.arrays.to_numpy_ndarrays()[0]

    random.seed(3)
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    FlowerServer.create(state, ex, strategy(), ArrayRecord([np.zeros(3)]), [1, 2, 3], 3)
    order = random.Random(0)
    while not FlowerServer(state, ex).done:
        server = FlowerServer(state, ex)  # emit, then everyone goes away
        requests = server.emit()
        del server
        order.shuffle(requests)
        for request in requests:  # clients run in any order, with collect attempts in between
            run_client(request, app, ex, tmp_path / "clients")
            attempt = FlowerServer(state, ex).collect()
            assert attempt.advanced == (request is requests[-1])
    final = FlowerServer(state, ex).result.arrays.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(final, reference)


def test_status_shows_who_is_outstanding(tmp_path):
    server, ex, _ = fresh(tmp_path)
    status = server.status()
    assert (status.phase, status.emitted, status.waiting_on, status.received) == ("train", False, [], [])
    server.emit()
    answer(server, ex, tmp_path, only=1)
    status = server.status()
    assert status.waiting_on == [2] and status.received == [1] and status.emitted
    answer(server, ex, tmp_path, only=2)
    server.collect()
    assert server.status().phase == "evaluate" and not server.status().emitted


def test_collect_before_emit_does_nothing(tmp_path):
    server, ex, _ = fresh(tmp_path)
    assert not server.collect().advanced and server.program.phase == "train"


def test_rerunning_a_request_does_not_apply_client_state_twice(tmp_path):
    server, ex, _ = fresh(tmp_path)
    first = requests_for(server, 1)
    run_client(first, app, ex, tmp_path / "clients")
    run_client(first, app, ex, tmp_path / "clients")  # a human runs the same request again
    answer(server, ex, tmp_path, only=2)
    assert server.collect().advanced
    assert server.result.train_metrics_clientapp[1]["calls"] == 1.0  # not 1.75 (node 1 counted twice)


def test_a_crashed_client_leaves_its_state_alone(tmp_path):
    server, ex, _ = fresh(tmp_path)
    request = requests_for(server, 1)
    FLAKY["fail"] = True
    with pytest.raises(RuntimeError, match="crashed"):
        run_client(request, flaky, ex, tmp_path / "clients")
    assert not ex.response(1, "train", 1).exists()
    FLAKY["fail"] = False
    run_client(request, flaky, ex, tmp_path / "clients")
    run_client(requests_for(server, 2), flaky, ex, tmp_path / "clients")
    assert server.collect().advanced
    assert server.result.train_metrics_clientapp[1]["calls"] == 1.0  # the crashed attempt's increment was discarded


def test_a_client_refuses_an_older_request(tmp_path):
    server, ex, _ = fresh(tmp_path)
    old = requests_for(server, 1)
    answer(server, ex, tmp_path)
    server.collect()
    answer(server, ex, tmp_path)  # evaluate phase
    with pytest.raises(ValueError, match="older request"):
        run_client(old, app, ex, tmp_path / "clients")


def test_a_late_copy_of_an_old_response_is_ignored(tmp_path):
    server, ex, _ = fresh(tmp_path)
    answer(server, ex, tmp_path)
    old = ex.response(1, "train", 1).read_bytes()
    server.collect()
    ex.response(1, "train", 1).write_bytes(old)  # arrives again after the phase is over
    answer(server, ex, tmp_path)
    assert server.collect().advanced and server.program.round == 2


def test_two_server_steps_cannot_run_at_once(tmp_path):
    server, ex, state = fresh(tmp_path)
    answer(server, ex, tmp_path)
    with open(state.with_name(state.name + ".lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="another server step"):
            server.collect()
    assert server.collect().advanced


def test_a_server_object_kept_across_another_processes_progress_does_not_fork_the_run(tmp_path):
    stale, ex, state = fresh(tmp_path)
    answer(stale, ex, tmp_path)
    other = FlowerServer(state, ex)
    assert other.collect().advanced  # round 1 train is aggregated elsewhere
    requests = stale.emit()  # `stale` still believes it is in the train phase
    assert [r.name for r in requests] == ["r0001-evaluate-n1.req", "r0001-evaluate-n2.req"]


# ---- stateful strategies: server-side momentum / adaptive moments must survive pickling between steps ----

from flwr.serverapp.strategy import FedAdagrad, FedAdam, FedAvgM, FedYogi  # noqa: E402
from fedlab_async.adapters.flower_server import Hooks  # noqa: E402

STATEFUL = {
    "fedavgm": lambda: FedAvgM(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2, server_learning_rate=0.7, server_momentum=0.9),
    "fedadam": lambda: FedAdam(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2, eta=0.1, tau=1e-3),
    "fedadagrad": lambda: FedAdagrad(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2, eta=0.1, tau=1e-3),
    "fedyogi": lambda: FedYogi(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2, eta=0.1, tau=1e-3),
}
MOMENTUM_ATTRS = ("momentum_vector", "m_t", "v_t")


def files_with(make, tmp_path, rounds, hooks=None):
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    FlowerServer.create(state, ex, make(), ArrayRecord([np.zeros(3)]), [1, 2], rounds, hooks=hooks)
    while not FlowerServer(state, ex).done:
        server = FlowerServer(state, ex)
        for request in server.emit():
            run_client(request, app, ex, tmp_path / "clients")
        assert FlowerServer(state, ex).collect().advanced
    return FlowerServer(state, ex)


@pytest.mark.parametrize("name", STATEFUL)
def test_stateful_strategy_matches_live_run(name, tmp_path):
    live = STATEFUL[name]().start(LiveGrid(), ArrayRecord([np.zeros(3)]), num_rounds=5)
    server = files_with(STATEFUL[name], tmp_path, 5)
    np.testing.assert_allclose(server.result.arrays.to_numpy_ndarrays()[0], live.arrays.to_numpy_ndarrays()[0], rtol=1e-10)
    kept = [a for a in MOMENTUM_ATTRS if getattr(server.program.strategy, a, None) is not None]
    assert kept, f"{name} should have server-side optimizer state after 5 rounds"
    plain = run_through_files(tmp_path / "plain", 5).result.arrays.to_numpy_ndarrays()[0]
    assert not np.allclose(plain, live.arrays.to_numpy_ndarrays()[0])  # the optimizer state really changes the result


class FailOnce(Hooks):
    """Fails after the strategy has already updated its momentum, once."""

    failed = False

    def train_aggregated(self, server_round, replies, aggregated):
        if server_round == 2 and not FailOnce.failed:
            FailOnce.failed = True
            raise RuntimeError("hook failed after the strategy aggregated")
        return aggregated


def test_a_failure_after_the_strategy_updated_its_momentum_does_not_leak_into_the_retry(tmp_path):
    live = STATEFUL["fedavgm"]().start(LiveGrid(), ArrayRecord([np.zeros(3)]), num_rounds=3)
    FailOnce.failed = False
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    FlowerServer.create(state, ex, STATEFUL["fedavgm"](), ArrayRecord([np.zeros(3)]), [1, 2], 3, hooks=FailOnce())
    raised = 0
    while not FlowerServer(state, ex).done:
        server = FlowerServer(state, ex)
        for request in server.emit():
            run_client(request, app, ex, tmp_path / "clients")
        collector = FlowerServer(state, ex)
        try:
            assert collector.collect().advanced
        except RuntimeError:
            raised += 1
            on_disk = FlowerServer(state, ex).program.strategy.momentum_vector
            now = collector.program.strategy.momentum_vector  # the failed step must not leave this object half-updated
            assert (now is None) == (on_disk is None) and (now is None or np.allclose(now[0], on_disk[0]))
            assert collector.collect().advanced
    assert raised == 1
    final = FlowerServer(state, ex).result.arrays.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(final, live.arrays.to_numpy_ndarrays()[0], rtol=1e-10)
