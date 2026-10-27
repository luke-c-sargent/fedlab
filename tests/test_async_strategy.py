"""A non-Flower strategy (linear regression, FedAvg with server momentum and early stopping) run as files."""

import random
import subprocess
import sys

import numpy as np
import pytest

from fedlab_async.adapters.strategy import StrategyServer, run_strategy_client
from fedlab_async.layout import Exchange
from fedlab_core import Payload, Plan, Strategy, decode, encode, run_in_process

TRUE_W = np.array([1.0, -2.0, 0.5])
NODES = [1, 2, 3]


def local_data(node_id):
    rng = np.random.default_rng(node_id)
    x = rng.normal(size=(20 + 10 * node_id, 3))
    return x, x @ TRUE_W + 0.05 * rng.normal(size=len(x))


def client_fn(phase, request, state, node_id):
    w = request.arrays["w"].copy()
    x, y = local_data(node_id)
    for _ in range(int(request.meta["local_steps"])):
        w -= request.meta["lr"] * 2 * x.T @ (x @ w - y) / len(y)
    rounds = int(state.meta.get("rounds", 0)) + 1  # client state carried between rounds
    return Payload({"w": w}, {"n": len(y), "rounds": rounds}), Payload({}, {"rounds": rounds})


class MomentumFedAvg(Strategy):
    def __init__(self, num_rounds=8, lr=0.05, local_steps=5, momentum=0.5, tol=1e-4):
        self.num_rounds, self.lr, self.local_steps, self.momentum, self.tol = num_rounds, lr, local_steps, momentum, tol
        self.round, self.stop = 1, False
        self.w, self.velocity = np.zeros(3), np.zeros(3)
        self.rounds_seen = []

    @property
    def done(self):
        return self.round > self.num_rounds or self.stop

    def position(self):
        return self.round, "train"

    def plan(self):
        return Plan(self.round, "train", {n: Payload({"w": self.w}, {"lr": self.lr, "local_steps": self.local_steps}) for n in NODES})

    def absorb(self, round, phase, replies):
        total = sum(r.meta["n"] for r in replies.values())
        average = sum(r.meta["n"] * r.arrays["w"] for r in replies.values()) / total
        self.velocity = self.momentum * self.velocity + (average - self.w)
        new = self.w + self.velocity
        self.stop = bool(np.linalg.norm(new - self.w) < self.tol)
        self.w = new
        self.rounds_seen.append({n: r.meta["rounds"] for n, r in replies.items()})
        self.round += 1


def in_memory(num_rounds):
    return run_in_process(MomentumFedAvg(num_rounds), client_fn)


def run_files(tmp_path, num_rounds, shuffle_seed=0):
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    StrategyServer.create(state, ex, MomentumFedAvg(num_rounds))
    order = random.Random(shuffle_seed)
    while not StrategyServer(state, ex).done:
        requests = StrategyServer(state, ex).emit()  # a new server process for every step
        order.shuffle(requests)  # clients run in any order
        for request in requests:
            run_strategy_client(request, client_fn, ex, tmp_path / "clients")
        assert StrategyServer(state, ex).collect().advanced
    return StrategyServer(state, ex), ex


def test_matches_an_in_memory_loop(tmp_path):
    server, _ = run_files(tmp_path, 8)
    reference = in_memory(8)
    np.testing.assert_allclose(server.strategy.w, reference.w)
    assert server.strategy.rounds_seen == reference.rounds_seen
    assert server.strategy.rounds_seen[-1] == {1: 8, 2: 8, 3: 8}


def test_learns_and_stops_early(tmp_path):
    server, _ = run_files(tmp_path, 500)
    assert server.strategy.stop and server.strategy.round < 500
    np.testing.assert_allclose(server.strategy.w, TRUE_W, atol=0.1)
    assert server.status().phase == "done"
    assert len(server.strategy.rounds_seen) == len(in_memory(500).rounds_seen) == server.strategy.round - 1


def test_rejects_a_response_from_another_run(tmp_path):
    a, ex_a = tmp_path / "a", Exchange(tmp_path / "a" / "ex")
    b, ex_b = tmp_path / "b", Exchange(tmp_path / "b" / "ex")
    for root, ex in ((a, ex_a), (b, ex_b)):
        StrategyServer.create(root / "state.pkl", ex, MomentumFedAvg(2))
        for request in StrategyServer(root / "state.pkl", ex).emit():
            run_strategy_client(request, client_fn, ex, root / "clients")
    ex_a.response(1, "train", 2).write_bytes(ex_b.response(1, "train", 2).read_bytes())
    result = StrategyServer(a / "state.pkl", ex_a).collect()
    assert not result.advanced and result.missing == [2] and "run_id" in next(iter(result.rejected.values()))


class Picky(MomentumFedAvg):
    def validate(self, round, phase, node_id, reply):
        if node_id == 2:
            raise ValueError("node 2 sent something unusable")


def test_algorithm_validate_hook_rejects_one_node(tmp_path):
    ex = Exchange(tmp_path / "ex")
    StrategyServer.create(tmp_path / "state.pkl", ex, Picky(2))
    server = StrategyServer(tmp_path / "state.pkl", ex)
    for request in server.emit():
        run_strategy_client(request, client_fn, ex, tmp_path / "clients")
    result = server.collect()
    assert not result.advanced and result.missing == [2] and "unusable" in next(iter(result.rejected.values()))
    assert (ex.root / "rejected").exists() and server.status().waiting_on == [2]


def test_core_and_strategy_adapter_do_not_import_flower():
    code = (
        "import sys, fedlab_async.server, fedlab_async.client, fedlab_async.adapters.strategy\n"
        "bad = sorted(m for m in sys.modules if m == 'flwr' or m.startswith('flwr.'))\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
