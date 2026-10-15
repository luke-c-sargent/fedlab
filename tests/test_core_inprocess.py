"""The neutral contract on a numpy strategy: phases, node selection, blobs, client state, own randomness."""

import ast
from pathlib import Path

import numpy as np
import pytest

import fedlab_core
from fedlab_core import Payload, Plan, Strategy, decode, encode, run_in_process

TRUE_W = np.array([1.0, -2.0, 0.5])
NODES = [1, 2, 3]


def client(phase, request, state, node_id):
    rounds = int(state.meta.get("rounds", 0))
    if phase == "prepare":
        return Payload(blobs={"echo": request.blobs["config"][::-1]}), Payload(meta={"rounds": 0})
    rng = np.random.default_rng(node_id)
    x = rng.normal(size=(30, 3))
    y = x @ TRUE_W
    w = request.arrays["w"].copy()
    for _ in range(5):
        w -= 0.05 * 2 * x.T @ (x @ w - y) / len(y)
    return Payload({"w": w}, {"n": len(y)}), Payload(meta={"rounds": rounds + 1})


class SampledFedAvg(Strategy):
    """A prepare exchange, then FedAvg on two of the three nodes each round, chosen with its own generator."""

    def __init__(self, rounds=6, seed=0):
        self.rounds, self.round, self.w, self.phase = rounds, 1, np.zeros(3), "prepare"
        self.rng, self.asked, self.echoed = np.random.default_rng(seed), [], None

    @property
    def done(self):
        return self.round > self.rounds

    def plan(self):
        if self.phase == "prepare":
            return Plan(0, "prepare", {NODES[0]: Payload(blobs={"config": b"abc"})})
        chosen = sorted(int(n) for n in self.rng.choice(NODES, size=2, replace=False))
        self.asked.append(chosen)
        return Plan(self.round, "train", {n: Payload({"w": self.w}) for n in chosen})

    def validate(self, round, phase, node_id, reply):
        if phase == "train" and reply.arrays["w"].shape != (3,):
            raise ValueError("wrong shape")

    def absorb(self, round, phase, replies):
        if phase == "prepare":
            self.echoed, self.phase = replies[NODES[0]].blobs["echo"], "train"
            return
        total = sum(r.meta["n"] for r in replies.values())
        self.w = sum(r.meta["n"] * r.arrays["w"] for r in replies.values()) / total
        self.round += 1


def test_runs_phases_selection_and_blobs():
    s = run_in_process(SampledFedAvg(), client)
    assert s.echoed == b"cba"  # bytes survive the round trip
    assert np.allclose(s.w, TRUE_W, atol=0.2)
    assert len(s.asked) == 6 and all(len(a) == 2 for a in s.asked)  # only the chosen nodes were asked


def test_same_seed_gives_same_result_and_global_rng_is_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = run_in_process(SampledFedAvg(seed=7), client)
    assert (np.random.get_state()[1] == before).all()
    b = run_in_process(SampledFedAvg(seed=7), client)
    assert a.asked == b.asked and np.array_equal(a.w, b.w)


def test_a_rejected_reply_stops_the_run():
    bad = lambda phase, request, state, node: (Payload({"w": np.zeros(2)}), Payload()) if phase == "train" else client(phase, request, state, node)
    with pytest.raises(ValueError, match="wrong shape"):
        run_in_process(SampledFedAvg(), bad)


def test_payload_round_trip_and_object_arrays_are_refused():
    p = Payload({"a": np.arange(4, dtype=np.float32)}, {"k": [1, 2]}, {"b": b"\x00\xff"})
    q = decode(encode(p))
    assert q.arrays["a"].dtype == np.float32 and q.meta == {"k": [1, 2]} and q.blobs == {"b": b"\x00\xff"}
    with pytest.raises(ValueError):
        encode(Payload({"o": np.array([object()])}))


def test_core_imports_no_runner_and_flower_only_in_its_subpackage():
    root = Path(fedlab_core.__file__).parent
    for path in root.rglob("*.py"):
        forbidden = {"fedlab_async", "fedlab", "torch"} | (set() if "flower" in path.relative_to(root).parts else {"flwr"})
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            assert not {n.split(".")[0] for n in names} & forbidden, f"{path.name} imports {names}"
