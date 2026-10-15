"""A neutral numpy strategy: each round, the model moves by the mean of what the nodes add to it.

The same `factory` and `client_fn` run in process, as files (`fedlab-async`), and on live Flower.
"""

import json
from pathlib import Path

import numpy as np

from fedlab_core import Payload, Plan, Strategy


class MeanShift(Strategy):
    def __init__(self, rounds: int, nodes: list[int], out: Path):
        self.rounds, self.nodes, self.out = rounds, nodes, out
        self.round, self.w, self.calls = 1, np.zeros(3), {}

    @property
    def done(self) -> bool:
        return self.round > self.rounds

    def position(self) -> tuple[int, str]:
        return self.round, "train"

    def plan(self) -> Plan:
        return Plan(self.round, "train", {n: Payload({"w": self.w}) for n in self.nodes})

    def absorb(self, round, phase, replies) -> None:
        self.w = np.mean([r.arrays["w"] for r in replies.values()], axis=0)
        self.calls = {n: r.meta["calls"] for n, r in replies.items()}
        self.round += 1
        if self.done:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "result.json").write_text(json.dumps({"w": self.w.tolist(), "calls": self.calls}, indent=2))


def client_fn(phase, request, state, node_id):
    calls = int(state.meta.get("calls", 0)) + 1  # proves client state survives between exchanges
    return Payload({"w": request.arrays["w"] + node_id}, {"calls": calls}), Payload(meta={"calls": calls})


def factory(options, run_dir):
    return MeanShift(int(options["rounds"]), [int(n) for n in str(options["nodes"]).split(",")], Path(run_dir))
