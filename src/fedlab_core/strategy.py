"""The framework-neutral contract: a `Strategy` decides what each node gets and combines what comes back.

A run is a sequence of exchanges. Each exchange has a `(round, phase)` name that is unique within the run, a
request for each node that is asked, and one reply from each of them. Runners (in process, files, live Flower)
only move payloads; they never look inside.

Rules a strategy follows so every runner gives the same result:
- Keep all state on the instance. File runs pickle it between steps.
- Own your randomness: hold a seeded `numpy.random.Generator`; never use the global `random` or `numpy.random`.
- Never assume every node answers: `absorb` gets only the nodes that were asked in `plan`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .payload import Payload, decode, encode


@dataclass
class Plan:
    round: int
    phase: str  # "train", "evaluate", "prepare", ... any name
    requests: dict[int, Payload]  # node id -> request; only these nodes are asked


class Strategy(ABC):
    @property
    @abstractmethod
    def done(self) -> bool: ...

    def position(self) -> tuple[int, str]:
        """(round, phase) of the next exchange, for status displays. Must not change any state."""
        return 0, "next"

    @abstractmethod
    def plan(self) -> Plan:
        """The next exchange. Only called when not done."""

    def validate(self, round: int, phase: str, node_id: int, reply: Payload) -> None:
        """Raise ValueError to reject one reply."""

    @abstractmethod
    def absorb(self, round: int, phase: str, replies: dict[int, Payload]) -> None:
        """Every node that was asked has answered: update the strategy's state."""


# (phase, request, state, node id) -> (reply, new state). State is whatever the node keeps between exchanges.
ClientFn = Callable[[str, Payload, Payload, int], tuple[Payload, Payload]]


def run_in_process(strategy: Strategy, client: ClientFn) -> Strategy:
    """Run a strategy to the end in this process. Payloads still go through `encode`, as they do in a file run."""
    states: dict[int, Payload] = {}
    while not strategy.done:
        plan = strategy.plan()
        replies = {}
        for node, request in plan.requests.items():
            reply, state = client(plan.phase, decode(encode(request)), decode(encode(states.get(node, Payload()))), node)
            states[node], replies[node] = state, decode(encode(reply))
            strategy.validate(plan.round, plan.phase, node, replies[node])
        strategy.absorb(plan.round, plan.phase, replies)
    return strategy
