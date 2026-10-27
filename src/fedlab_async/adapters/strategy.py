"""Run any `fedlab_core.Strategy` as files: the strategy becomes a `Program`, a client function becomes a request handler.

Payloads are `fedlab_core` payloads (`.npz`, never pickle). The strategy is pickled with the run state, so it
keeps its state on the instance and owns its randomness (see `fedlab_core.strategy`).
"""

from __future__ import annotations

from pathlib import Path

from fedlab_core import ClientFn, Payload, Strategy, decode, encode

from ..client import run_request
from ..envelope import Header
from ..layout import Exchange
from ..server import Plan, Program, Server, Slot, Timing


class StrategyProgram(Program):
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    @property
    def done(self) -> bool:
        return self.strategy.done

    def position(self) -> tuple[int, str]:
        return self.strategy.position()

    def plan(self, run_id: str) -> Plan:
        plan = self.strategy.plan()
        return Plan(plan.round, plan.phase, {node: encode(request) for node, request in plan.requests.items()})

    def validate(self, slot: Slot, payload: bytes) -> None:
        self.strategy.validate(slot.round, slot.phase, slot.node_id, decode(payload))

    def absorb(self, run_id: str, round: int, phase: str, responses: dict[int, bytes], timing: Timing) -> None:
        self.strategy.absorb(round, phase, {node: decode(raw) for node, raw in responses.items()})


class StrategyServer(Server):
    @classmethod
    def create(cls, state_path: Path, exchange: Exchange, strategy: Strategy, **kwargs) -> StrategyServer:  # type: ignore[override]
        return super().create(state_path, exchange, StrategyProgram(strategy), **kwargs)

    @property
    def strategy(self) -> Strategy:
        return self.state.program.strategy


def run_strategy_client(request_path: Path, fn: ClientFn, exchange: Exchange, state_dir: Path) -> Path:
    def handler(header: Header, payload: bytes, state: bytes) -> tuple[bytes, bytes]:
        reply, new_state = fn(header.phase, decode(payload), decode(state) if state else Payload(), header.node_id)
        return encode(reply), encode(new_state)

    return run_request(request_path, handler, exchange, state_dir)
