"""Plain adapter: no framework. A federated algorithm is two methods, a client is one function.

Payloads are numpy arrays plus JSON-able metadata, stored as `.npz` (never pickle). One exchange per round:
the server sends each node a `Payload`, each node answers with a `Payload`, `aggregate` combines them.

The `Algorithm` is pickled with the run state, so keep its state on the instance. Client functions should
be plain module-level functions; their state between rounds is the `Payload` they return.
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..client import run_request
from ..envelope import Header
from ..layout import Exchange
from ..server import Plan, Program, Server, Slot, Timing

_META = "__meta__"


@dataclass
class Payload:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # anything json.dumps accepts


def encode(payload: Payload) -> bytes:
    arrays = {}
    for name, array in payload.arrays.items():
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError(f"array {name!r} has dtype {array.dtype}; only numeric and string arrays are allowed")
        arrays[f"a_{name}"] = array
    buffer = io.BytesIO()
    np.savez(buffer, **arrays, **{_META: np.array(json.dumps(payload.meta))})
    return buffer.getvalue()


def decode(raw: bytes) -> Payload:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            meta = json.loads(str(saved[_META]))
            return Payload({name[2:]: saved[name] for name in saved.files if name.startswith("a_")}, meta)
    except Exception as error:
        raise ValueError(f"not a plain payload ({error})") from error


class Algorithm(ABC):
    """Server side of a federated algorithm. Set `self.stop = True` in `aggregate` to end the run early."""

    stop: bool = False

    @abstractmethod
    def configure(self, round: int) -> dict[int, Payload]:
        """What each participating node gets this round (node id -> payload)."""

    @abstractmethod
    def aggregate(self, round: int, responses: dict[int, Payload]) -> None:
        """Combine the responses of all nodes that were asked."""

    def validate(self, round: int, node_id: int, response: Payload) -> None:
        """Raise ValueError to reject one response."""

    def finished(self) -> None:
        """Called once, after the last round."""


class PlainProgram(Program):
    def __init__(self, algorithm: Algorithm, num_rounds: int):
        self.algorithm, self.num_rounds, self.round = algorithm, num_rounds, 1

    @property
    def done(self) -> bool:
        return self.round > self.num_rounds or self.algorithm.stop

    def position(self) -> tuple[int, str]:
        return self.round, "round"

    def plan(self, run_id: str) -> Plan:
        return Plan(self.round, "round", {n: encode(p) for n, p in self.algorithm.configure(self.round).items()})

    def validate(self, slot: Slot, payload: bytes) -> None:
        self.algorithm.validate(slot.round, slot.node_id, decode(payload))

    def absorb(self, run_id: str, round: int, phase: str, responses: dict[int, bytes], timing: Timing) -> None:
        self.algorithm.aggregate(round, {n: decode(raw) for n, raw in responses.items()})
        self.round += 1
        if self.done:
            self.algorithm.finished()


class PlainServer(Server):
    @classmethod
    def create(cls, state_path: Path, exchange: Exchange, algorithm: Algorithm, num_rounds: int, **kwargs) -> PlainServer:  # type: ignore[override]
        return super().create(state_path, exchange, PlainProgram(algorithm, num_rounds), **kwargs)

    @property
    def algorithm(self) -> Algorithm:
        return self.state.program.algorithm


ClientFn = Callable[[Payload, Payload, int], tuple[Payload, Payload]]  # (request, state, node id) -> (response, state)


def run_plain_client(request_path: Path, fn: ClientFn, exchange: Exchange, state_dir: Path) -> Path:
    def handler(header: Header, payload: bytes, state: bytes) -> tuple[bytes, bytes]:
        response, new_state = fn(decode(payload), decode(state) if state else Payload(), header.node_id)
        return encode(response), encode(new_state)

    return run_request(request_path, handler, exchange, state_dir)
