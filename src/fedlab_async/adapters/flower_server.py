"""A Flower `Strategy` as a `Program`: requests are Flower `Message`s (protobuf), the strategy sees them as usual.

`FlowerProgram.plan` asks the strategy for this phase's messages (with a recording grid that only knows the
node ids); `absorb` hands the replies to `aggregate_train` / `aggregate_evaluate`. Optional `Hooks` add
per-round logic around the strategy, such as early stopping.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Message, Metadata, RecordDict
from flwr.app.message.message import make_message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.result import Result
from flwr.supercore.task_identity import TaskIdentity

from ..layout import Exchange
from ..server import Plan, Program, Server, Slot, Timing

NO_EXPIRY = 10 * 365 * 24 * 3600  # message TTL in seconds; requests may wait a long time for a client


class Hooks:
    """Optional per-round logic around the strategy; subclass or duck-type it. Pickled with the run state.

    The defaults do nothing, so a bare strategy behaves exactly like `Strategy.start()`.
    """

    def config(self, server_round: int, phase: str) -> ConfigRecord | None:
        """Config sent to clients this phase; None means the fixed config given to `create`."""
        return None

    def node_config(self, server_round: int, phase: str, node_id: int) -> dict | None:
        """Extra config for one node, merged over the shared config in that node's request only
        (for example its data assignment). Values must be what a `ConfigRecord` accepts."""
        return None

    def round_started(self, server_round: int) -> None: ...

    def check_replies(self, server_round: int, phase: str, replies: list[Message]) -> None:
        """Called before the strategy aggregates a phase; raise to reject the replies."""

    def train_aggregated(self, server_round: int, replies: list[Message], aggregated: ArrayRecord | None) -> ArrayRecord | None:
        """Called with the strategy's aggregate; return the model to use (None keeps the previous one)."""
        return aggregated

    def evaluated(self, server_round: int, replies: list[Message], arrays: ArrayRecord, timing: dict) -> bool:
        """Return True to stop after this round. `timing` has `elapsed_seconds` and `server_seconds`."""
        return False

    def finished(self) -> None: ...


class _RecordingGrid(Grid):
    """Just enough Grid for `configure_*`: strategies ask it which nodes exist, and nothing else."""

    def __init__(self, node_ids: list[int], run_id: int):
        self._node_ids, self._run_id = node_ids, run_id
        # Flower 1.40 builds a Message's sender from process-wide state; build_request rewrites the metadata anyway
        TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = run_id, 0, 0

    def set_run(self, run) -> None:
        raise NotImplementedError

    @property
    def run(self):
        raise NotImplementedError("the file-backed grid has no live Run")

    def create_message(self, content: RecordDict, message_type: str, dst_node_id: int, group_id: str, ttl=None) -> Message:
        return Message(content, dst_node_id=dst_node_id, message_type=message_type, group_id=group_id, ttl=ttl)

    def get_node_ids(self) -> Iterable[int]:
        return list(self._node_ids)

    def push_messages(self, messages):
        raise NotImplementedError("use emit()/collect(), not live sends")

    pull_messages = send_and_receive = push_messages


def _int_id(run_id: str) -> int:
    return int(run_id[:15], 16)


def _message_id(run_id: str, round: int, phase: str, node_id: int) -> str:
    return f"{run_id}-r{round}-{phase}-n{node_id}"


def build_request(run_id: str, round: int, phase: str, node_id: int, message_type: str, content: RecordDict) -> bytes:
    """A request payload for one node. `message_type` is a Flower type such as "train" or "query.prepare"."""
    request = make_message(
        Metadata(
            run_id=_int_id(run_id), message_id=_message_id(run_id, round, phase, node_id), src_node_id=0,
            dst_node_id=node_id, reply_to_message_id="", group_id=str(round), created_at=time.time(), ttl=NO_EXPIRY,
            message_type=message_type,
        ),
        content=content,
    )
    return message_to_proto(request).SerializeToString()


def check_reply(slot: Slot, payload: bytes) -> None:
    """Raise ValueError unless `payload` is a Flower reply to exactly this request."""
    reply = _decode(payload)
    if reply.metadata.reply_to_message_id != _message_id(slot.run_id, slot.round, slot.phase, slot.node_id):
        raise ValueError(f"reply is not to this request (round {slot.round} {slot.phase}, node {slot.node_id})")


def _decode(payload: bytes) -> Message:
    try:
        return message_from_proto(ProtoMessage.FromString(payload))
    except Exception as error:  # protobuf raises its own error types; callers expect ValueError
        raise ValueError(f"not a Flower message ({error})") from error


class FlowerProgram(Program):
    def __init__(
        self, strategy: Strategy, initial_arrays: ArrayRecord, node_ids: list[int], num_rounds: int,
        train_config: ConfigRecord | None = None, evaluate_config: ConfigRecord | None = None, hooks: Hooks | None = None,
    ):
        self.strategy, self.arrays, self.node_ids, self.num_rounds = strategy, initial_arrays, sorted(node_ids), num_rounds
        self.train_config = train_config if train_config is not None else ConfigRecord()
        self.evaluate_config = evaluate_config if evaluate_config is not None else ConfigRecord()
        self.hooks = hooks if hooks is not None else Hooks()
        self.round, self.phase = 1, "train"  # phase: "train" | "evaluate" | "done"
        self.result = Result()

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def position(self) -> tuple[int, str]:
        return self.round, self.phase

    def plan(self, run_id: str) -> Plan:
        grid = _RecordingGrid(self.node_ids, _int_id(run_id))
        configure = self.strategy.configure_train if self.phase == "train" else self.strategy.configure_evaluate
        config = self.hooks.config(self.round, self.phase)
        if config is None:
            config = self.train_config if self.phase == "train" else self.evaluate_config
        if self.phase == "train":
            self.hooks.round_started(self.round)
        requests: dict[int, bytes] = {}
        for message in configure(self.round, self.arrays, config, grid):
            node_id = message.metadata.dst_node_id
            content = message.content
            node_config = getattr(self.hooks, "node_config", None)  # optional: duck-typed hooks need not have it
            extra = node_config(self.round, self.phase, node_id) if node_config else None
            if extra:  # strategies share one RecordDict between nodes, so copy before changing it
                key = getattr(self.strategy, "configrecord_key", "config")
                content = RecordDict({k: v for k, v in content.items()})
                content[key] = ConfigRecord({**dict(content[key]), **extra})
            requests[node_id] = build_request(run_id, self.round, self.phase, node_id, message.metadata.message_type, content)
        return Plan(self.round, self.phase, requests)

    def validate(self, slot: Slot, payload: bytes) -> None:
        check_reply(slot, payload)

    def absorb(self, run_id: str, round: int, phase: str, responses: dict[int, bytes], timing: Timing) -> None:
        replies = [_decode(responses[n]) for n in sorted(responses)]
        self.hooks.check_replies(round, phase, replies)
        if phase == "train":
            arrays, metrics = self.strategy.aggregate_train(round, replies)
            arrays = self.hooks.train_aggregated(round, replies, arrays)
            if arrays is not None:
                self.result.arrays = self.arrays = arrays
            if metrics is not None:
                self.result.train_metrics_clientapp[round] = metrics
            self.phase = "evaluate"
        else:
            metrics = self.strategy.aggregate_evaluate(round, replies)
            if metrics is not None:
                self.result.evaluate_metrics_clientapp[round] = metrics
            stop = self.hooks.evaluated(
                round, replies, self.arrays,
                {"elapsed_seconds": timing.elapsed_seconds, "server_seconds": timing.server_seconds},
            )
            self.round += 1
            self.phase = "train" if self.round <= self.num_rounds and not stop else "done"
            if self.phase == "done":
                self.hooks.finished()


class FlowerServer(Server):
    def __init__(self, state_path: Path, exchange: Exchange, validate_reply: Callable[[Message], None] | None = None):
        check = None if validate_reply is None else (lambda slot, payload: validate_reply(_decode(payload)))
        super().__init__(state_path, exchange, check)

    @classmethod
    def create(  # type: ignore[override]
        cls, state_path: Path, exchange: Exchange, strategy: Strategy, initial_arrays: ArrayRecord, node_ids: list[int],
        num_rounds: int, train_config: ConfigRecord | None = None, evaluate_config: ConfigRecord | None = None,
        validate_reply: Callable[[Message], None] | None = None, hooks: Hooks | None = None,
    ) -> FlowerServer:
        program = FlowerProgram(strategy, initial_arrays, node_ids, num_rounds, train_config, evaluate_config, hooks)
        return super().create(state_path, exchange, program, validate_reply=validate_reply)

    @property
    def result(self) -> Result:
        return self.state.program.result
