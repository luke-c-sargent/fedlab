"""A neutral `Strategy` as the body of a Flower `ServerApp`, and a neutral client function as a `ClientApp`.

Each exchange is one Flower query message per node. The request payload travels as `.npz` bytes in a
`ConfigRecord`, so Flower never looks inside it and the strategy sees exactly what it sees in a file run.

Strategy node ids are the ids a strategy plans for (1, 2, ...). Flower gives every SuperNode a random id, so each
SuperNode is started with `--node-config 'node-id=N'` and the server learns the mapping with a `hello` query.
Client state between exchanges is kept in `$FEDLAB_STATE_DIR` (each Flower message runs in a fresh process).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid

from ..payload import Payload, decode, encode
from ..strategy import ClientFn, Strategy

TIMEOUT = 24 * 3600.0


def _reply(msg: Message, content: dict) -> Message:
    return Message(RecordDict({"payload": ConfigRecord(content)}), reply_to=msg)


def _send(grid: Grid, messages: list[Message], timeout: float) -> dict[int, Message]:
    """Replies by the Flower id of the node that sent them; a node that failed or stayed silent is an error."""
    replies = {}
    for reply in grid.send_and_receive(messages, timeout=timeout):
        if reply.has_error():
            raise RuntimeError(f"node {reply.metadata.src_node_id} failed: {reply.error.reason}")
        replies[reply.metadata.src_node_id] = reply
    silent = {m.metadata.dst_node_id for m in messages} - set(replies)
    if silent:
        raise RuntimeError(f"no reply from Flower node(s) {sorted(silent)} within {timeout:.0f}s")
    return replies


def discover(grid: Grid, timeout: float = TIMEOUT) -> dict[int, int]:
    """Strategy node id -> Flower node id, asked of every SuperNode that is online."""
    hello = [
        grid.create_message(RecordDict({"payload": ConfigRecord({})}), "query.hello", flower_id, "hello")
        for flower_id in grid.get_node_ids()
    ]
    nodes = {int(r.content["payload"]["node-id"]): flower_id for flower_id, r in _send(grid, hello, timeout).items()}
    if len(nodes) != len(hello):
        raise RuntimeError("two SuperNodes report the same node-id")
    return nodes


def serve(strategy: Strategy, grid: Grid, timeout: float = TIMEOUT) -> Strategy:
    """Run the strategy to the end on the nodes that are online. The body of a ServerApp."""
    nodes = discover(grid, timeout)
    while not strategy.done:
        plan = strategy.plan()
        offline = sorted(set(plan.requests) - set(nodes))
        if offline:
            raise RuntimeError(f"round {plan.round} {plan.phase}: node(s) {offline} are not online (online: {sorted(nodes)})")
        messages = [
            grid.create_message(
                RecordDict({"payload": ConfigRecord({"npz": encode(request), "phase": plan.phase})}),
                "query.exchange", nodes[node], str(plan.round),
            )
            for node, request in plan.requests.items()
        ]
        by_flower_id = {flower_id: node for node, flower_id in nodes.items()}
        replies = {}
        for flower_id, reply in _send(grid, messages, timeout).items():
            node = by_flower_id[flower_id]
            replies[node] = decode(bytes(reply.content["payload"]["npz"]))
            strategy.validate(plan.round, plan.phase, node, replies[node])
        strategy.absorb(plan.round, plan.phase, replies)
    return strategy


def make_client_app(client_fn: Callable[[Context], ClientFn]) -> ClientApp:
    """A ClientApp that runs `client_fn(context)` for each exchange. `node_config` must hold `node-id`."""
    app = ClientApp()

    @app.query("hello")
    def hello(msg: Message, context: Context) -> Message:
        return _reply(msg, {"node-id": int(context.node_config["node-id"])})

    @app.query("exchange")
    def exchange(msg: Message, context: Context) -> Message:
        node, request = int(context.node_config["node-id"]), dict(msg.content["payload"])
        state_file = Path(os.environ.get("FEDLAB_STATE_DIR", "state")) / f"core-{context.run_id}-n{node}.state"
        state = decode(state_file.read_bytes()) if state_file.exists() else Payload()
        reply, state = client_fn(context)(str(request["phase"]), decode(bytes(request["npz"])), state, node)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_file.with_name(state_file.name + ".tmp")
        temporary.write_bytes(encode(state))
        os.replace(temporary, state_file)
        for old in state_file.parent.glob(f"core-*-n{node}.state"):
            if old != state_file:
                old.unlink(missing_ok=True)  # state from earlier runs is never reused
        return _reply(msg, {"npz": encode(reply)})

    return app
