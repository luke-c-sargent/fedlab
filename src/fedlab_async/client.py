"""The framework-neutral client: turn one request file into one response file, keeping a per-node state.

The handler gets the request payload and the node's state (bytes), and returns the response payload and the
new state. The state file keeps the state from before the last request, so running the same request again
starts from the same input and overwrites the same response. A request older than the last one this node
processed is refused, because the state it needs is gone. If the handler raises, nothing is written.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

from .envelope import Header, read, write
from .layout import Exchange

Handler = Callable[[Header, bytes, bytes], tuple[bytes, bytes]]  # (request header, payload, state) -> (response, state)


def _load(path: Path) -> tuple[Header, bytes, bytes] | None:
    """(header of the last request processed, state before it, state after it)."""
    if not path.exists():
        return None
    header, payload = read(path)
    (size,) = struct.unpack(">Q", payload[:8])
    return header, payload[8 : 8 + size], payload[8 + size :]


def run_request(request_path: Path, handler: Handler, exchange: Exchange, state_dir: Path) -> Path:
    header, payload = read(request_path)
    if header.kind != "request":
        raise ValueError(f"{request_path.name}: not a request file")
    state_file = Path(state_dir) / f"client-{header.run_id}-n{header.node_id}.state"

    before = b""
    last = _load(state_file)
    if last is not None:
        last_header, last_before, last_after = last
        if header.nonce == last_header.nonce:
            before = last_before  # a re-run of the last request: start from the same input
        elif header.seq <= last_header.seq:
            raise ValueError(
                f"{request_path.name}: this node already ran round {last_header.round} {last_header.phase}; "
                "an older request cannot be run again"
            )
        else:
            before = last_after

    response, after = handler(header, payload, before)

    write(state_file, Header("client-state", header.run_id, header.round, header.phase, header.node_id, header.nonce,
                             seq=header.seq), struct.pack(">Q", len(before)) + before + after)
    out = exchange.response(header.round, header.phase, header.node_id)
    write(out, Header("response", header.run_id, header.round, header.phase, header.node_id, header.nonce, seq=header.seq),
          response)
    return out
