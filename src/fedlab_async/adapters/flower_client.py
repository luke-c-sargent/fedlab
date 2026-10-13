"""Client side of a Flower run: call a `ClientApp` on one request file, exactly as the SuperNode would.

`Context.state` persists between requests (protobuf, not pickle); the idempotence and ordering rules are the
core client's (see `fedlab_async.client`).
"""

from __future__ import annotations

from pathlib import Path

from flwr.app import Context, RecordDict, UserConfig
from flwr.clientapp import ClientApp
from flwr.common.serde import message_from_proto, message_to_proto, recorddict_from_proto, recorddict_to_proto
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict

from ..client import run_request
from ..envelope import Header
from ..layout import Exchange


def _decode_state(raw: bytes) -> RecordDict:
    if not raw:
        return RecordDict()
    proto = ProtoRecordDict()
    proto.ParseFromString(raw)
    return recorddict_from_proto(proto)


def run_client(
    request_path: Path, clientapp: ClientApp, exchange: Exchange, state_dir: Path,
    node_config: UserConfig | None = None, run_config: UserConfig | None = None,
) -> Path:
    def handler(header: Header, payload: bytes, state: bytes) -> tuple[bytes, bytes]:
        message = message_from_proto(ProtoMessage.FromString(payload))
        context = Context(
            run_id=message.metadata.run_id, node_id=header.node_id, node_config=node_config or {},
            state=_decode_state(state), run_config=run_config or {},
        )
        reply = clientapp(message, context)
        return message_to_proto(reply).SerializeToString(), recorddict_to_proto(context.state).SerializeToString()

    return run_request(request_path, handler, exchange, state_dir)
