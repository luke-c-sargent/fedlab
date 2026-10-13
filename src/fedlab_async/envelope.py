"""One exchange file: a JSON header line, then the payload bytes. Written atomically; never pickled.

The header is what `collect` trusts, never the file name. `signature` is reserved for later (empty now).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1


class EnvelopeError(ValueError):
    """The file is unreadable, truncated, or its header does not match its payload."""


@dataclass(frozen=True)
class Header:
    kind: str  # "request" | "response"
    run_id: str
    round: int
    phase: str
    node_id: int
    nonce: str
    payload_sha256: str = ""
    schema_version: int = SCHEMA_VERSION
    signature: str = ""
    seq: int = 0  # which exchange of the run this is (1, 2, 3, ...); lets a client tell old requests from new


def new_nonce() -> str:
    return secrets.token_hex(16)


def write(path: Path, header: Header, payload: bytes) -> None:
    header = Header(**{**asdict(header), "payload_sha256": hashlib.sha256(payload).hexdigest()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(asdict(header), sort_keys=True).encode() + b"\n" + payload)
    os.replace(temporary, path)


def peek(path: Path) -> Header:
    """Just the header, without reading or checking the payload (cheap, for listing large files)."""
    try:
        with path.open("rb") as handle:
            return Header(**json.loads(handle.readline()))
    except (OSError, ValueError, TypeError) as error:
        raise EnvelopeError(f"{path.name}: unreadable envelope ({error})") from error


def read(path: Path) -> tuple[Header, bytes]:
    try:
        raw = path.read_bytes()
        line, _, payload = raw.partition(b"\n")
        fields = json.loads(line)
        header = Header(**fields)
    except (OSError, ValueError, TypeError) as error:
        raise EnvelopeError(f"{path.name}: unreadable envelope ({error})") from error
    if header.schema_version != SCHEMA_VERSION:
        raise EnvelopeError(f"{path.name}: schema version {header.schema_version}, expected {SCHEMA_VERSION}")
    if hashlib.sha256(payload).hexdigest() != header.payload_sha256:
        raise EnvelopeError(f"{path.name}: payload hash mismatch (truncated or corrupted)")
    return header, payload
