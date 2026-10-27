"""What moves between a strategy and its nodes: named arrays, JSON-able metadata, and raw byte blobs.

Encoded as `.npz` (never pickle), so the same bytes can sit in a file, a Flower message, or a test.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np

_META = "__meta__"


@dataclass
class Payload:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # anything json.dumps accepts
    blobs: dict[str, bytes] = field(default_factory=dict)  # files and other raw bytes


def encode(payload: Payload) -> bytes:
    stored = {}
    for name, array in payload.arrays.items():
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError(f"array {name!r} has dtype {array.dtype}; only numeric and string arrays are allowed")
        stored[f"a_{name}"] = array
    for name, blob in payload.blobs.items():
        stored[f"b_{name}"] = np.frombuffer(blob, dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **stored, **{_META: np.array(json.dumps(payload.meta))})
    return buffer.getvalue()


def decode(raw: bytes) -> Payload:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            return Payload(
                {n[2:]: saved[n] for n in saved.files if n.startswith("a_")},
                json.loads(str(saved[_META])),
                {n[2:]: saved[n].tobytes() for n in saved.files if n.startswith("b_")},
            )
    except Exception as error:
        raise ValueError(f"not a payload ({error})") from error
