from __future__ import annotations

import importlib
from typing import Any


def load_object(spec: str) -> Any:
    """`package.module:name` -> the object."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected 'package.module:name', got {spec!r}")
    return getattr(importlib.import_module(module), name)
