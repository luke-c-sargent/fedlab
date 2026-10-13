"""The directory of one run, as the server sees it.

    <run>/exchange/    what is moved between parties: requests/, responses/, rejected/
    <run>/private/     server only: state (pickled), lock, audit log, copies of outstanding requests
    <run>/results/     whatever the program writes when it finishes

Keep `private/` out of anything shared: the state file is pickled, so whoever can write it controls the server.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout import Exchange
from .server import Server


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def exchange(self) -> Exchange:
        return Exchange(self.root / "exchange")

    @property
    def state(self) -> Path:
        return self.root / "private" / "state.pkl"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def exists(self) -> bool:
        return self.state.exists()

    def server(self) -> Server:
        if not self.exists():
            raise FileNotFoundError(f"{self.root} is not a run (no private/state.pkl); create one with `init`")
        return Server(self.state, self.exchange)


def load_object(spec: str) -> Any:
    """`package.module:name` -> the object."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected 'package.module:name', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def add_paths(paths: list[str]) -> None:
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def load_config(path: Path) -> dict:
    """A run or site description: YAML (.yaml/.yml) or JSON."""
    text = Path(path).read_text()
    if Path(path).suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data
