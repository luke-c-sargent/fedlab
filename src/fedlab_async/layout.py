"""Where exchange files live. The exchange directory is what gets moved between parties.

Server state is deliberately NOT in here: it is pickled, so it must never sit where others can write.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Exchange:
    root: Path

    def request(self, round: int, phase: str, node_id: int) -> Path:
        return self.root / "requests" / f"r{round:04d}-{phase}-n{node_id}.req"

    def response(self, round: int, phase: str, node_id: int) -> Path:
        return self.root / "responses" / f"r{round:04d}-{phase}-n{node_id}.res"

    def quarantine(self, path: Path, reason: str) -> Path:
        """Move a rejected file aside with the reason next to it. Never deletes."""
        target = self.root / "rejected" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        while target.exists():
            n += 1
            target = target.with_name(f"{path.name}.{n}")
        shutil.move(str(path), target)
        target.with_name(target.name + ".reason").write_text(reason + "\n")
        return target
