"""Which code a site is running. The server states what it expects; a client refuses to run on a mismatch.

Sites install the training code themselves (a documented prerequisite), so the request carries the identity of
the code it was built for. Only what executes or configures counts: every `*.py` and `*.json` under
`centralized_test/` and `federated_test/` (scripts and the config files), and the COMPASS commit. Notes, SLURM
scripts and bytecode do not.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

TREES = ("centralized_test", "federated_test")
SUFFIXES = {".py", ".json"}


class CodeMismatch(Exception):
    """The code installed here is not the code this request was made for."""


def tree_digest(root: Path) -> str:
    """One hash over the relative paths and contents of every counted file under `root`."""
    entries = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
            entries.append([path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def compass_commit(source: Path) -> str | None:
    done = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def code_identity(code_root: Path, compass_source: Path | None = None) -> dict:
    missing = [t for t in TREES if not (Path(code_root) / t).is_dir()]
    if missing:
        raise FileNotFoundError(f"{code_root} has no {', '.join(missing)}")
    identity: dict = {"trees": {t: tree_digest(Path(code_root) / t) for t in TREES}}
    if compass_source is not None:
        identity["compass_commit"] = compass_commit(Path(compass_source))
    return identity


def check_code_identity(expected: Mapping, actual: Mapping) -> None:
    """Raise `CodeMismatch` naming every component that differs. Components absent from `expected` are not checked."""
    problems = []
    for tree, digest in expected.get("trees", {}).items():
        if actual.get("trees", {}).get(tree) != digest:
            problems.append(f"{tree}/ differs (expected {digest[:12]}..., installed {str(actual.get('trees', {}).get(tree))[:12]}...)")
    if "compass_commit" in expected and actual.get("compass_commit") != expected["compass_commit"]:
        problems.append(f"COMPASS commit is {actual.get('compass_commit')}, expected {expected['compass_commit']}")
    if problems:
        raise CodeMismatch("this site's code is not the code the run was made for: " + "; ".join(problems))
