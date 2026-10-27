"""Build and push the node image: Flower, torch (CPU or CUDA), the COMPASS toolchain and the pinned COMPASS source."""

from __future__ import annotations

import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Callable

HERE = Path(__file__).parent
REPO = "quay.io/sargentl/fedlab-compass"
FLAVORS = ("cpu", "cu126")
BUILDER = "fedlab-builder"  # its own cache, so removing it leaves your other Docker build cache alone


def build(flavor: str, repo: str = REPO, log: Callable[[str], None] = print) -> str:
    """Build the image for linux/amd64 and push it. Returns the pinned reference, `repo@sha256:...`."""
    if flavor not in FLAVORS:
        raise ValueError(f"flavor must be one of {FLAVORS}")
    flwr = version("flwr")
    tags = [f"{repo}:{flavor}", f"{repo}:{flwr}-{flavor}"]
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "meta.json"
        log(f"build {tags[0]} (linux/amd64, flwr {flwr}); this takes a while")
        subprocess.run(
            ["docker", "buildx", "build", "--builder", BUILDER, "--platform", "linux/amd64",
             "--build-arg", f"FLWR_VERSION={flwr}", "--build-arg", f"TORCH_FLAVOR={flavor}",
             *(a for t in tags for a in ("-t", t)), "--push", "--metadata-file", str(meta), str(HERE)],
            check=True,
        )
        digest = json.loads(meta.read_text())["containerimage.digest"]
    return f"{repo}@{digest}"


def create_builder() -> None:
    subprocess.run(["docker", "buildx", "create", "--name", BUILDER, "--driver", "docker-container"], check=False, capture_output=True)


def remove_builder() -> None:
    """Remove the builder and its build cache."""
    subprocess.run(["docker", "buildx", "rm", "--force", BUILDER], check=False, capture_output=True)
