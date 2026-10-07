"""Smoke module: a tiny CPU-only federation that proves the whole fedlab lifecycle works."""

from __future__ import annotations

from pathlib import Path

from ..flower_module import FlowerModule

HERE = Path(__file__).parent


class SmokeModule(FlowerModule):
    name = "smoke"
    description = "Tiny numpy FedAvg over every client; checks provisioning, TLS/auth, runs, and result collection."

    @property
    def app_path(self) -> Path:
        return HERE / "app"

    @property
    def requirements(self) -> Path:
        return HERE / "requirements.txt"

    def run_config(self) -> dict:
        return {"num-server-rounds": int(self.options.get("rounds", 3))}
