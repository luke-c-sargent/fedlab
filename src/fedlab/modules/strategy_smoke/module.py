"""A neutral `fedlab_core` strategy on live Flower: checks the SuperLink/SuperNode path that `serve` uses."""

from __future__ import annotations

from pathlib import Path

import fedlab_core

from ..base import Deployment, Site
from ..flower_module import RESULTS_DIR, FlowerModule

HERE = Path(__file__).parent
PYLIB = "pylib"  # ~/pylib holds fedlab_core; PYTHONPATH points there
PROGRAM = "fedlab_strategy_smoke"


class StrategySmokeModule(FlowerModule):
    name = "strategy_smoke"
    description = "A neutral numpy strategy on live Flower (fedlab_core.flower): node ids, payloads, client state."

    def __init__(self, cfg, options):
        super().__init__(cfg, options)
        self._ids: dict[str, int] = {}
        self._home = ""

    @property
    def app_path(self) -> Path:
        return HERE / "app"

    @property
    def requirements(self) -> Path:
        return HERE / "requirements.txt"

    def _assign(self, d: Deployment) -> None:
        self._ids = {c.node.name: i for i, c in enumerate(d.clients, 1)}  # strategy node ids 1, 2, ...
        self._home = d.server.host.home

    def node_config(self, site: Site) -> dict:
        return {"node-id": self._ids[site.node.name], "client": f"{PROGRAM}:client_fn"}

    def run_config(self) -> dict:
        return {
            "program": f"{PROGRAM}:factory", "rounds": int(self.options.get("rounds", 3)),
            "nodes": ",".join(str(n) for n in sorted(self._ids.values())), "run-dir": f"{self._home}/{RESULTS_DIR}/strategy-smoke",
        }

    def service_env(self, d: Deployment, site: Site) -> dict[str, str]:
        home = site.host.home
        return {**super().service_env(d, site), "PYTHONPATH": f"{home}/{PYLIB}", "FEDLAB_STATE_DIR": f"{home}/state"}

    def stage_site(self, d: Deployment, site: Site) -> None:
        site.host.put(Path(fedlab_core.__file__).parent, f"{site.host.home}/{PYLIB}/fedlab_core")

    def start(self, d: Deployment) -> None:
        self._assign(d)
        super().start(d)

    def run(self, d: Deployment) -> str:
        self._assign(d)
        return super().run(d)
