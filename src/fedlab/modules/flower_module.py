"""Base class for modules whose training framework is Flower (Message API, SuperLink + SuperNodes)."""

from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..flower.federation import Federation
from .base import Deployment, FLModule, Site

PYTHON_VERSION = "3.11"  # Flower 1.33 needs >= 3.11
ENV_DIR = "fl"  # virtualenv at ~/fl on every node
APP_DIR = "app"  # the Flower app on the server at ~/app
RESULTS_DIR = "results"  # service-owned output root on the server at ~/results


class FlowerModule(FLModule):
    """Implements stage/start/run/collect/stop for a Flower app; subclasses supply the app and data."""

    # ---- what a subclass provides --------------------------------------------------
    @property
    @abstractmethod
    def app_path(self) -> Path:
        """Local directory of the Flower app (contains pyproject.toml)."""

    @property
    @abstractmethod
    def requirements(self) -> Path:
        """Local requirements file for the nodes' virtualenv."""

    @abstractmethod
    def run_config(self) -> dict:
        """Values for `flwr run --run-config` (keys must exist in the app's pyproject)."""

    def node_config(self, site: Site) -> dict:
        """Per-SuperNode config, readable in the ClientApp as `context.node_config`."""
        return {"role": site.node.role, **({"site": site.node.site} if site.node.site else {})}

    def service_env(self, d: Deployment, site: Site) -> dict[str, str]:
        """Extra environment for the node's Flower services (and so for the app code)."""
        env = {"FEDLAB_OUTPUT_DIR": f"{site.host.home}/{RESULTS_DIR}"} if site is d.server else {}
        return env

    def stage_site(self, d: Deployment, site: Site) -> None:
        """Upload module-specific code and data to one node."""

    # ---- lifecycle -------------------------------------------------------------------
    def stage(self, d: Deployment) -> None:
        with ThreadPoolExecutor() as pool:  # nodes install and upload independently
            list(pool.map(lambda site: self._stage_site(d, site), d.sites))
        d.log("stage server: Flower app")
        d.server.host.put(self.app_path, f"{d.server.host.home}/{APP_DIR}")

    def _stage_site(self, d: Deployment, site: Site) -> None:
        host, name = site.host, site.node.name
        d.log(f"stage {name}: python environment")
        host.put(self.requirements, f"{host.home}/requirements.txt")
        host.run(
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh\n"
            f"test -x {host.home}/{ENV_DIR}/bin/python || uv venv --python {PYTHON_VERSION} {host.home}/{ENV_DIR}\n"
            f"uv pip install --python {host.home}/{ENV_DIR}/bin/python -r {host.home}/requirements.txt",
            on_line=lambda line: d.log(f"  {name}: {line}") if line.strip() else None,
        )
        d.log(f"stage {name}: module files")
        self.stage_site(d, site)

    def start(self, d: Deployment) -> None:
        Federation(d, ENV_DIR).start(self.node_config, lambda site: self.service_env(d, site))

    def run(self, d: Deployment) -> str:
        host = d.server.host
        fed = Federation(d, ENV_DIR)
        return fed.submit(f"{host.home}/{APP_DIR}", self.run_config(), on_line=lambda line: d.log(f"  {line}") if line.strip() else None)

    def collect(self, d: Deployment, dest: Path) -> list[Path]:
        host = d.server.host
        host.mkdir(f"{host.home}/{RESULTS_DIR}")
        host.get(f"{host.home}/{RESULTS_DIR}", dest)
        # client logs help when a run misbehaves
        for site in d.clients:
            logs = site.host.service_log("supernode", 400)
            (dest / f"{site.node.name}.supernode.log").write_text(logs)
        (dest / f"{d.server.node.name}.superlink.log").write_text(host.service_log("superlink", 400))
        return sorted(p for p in dest.rglob("*") if p.is_file())

    def stop(self, d: Deployment) -> None:
        Federation(d, ENV_DIR).stop()
