"""Base class for modules whose training framework is Flower (Message API, SuperLink + SuperNodes)."""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .. import metrics
from ..flower.federation import FLEET_PORT, Federation
from ..flower.orchestrated import EXEC, NODE, OrchestratedFederation
from .base import Deployment, FLModule, Site

PYTHON_VERSION = "3.12"  # Flower 1.42 needs >= 3.11; 3.12 matches the orchestrated image
ENV_DIR = "fl"  # virtualenv at ~/fl on every node
APP_DIR = "app"  # the Flower app on the server at ~/app
RESULTS_DIR = "results"  # service-owned output root on the server at ~/results


PACKAGE_LINE = re.compile(r"^\s*[+~-] \S+==\S+\s*$")  # uv's "+ package==1.2.3" listing


def quiet_packages(emit: Callable[[str], None]) -> Callable[[str], None]:
    """Forward installer output, but replace uv's per-package listing with one summary line."""
    count = 0

    def on_line(line: str) -> None:
        nonlocal count
        if PACKAGE_LINE.match(line):
            count += 1
        elif line.strip():
            if count:
                emit(f"{count} package change(s) listed")
                count = 0
            emit(line)

    return on_line


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
        return {"FEDLAB_OUTPUT_DIR": f"{site.host.home}/{RESULTS_DIR}"} if site is d.server else {}

    def stage_site(self, d: Deployment, site: Site) -> None:
        """Upload module-specific code and data to one node."""

    # ---- lifecycle -------------------------------------------------------------------
    def stage(self, d: Deployment) -> None:
        metrics.start(d)  # from the start, so installs and uploads are measured too
        with ThreadPoolExecutor() as pool:  # nodes install and upload independently
            list(pool.map(lambda site: self._stage_site(d, site), d.sites))
        if not self.orchestrated:  # orchestrated runs upload the app to the orchestrator when they start
            d.log("stage server: Flower app")
            d.server.host.put(self.app_path, f"{d.server.host.home}/{APP_DIR}")

    @property
    def orchestrated(self) -> bool:
        return self.cfg.federation == "orchestrated"

    def _stage_site(self, d: Deployment, site: Site) -> None:
        if self.orchestrated:
            OrchestratedFederation(d, d.orchestrator).prepare_node(site)  # the image replaces the virtualenv
        else:
            self.install_environment(d, site, self.requirements)
        d.log(f"stage {site.node.name}: module files")
        self.stage_site(d, site)

    def install_environment(self, d: Deployment, site: Site, requirements: Path | str) -> None:
        """Create the node's virtualenv (~/fl) and install `requirements` (a local file, or the text itself)."""
        host, name = site.host, site.node.name
        d.log(f"stage {name}: python environment")
        if isinstance(requirements, Path):
            host.put(requirements, f"{host.home}/requirements.txt")
        else:
            host.put_text(requirements, f"{host.home}/requirements.txt")
        host.run(
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh\n"
            f"test -x {host.home}/{ENV_DIR}/bin/python || uv venv --python {PYTHON_VERSION} {host.home}/{ENV_DIR}\n"
            # best-match across indexes: numpy etc. also exist on the PyTorch index, only torch's "+cuXXX" builds must come from it
            f"UV_INDEX_STRATEGY=unsafe-best-match uv pip install --python {host.home}/{ENV_DIR}/bin/python -r {host.home}/requirements.txt\n"
            # as in the image: Triton would need a C compiler on first use, and nothing here uses torch.compile
            f"uv pip uninstall --python {host.home}/{ENV_DIR}/bin/python triton || true",
            on_line=quiet_packages(lambda line: d.log(f"  {name}: {line}")),
        )

    def start(self, d: Deployment) -> None:
        if self.orchestrated:
            OrchestratedFederation(d, d.orchestrator).start(self.node_config, lambda site: self.service_env(d, site))
            return
        Federation(d, ENV_DIR).start(self.node_config, lambda site: self.service_env(d, site))
        address = getattr(d.server.host, "address", None)  # LocalHost has none
        if address:
            metrics.probe_rtt(d, address, FLEET_PORT)

    def run(self, d: Deployment) -> str:
        host = d.server.host
        metrics.mark(d, "run_start")
        try:
            if self.orchestrated:
                return OrchestratedFederation(d, d.orchestrator).submit(self.app_path, self.run_config(), on_line=lambda line: d.log(f"  {line}") if line.strip() else None)
            fed = Federation(d, ENV_DIR, self.service_env(d, d.server))
            return fed.submit(f"{host.home}/{APP_DIR}", self.run_config(), on_line=lambda line: d.log(f"  {line}") if line.strip() else None)
        finally:
            metrics.mark(d, "run_end")

    def collect(self, d: Deployment, dest: Path) -> list[Path]:
        host = d.server.host
        host.mkdir(f"{host.home}/{RESULTS_DIR}")
        host.get(f"{host.home}/{RESULTS_DIR}", dest)
        if self.orchestrated:
            fed = OrchestratedFederation(d, d.orchestrator)
            for site in d.clients:
                (dest / f"{site.node.name}.supernode.log").write_text(fed.logs(site, NODE))
            (dest / f"{d.server.node.name}.superexec.log").write_text(fed.logs(d.server, EXEC))
            metrics.collect(d, dest)
            summary_file = dest / "metrics" / "summary.json"
            summary = json.loads(summary_file.read_text())
            summary["orchestrator_traffic"] = d.orchestrator.traffic(dest / "metrics")  # what the orchestrator moved, per request
            summary_file.write_text(json.dumps(summary, indent=2))
            return sorted(p for p in dest.rglob("*") if p.is_file())
        # client logs help when a run misbehaves
        for site in d.clients:
            logs = site.host.service_log("supernode", 400)
            (dest / f"{site.node.name}.supernode.log").write_text(logs)
        (dest / f"{d.server.node.name}.superlink.log").write_text(host.service_log("superlink", 400))
        metrics.collect(d, dest)
        return sorted(p for p in dest.rglob("*") if p.is_file())

    def stop(self, d: Deployment) -> None:
        if self.orchestrated:
            OrchestratedFederation(d, d.orchestrator).stop()
        else:
            Federation(d, ENV_DIR).stop()
