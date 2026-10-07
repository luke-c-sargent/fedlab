"""The module interface: one federated-learning experiment that fedlab can deploy, run, and collect."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

from ..config import Node, Settings
from ..remote import Host


@dataclass
class Site:
    """A fleet node paired with a way to run commands on it."""

    node: Node
    host: Host


@dataclass
class Deployment:
    """Everything a module needs to act on a running fleet."""

    cfg: Settings
    server: Site
    clients: list[Site]
    state_dir: Path  # local scratch for this deployment (certificates, ...)
    log: Callable[[str], None] = field(default=print)

    @property
    def sites(self) -> list[Site]:
        return [self.server, *self.clients]

    def client_for(self, data_site: str) -> Site:
        matches = [c for c in self.clients if c.node.site == data_site]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one client with site {data_site!r}, found {len(matches)}")
        return matches[0]


class FLModule(ABC):
    """A federated-learning experiment.

    fedlab drives the lifecycle: `prepare` (local, optional) -> `stage` -> `start` -> `run` -> `collect`,
    then `stop`. Implementations own everything experiment-specific (software, data, framework).
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, cfg: Settings, options: dict):
        self.cfg = cfg
        self.options = options

    def validate(self, nodes: list[Node]) -> list[str]:
        """Local preflight (no cloud calls): return a list of problems, empty if all is well."""
        return []

    def prepare(self, log: Callable[[str], None]) -> None:
        """Optional local step that builds inputs (for example, data preparation)."""

    @abstractmethod
    def stage(self, d: Deployment) -> None:
        """Install software and upload code and data to the nodes."""

    @abstractmethod
    def start(self, d: Deployment) -> None:
        """Bring up the federation (long-running services)."""

    @abstractmethod
    def run(self, d: Deployment) -> str:
        """Run the experiment to completion; return a run identifier. Raise if it fails."""

    @abstractmethod
    def collect(self, d: Deployment, dest: Path) -> list[Path]:
        """Copy results to the local directory `dest`; return the files copied."""

    @abstractmethod
    def stop(self, d: Deployment) -> None:
        """Stop the services started by `start` (best effort)."""
