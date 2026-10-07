from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Node, Settings
    from ..ssh import Keypair

# Normalised states: pending, running, stopping, stopped, absent
@dataclass
class NodeInfo:
    name: str
    provider: str
    location: str
    state: str
    public_ip: str | None = None
    running_since: datetime | None = None  # tz-aware; only set while running


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


class Provider(ABC):
    def __init__(self, cfg: Settings, key: Keypair):
        self.cfg = cfg
        self.key = key

    @abstractmethod
    def up(self, node: Node) -> None:
        """Create the node (and shared firewall/key) if missing; start it if stopped. Idempotent."""

    @abstractmethod
    def start(self, node: Node) -> None: ...

    @abstractmethod
    def stop(self, node: Node) -> None: ...

    @abstractmethod
    def describe(self, node: Node) -> NodeInfo: ...

    @abstractmethod
    def destroy(self, nodes: list[Node], dry_run: bool = False) -> list[str]:
        """Delete every resource tagged for this run in the nodes' locations.

        Returns human-readable descriptions of what was (or with dry_run, would be) deleted.
        """

    @abstractmethod
    def check(self, nodes: list[Node]) -> list[CheckResult]:
        """Read-only preflight: credentials, permissions and requirements for these nodes.

        Must not create, modify or delete anything. Never raises; failures are returned as results.
        """
