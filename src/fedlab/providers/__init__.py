from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CheckResult, NodeInfo, Provider

if TYPE_CHECKING:
    from ..config import Settings
    from ..ssh import Keypair


def get_provider(name: str, cfg: Settings, key: Keypair) -> Provider:
    if name == "aws":
        from .aws import AwsProvider

        return AwsProvider(cfg, key)
    if name == "gcp":
        from .gcp import GcpProvider

        return GcpProvider(cfg, key)
    raise ValueError(f"unknown provider {name!r}")


__all__ = ["CheckResult", "NodeInfo", "Provider", "get_provider"]
