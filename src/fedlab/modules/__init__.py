from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Deployment, FLModule, Site

if TYPE_CHECKING:
    from ..config import Settings

# name -> "module.path:ClassName", imported on demand so unused modules cost nothing
REGISTRY = {
    "smoke": "fedlab.modules.smoke.module:SmokeModule",
    "compass_tcga_gtex": "fedlab.modules.compass_tcga_gtex.module:CompassTcgaGtexModule",
}


def module_class(name: str) -> type[FLModule]:
    import importlib

    if name not in REGISTRY:
        raise ValueError(f"unknown module {name!r}; available: {sorted(REGISTRY)}")
    path, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(path), cls)


def get_module(cfg: Settings) -> FLModule:
    if not cfg.module:
        raise ValueError(f"no module configured; set `module:` to one of {sorted(REGISTRY)}")
    return module_class(cfg.module)(cfg, cfg.module_options)


__all__ = ["Deployment", "FLModule", "Site", "REGISTRY", "get_module", "module_class"]
