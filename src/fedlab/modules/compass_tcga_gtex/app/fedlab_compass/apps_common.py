"""Glue shared by the ClientApp and ServerApp: pick the backend named in the run config."""

from __future__ import annotations

import os
from pathlib import Path

from .contract import ClientBackend, ServerBackend


def server_backend(run_config) -> ServerBackend:
    if str(run_config["backend"]) == "stub":
        from .stub import StubServer

        return StubServer()
    from .real import RealServer

    return RealServer(run_config)


def client_backend(run_config, node_config) -> ClientBackend:
    site = str(node_config["site"])
    if str(run_config["backend"]) == "stub":
        from .stub import StubClient

        return StubClient(site)
    from .real import RealClient

    return RealClient(site, run_config, node_config)


def server_backend_factory(run_config):
    """A picklable recipe for `server_backend(run_config)`, for runs that rebuild the backend between steps."""
    from functools import partial

    return partial(server_backend, dict(run_config))


class EnvFactory:
    """Build the real server backend with the environment it reads. The factory is rebuilt after every pickle load."""

    def __init__(self, inner, code_root: Path, compass_source: Path | None, site_root: Path):
        self.inner, self.code_root, self.compass_source, self.site_root = inner, code_root, compass_source, site_root

    def __call__(self):
        os.environ["FEDLAB_COMPASS_CODE"] = str(self.code_root)
        os.environ["FEDLAB_COMPASS_SITE_ROOT"] = str(self.site_root)
        if self.compass_source is not None:
            os.environ["COMPASS_SOURCE_DIR"] = str(self.compass_source)
        return self.inner()
