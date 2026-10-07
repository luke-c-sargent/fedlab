"""Glue shared by the ClientApp and ServerApp: pick the backend named in the run config."""

from __future__ import annotations

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
