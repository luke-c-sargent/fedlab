"""Flower entry points for a neutral strategy, configured from run and node config (no code per strategy).

    run-config:   program="package.module:factory"  run-dir="<path>"  and any other keys the strategy takes
    node-config:  node-id=N  and one of  client="package.module:client_fn"  or  client-factory="package.module:make"
                  (a factory is called as make(node_config, run_config) and returns the client function)

`factory(options: dict, run_dir: Path)` returns a `Strategy`, the same factory `fedlab-async init` takes; `options`
is the rest of the run config. Flower run-config values are flat (bool, int, float, string), so a list is a string.
"""

from __future__ import annotations

from pathlib import Path

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from ..loader import load_object
from .runner import make_client_app, serve

server = ServerApp()
def _client_fn(context: Context):
    if "client-factory" in context.node_config:
        return load_object(str(context.node_config["client-factory"]))(dict(context.node_config), dict(context.run_config))
    return load_object(str(context.node_config["client"]))


client = make_client_app(_client_fn)


@server.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    options = {k: v for k, v in config.items() if k not in ("program", "run-dir")}
    serve(load_object(str(config["program"]))(options, Path(str(config.get("run-dir", ".")))), grid)
