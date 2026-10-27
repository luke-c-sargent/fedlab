"""ServerApp: runs the COMPASS strategy on the SuperNodes that are online; results go to $FEDLAB_OUTPUT_DIR/run-<id>/."""

import os
from pathlib import Path

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fedlab_core.flower.runner import serve

from .apps_common import server_backend_factory
from .contract import NODE_IDS
from .strategy import CompassStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    out_dir = Path(os.environ.get("FEDLAB_OUTPUT_DIR", "results")) / f"run-{context.run_id}"
    nodes = {node: site for site, node in NODE_IDS.items()}
    serve(CompassStrategy(nodes, int(cfg["num-server-rounds"]), int(cfg["patience"]), out_dir, backend_factory=server_backend_factory(cfg)), grid)
