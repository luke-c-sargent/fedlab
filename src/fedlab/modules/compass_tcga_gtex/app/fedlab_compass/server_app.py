"""ServerApp: runs the federation loop and writes results to $FEDLAB_OUTPUT_DIR/run-<id>/."""

import os
from pathlib import Path

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from .apps_common import server_backend
from .orchestrator import run_federation

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    out_dir = Path(os.environ.get("FEDLAB_OUTPUT_DIR", "results")) / f"run-{context.run_id}"
    run_federation(
        grid, server_backend(cfg), rounds=int(cfg["num-server-rounds"]), patience=int(cfg["patience"]), out_dir=out_dir
    )
