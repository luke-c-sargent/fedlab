"""Smoke ServerApp: plain FedAvg; writes the result to $FEDLAB_OUTPUT_DIR/run-<id>/result.json."""

import json
import os
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    strategy = FedAvg(
        fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=2, min_evaluate_nodes=2,
        min_available_nodes=2, weighted_by_key="num-examples",
    )
    result = strategy.start(
        grid=grid, initial_arrays=ArrayRecord([np.zeros(3)]), num_rounds=int(context.run_config["num-server-rounds"])
    )
    out = Path(os.environ.get("FEDLAB_OUTPUT_DIR", "results")) / f"run-{context.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(
            {
                "arrays": [a.tolist() for a in result.arrays.to_numpy_ndarrays()],
                "train": {str(k): dict(v) for k, v in result.train_metrics_clientapp.items()},
            },
            indent=2,
        )
    )
