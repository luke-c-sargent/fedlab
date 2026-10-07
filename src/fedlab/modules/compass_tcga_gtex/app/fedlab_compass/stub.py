"""A numpy stand-in for COMPASS training, for plumbing checks and tests (no torch, no data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import SITES, BestModel, ClientBackend, FitResult, OptimizerState, RoundRecord, ServerBackend, Validation
from .logic import pack_optimizer_state, unpack_optimizer_state

DIM = 4
TARGETS = {"tcga": 1.0, "gtex": -1.0}  # each site pulls the model toward its own target
WEIGHTS = {"tcga": 33, "gtex": 1}  # context counts, as in the real experiment


class StubClient(ClientBackend):
    def __init__(self, site: str):
        self.site = site
        self.weight = WEIGHTS[site]

    def _loss(self, arrays: list[np.ndarray]) -> float:
        return float(np.mean((arrays[0] - TARGETS[self.site]) ** 2))

    def fit(self, arrays, config, optimizer):
        step = int(unpack_optimizer_state(optimizer)["state"][0]["step"]) if optimizer else 0
        step += 1
        updated = [arrays[0] + 0.5 * (TARGETS[self.site] - arrays[0])]
        packed = pack_optimizer_state({"state": {0: {"step": np.float32(step)}}, "param_groups": [{"lr": 0.5, "params": [0]}]})
        metrics = {"site": self.site, "train_loss": self._loss(updated), "optimizer_step": step}
        return FitResult(updated, packed, metrics)

    def evaluate(self, arrays, config):
        contexts = SITES[self.site][1]
        loss = self._loss(arrays)
        stats = [{"context": c, "loss_sum": loss, "row_count": 1} for c in contexts]
        return {"site": self.site, "validation_loss": loss, "validation_statistics": json.dumps(stats)}


class StubServer(ServerBackend):
    def initial_arrays(self):
        return [np.zeros(DIM, dtype=np.float32)]

    def combine_validation(self, by_site):
        stats = [s for m in by_site.values() for s in json.loads(m["validation_statistics"])]
        loss = float(np.mean([s["loss_sum"] / s["row_count"] for s in stats]))  # equal weight per context
        return Validation(loss, {"contexts": len(stats), "per_site": {k: v["validation_loss"] for k, v in by_site.items()}})

    def finalize(self, best: BestModel, history: list[RoundRecord], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stub_result.json").write_text(
            json.dumps(
                {
                    "best_round": best.round,
                    "best_loss": best.loss,
                    "best_arrays": [a.tolist() for a in best.arrays],
                    "rounds_completed": len(history),
                    "validation_loss": [r.validation_loss for r in history],
                    "optimizer_steps": [r.extra["train"]["tcga"]["optimizer_step"] for r in history],
                },
                indent=2,
            )
        )
