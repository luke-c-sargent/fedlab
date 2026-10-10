"""The federated round loop: Flower strategy hooks plus early stopping and best-model tracking.

This mirrors `Strategy.start()` in Flower but can stop early, and it keeps the learning logic
(what to send, how to combine, when to stop) separate from the training backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from .contract import SITES, RoundRecord, ServerBackend
from .logic import EarlyStopper

WEIGHT_KEY = "num-examples"


def make_strategy() -> FedAvg:
    """Two clients, every round, weighted by each client's `num-examples` (its context count)."""
    return FedAvg(
        fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=2, min_evaluate_nodes=2,
        min_available_nodes=2, weighted_by_key=WEIGHT_KEY,
    )


def pack_reply(arrays: list[np.ndarray] | None, metrics: dict, weight: int) -> RecordDict:
    """Client reply: numbers go in the (single) MetricRecord, strings in a ConfigRecord."""
    numbers = {k: (int(v) if isinstance(v, bool) else v) for k, v in metrics.items() if not isinstance(v, str)}
    strings = {k: v for k, v in metrics.items() if isinstance(v, str)}
    content = RecordDict({"metrics": MetricRecord({**numbers, WEIGHT_KEY: weight})})
    if strings:
        content["details"] = ConfigRecord(strings)
    if arrays is not None:
        content["arrays"] = ArrayRecord(arrays)
    return content


def unpack_metrics(reply: Message) -> dict:
    """The inverse of the metrics half of `pack_reply`."""
    merged = dict(reply.content["metrics"])
    if "details" in reply.content:
        merged.update(dict(reply.content["details"]))
    return merged


def _site_metrics(replies: list[Message], what: str) -> dict[str, dict]:
    by_site: dict[str, dict] = {}
    for reply in replies:
        if reply.has_error():
            raise RuntimeError(f"{what}: a client failed: {reply.error.reason}")
        metrics = unpack_metrics(reply)
        site = str(metrics.get("site", ""))
        if site not in SITES or site in by_site:
            raise RuntimeError(f"{what}: invalid or duplicate client site {site!r}")
        by_site[site] = metrics
    if set(by_site) != set(SITES):
        raise RuntimeError(f"{what}: expected replies from {sorted(SITES)}, got {sorted(by_site)}")
    return by_site


def run_federation(
    grid: Grid,
    backend: ServerBackend,
    rounds: int,
    patience: int,
    out_dir: Path,
    strategy: FedAvg | None = None,
    timeout: float = 24 * 3600,
) -> tuple[list[RoundRecord], EarlyStopper]:
    strategy = strategy or make_strategy()
    stopper = EarlyStopper(patience)
    arrays = ArrayRecord(backend.initial_arrays())
    history: list[RoundRecord] = []

    for server_round in range(1, rounds + 1):
        started = time.perf_counter()
        log(INFO, "round %d/%d: training on both clients", server_round, rounds)
        train_replies = list(
            grid.send_and_receive(
                strategy.configure_train(server_round, arrays, ConfigRecord(backend.fit_config(server_round)), grid),
                timeout=timeout,
            )
        )
        train_by_site = _site_metrics(train_replies, f"round {server_round} train")
        aggregated, _ = strategy.aggregate_train(server_round, train_replies)
        if aggregated is None:
            raise RuntimeError(f"round {server_round}: training aggregation produced nothing")
        replacement = backend.aggregate(
            [
                (reply.content["arrays"].to_numpy_ndarrays(), int(reply.content["metrics"][WEIGHT_KEY]))
                for reply in train_replies
            ]
        )
        arrays = ArrayRecord(replacement) if replacement is not None else aggregated

        eval_replies = list(
            grid.send_and_receive(strategy.configure_evaluate(server_round, arrays, ConfigRecord({}), grid), timeout=timeout)
        )
        eval_by_site = _site_metrics(eval_replies, f"round {server_round} evaluate")
        validation = backend.combine_validation(eval_by_site)

        train_loss = sum(float(m["train_loss"]) * float(m[WEIGHT_KEY]) for m in train_by_site.values()) / sum(
            float(m[WEIGHT_KEY]) for m in train_by_site.values()
        )
        record = RoundRecord(
            server_round, train_loss, validation.loss,
            {"wall_seconds": time.perf_counter() - started, "train": train_by_site, "validation": validation.summary},
        )
        history.append(record)
        stop = stopper.update(server_round, validation.loss, arrays.to_numpy_ndarrays(), validation.summary)
        log(INFO, "round %d/%d done in %.0fs: train=%.6f validation=%.6f best=%.6f (round %d)", server_round, rounds,
            record.extra["wall_seconds"], train_loss, validation.loss, stopper.best.loss, stopper.best.round)
        if stop:
            log(INFO, "early stopping after %d rounds without improvement", stopper.patience)
            break

    if stopper.best is None:
        raise RuntimeError("no round completed")
    backend.finalize(stopper.best, history, out_dir)
    return history, stopper
