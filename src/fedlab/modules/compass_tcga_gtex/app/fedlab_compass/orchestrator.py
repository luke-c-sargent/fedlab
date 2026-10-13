"""The federated round loop: Flower strategy hooks plus early stopping and best-model tracking.

This mirrors `Strategy.start()` in Flower but can stop early, and it keeps the learning logic
(what to send, how to combine, when to stop) separate from the training backend.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from .contract import DATA_BINDING, SITES, RoundRecord, ServerBackend
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


class Federation:
    """The per-round learning logic: site checks, backend aggregation, validation, early stopping, finalize.

    The live loop below and the file-based driver (`fedlab_async`) both call these methods in the same
    order, so the logic lives in one place. Instances are pickled between steps of a file-based run.

    Give `backend_factory` (a picklable zero-argument callable) when the backend holds anything that cannot
    be pickled, such as torch modules: the backend is then dropped on pickling and rebuilt on first use, so
    it must keep no state between rounds. Without a factory the backend itself is pickled.
    """

    def __init__(
        self, backend: ServerBackend | None, rounds: int, patience: int, out_dir: Path,
        backend_factory: Callable[[], ServerBackend] | None = None,
    ):
        if backend is None and backend_factory is None:
            raise ValueError("give a backend or a backend_factory")
        self._backend, self._backend_factory = backend, backend_factory
        self.rounds, self.out_dir = rounds, out_dir
        self.stopper = EarlyStopper(patience)
        self.history: list[RoundRecord] = []
        self._started = 0.0
        self._by_site: dict[str, dict[str, dict]] = {}

    @property
    def backend(self) -> ServerBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        if self._backend_factory is not None:
            state["_backend"] = None
        return state

    def config(self, server_round: int, phase: str) -> ConfigRecord:
        return ConfigRecord(self.backend.fit_config(server_round) if phase == "train" else {})

    def round_started(self, server_round: int) -> None:
        self._started = time.time()
        log(INFO, "round %d/%d: training on both clients", server_round, self.rounds)

    def check_replies(self, server_round: int, phase: str, replies: list[Message]) -> None:
        """Reject error replies and wrong or duplicate sites, before anything is aggregated."""
        by_site = _site_metrics(replies, f"round {server_round} {phase}")
        expected = self.backend.expected_data_binding()
        if expected is not None:
            for site, metrics in sorted(by_site.items()):  # replies arrive in random order; name the same site every time
                reported = metrics.get(DATA_BINDING)
                if reported != expected:
                    raise RuntimeError(
                        f"round {server_round} {phase}: {site} " + (
                            "did not report which federated dataset manifest it used" if reported is None else
                            f"used a different federated dataset manifest ({str(reported)[:12]}..., expected {expected[:12]}...); "
                            "its scaler and data would not match the other sites'"
                        )
                    )
        self._by_site[phase] = by_site

    def train_aggregated(self, server_round: int, replies: list[Message], aggregated: ArrayRecord | None) -> ArrayRecord:
        """Return the new global model (the backend may replace Flower's average)."""
        if aggregated is None:
            raise RuntimeError(f"round {server_round}: training aggregation produced nothing")
        replacement = self.backend.aggregate(
            [(reply.content["arrays"].to_numpy_ndarrays(), int(reply.content["metrics"][WEIGHT_KEY])) for reply in replies]
        )
        return ArrayRecord(replacement) if replacement is not None else aggregated

    def evaluated(self, server_round: int, replies: list[Message], arrays: ArrayRecord, timing: dict | None = None) -> bool:
        """Record the round. Returns True when training should stop.

        `timing` (from a file-based run) holds `elapsed_seconds` (first request to last response, including
        any waiting) and `server_seconds` (time spent in server steps). A live run measures elapsed itself.
        """
        train_by_site = self._by_site["train"]
        validation = self.backend.combine_validation(self._by_site["evaluate"])
        weights = sum(float(m[WEIGHT_KEY]) for m in train_by_site.values())
        train_loss = sum(float(m["train_loss"]) * float(m[WEIGHT_KEY]) for m in train_by_site.values()) / weights
        timing = dict(timing or {"elapsed_seconds": time.time() - self._started})
        client = [float(m["local_fit_wall_seconds"]) for m in train_by_site.values() if "local_fit_wall_seconds" in m]
        if client:
            timing["client_seconds"] = max(client)  # the slowest client is the round's compute critical path
        self.history.append(
            RoundRecord(
                server_round, train_loss, validation.loss,
                {"wall_seconds": timing["elapsed_seconds"], **timing, "train": train_by_site, "validation": validation.summary},
            )
        )
        stop = self.stopper.update(server_round, validation.loss, arrays.to_numpy_ndarrays(), validation.summary)
        log(INFO, "round %d/%d done in %.0fs: train=%.6f validation=%.6f best=%.6f (round %d)", server_round, self.rounds,
            timing["elapsed_seconds"], train_loss, validation.loss, self.stopper.best.loss, self.stopper.best.round)
        if stop:
            log(INFO, "early stopping after %d rounds without improvement", self.stopper.patience)
        return stop

    def finished(self) -> None:
        if self.stopper.best is None:
            raise RuntimeError("no round completed")
        self.backend.finalize(self.stopper.best, self.history, self.out_dir)


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
    federation = Federation(backend, rounds, patience, out_dir)
    arrays = ArrayRecord(backend.initial_arrays())

    for server_round in range(1, rounds + 1):
        federation.round_started(server_round)
        train_replies = list(
            grid.send_and_receive(
                strategy.configure_train(server_round, arrays, federation.config(server_round, "train"), grid), timeout=timeout
            )
        )
        federation.check_replies(server_round, "train", train_replies)
        aggregated, _ = strategy.aggregate_train(server_round, train_replies)
        arrays = federation.train_aggregated(server_round, train_replies, aggregated)
        eval_replies = list(
            grid.send_and_receive(
                strategy.configure_evaluate(server_round, arrays, federation.config(server_round, "evaluate"), grid),
                timeout=timeout,
            )
        )
        federation.check_replies(server_round, "evaluate", eval_replies)
        if federation.evaluated(server_round, eval_replies, arrays):
            break

    federation.finished()
    return federation.history, federation.stopper
