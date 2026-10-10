"""What the orchestrator needs from a training backend. Pure Python/numpy: no torch, no COMPASS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# site name (as set in the node config) -> (client id used in metrics, validation contexts the site owns)
SITES = {"tcga": ("TCGA", list(range(33))), "gtex": ("GTEx", [33])}


@dataclass
class OptimizerState:
    """A framework-neutral snapshot of a local optimizer."""

    arrays: list[np.ndarray]
    meta: str  # JSON


@dataclass
class FitResult:
    arrays: list[np.ndarray]
    optimizer: OptimizerState | None
    metrics: dict  # numbers and strings; the app routes numbers to a MetricRecord, strings to a ConfigRecord


@dataclass
class Validation:
    loss: float  # the checkpoint-selection metric (lower is better)
    summary: dict  # anything worth recording for the round


@dataclass
class BestModel:
    round: int
    loss: float
    arrays: list[np.ndarray]
    validation: dict


@dataclass
class RoundRecord:
    round: int
    train_loss: float | None = None
    validation_loss: float | None = None
    extra: dict = field(default_factory=dict)


class ClientBackend(ABC):
    """Runs on a SuperNode. A new instance is built for every message (each runs in a fresh process)."""

    site: str
    weight: int  # FedAvg weight for this client

    @abstractmethod
    def fit(self, arrays: list[np.ndarray], config: dict, optimizer: OptimizerState | None) -> FitResult: ...

    @abstractmethod
    def evaluate(self, arrays: list[np.ndarray], config: dict) -> dict:
        """Return a flat dict of metrics (numbers and strings)."""


class ServerBackend(ABC):
    """Runs inside the ServerApp."""

    @abstractmethod
    def initial_arrays(self) -> list[np.ndarray]: ...

    def fit_config(self, server_round: int) -> dict:
        return {"server_round": server_round}

    def aggregate(self, updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray] | None:
        """Optionally replace Flower's weighted average. `updates` are (arrays, weight) pairs."""
        return None

    @abstractmethod
    def combine_validation(self, by_site: dict[str, dict]) -> Validation:
        """Combine each site's evaluate metrics (keyed by site name) into one selection loss."""

    @abstractmethod
    def finalize(self, best: BestModel, history: list[RoundRecord], out_dir: Path) -> None:
        """Write the final model and any run records."""
