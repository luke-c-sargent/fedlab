"""COMPASS training as a neutral `fedlab_core` Strategy: weighted FedAvg, early stopping, best-model tracking.

This is the original Flower-based orchestrator without Flower: the same checks and records, over `Payload`s. Each exchange is
one request per site with the global model in `arrays` (named "0", "1", ...). A reply holds the site's arrays
(train only) and `meta["metrics"]`, which includes the site's FedAvg weight as `num-examples`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fedlab_core import Payload, Plan, Strategy

from .aggregate import ClientReport, aggregate
from .apps_common import EnvFactory, server_backend_factory
from .contract import DATA_BINDING, SITES, RoundRecord, ServerBackend
from .logic import EarlyStopper, weighted_mean
from .site import CONFIG, load_fc, write_if_different

if TYPE_CHECKING:
    from fedlab_async.assets import Manifest

WEIGHT_KEY = "num-examples"


def named(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {str(i): a for i, a in enumerate(arrays)}


def ordered(arrays: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    return [arrays[str(i)] for i in range(len(arrays))]


class CompassStrategy(Strategy):
    """`nodes` maps each node id to its site. Give a `backend` or a picklable `backend_factory` (the backend is then
    dropped on pickling and rebuilt on first use, so it must keep no state between rounds)."""

    def __init__(
        self, nodes: Mapping[int, str], rounds: int, patience: int, out_dir: Path, backend: ServerBackend | None = None,
        backend_factory: Callable[[], ServerBackend] | None = None, shared: Mapping[str, bytes] | None = None,
        code_identity: str | None = None,
    ):
        if backend is None and backend_factory is None:
            raise ValueError("give a backend or a backend_factory")
        if sorted(nodes.values()) != sorted(SITES):
            raise ValueError(f"nodes must cover exactly the sites {sorted(SITES)}; got {sorted(nodes.values())}")
        self.nodes, self.rounds, self.out_dir = dict(nodes), rounds, Path(out_dir)
        self._backend, self._backend_factory = backend, backend_factory
        self.shared, self.code_identity = dict(shared or {}), code_identity  # sent with every request
        self.stopper, self.history = EarlyStopper(patience), []
        self.round, self.phase = 1, "train"  # phase: "train" | "evaluate" | "done"
        self.arrays = self.backend.initial_arrays()
        self._by_site: dict[str, dict[str, dict]] = {}
        self._started = self._server_seconds = 0.0

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

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def position(self) -> tuple[int, str]:
        return self.round, self.phase

    def plan(self) -> Plan:
        began = time.perf_counter()
        if self.phase == "train":
            self._started, self._server_seconds = time.time(), 0.0
            config = {**self.backend.fit_config(self.round), "server-round": self.round}
        else:
            config = {"server-round": self.round}
        meta = {"config": config, **({"code_identity": self.code_identity} if self.code_identity else {})}
        request = Payload(named(self.arrays), meta, dict(self.shared))
        self._server_seconds += time.perf_counter() - began
        return Plan(self.round, self.phase, {node: request for node in self.nodes})

    def absorb(self, round: int, phase: str, replies: dict[int, Payload]) -> None:
        began = time.perf_counter()
        self._by_site[phase] = self._check(round, phase, replies)
        if phase == "train":
            updates = [(ordered(replies[n].arrays), int(replies[n].meta["metrics"][WEIGHT_KEY])) for n in sorted(replies)]
            replacement = self.backend.aggregate(updates)
            self.arrays = replacement if replacement is not None else weighted_mean(updates)
            self.phase = "evaluate"
            self._server_seconds += time.perf_counter() - began
            return
        stop = self._record(round, time.perf_counter() - began)
        self.round += 1
        self.phase = "train" if self.round <= self.rounds and not stop else "done"
        if self.done:
            if self.stopper.best is None:
                raise RuntimeError("no round completed")
            self.backend.finalize(self.stopper.best, self.history, self.out_dir)

    # -- the checks and records of the original orchestrator, over payloads ---------------------------------

    def _check(self, round: int, phase: str, replies: dict[int, Payload]) -> dict[str, dict]:
        """Wrong or duplicate sites and a different dataset manifest are refused before anything is aggregated."""
        what = f"round {round} {phase}"
        by_site: dict[str, dict] = {}
        for node in sorted(replies):
            metrics = replies[node].meta["metrics"]
            site = str(metrics.get("site", ""))
            if site not in SITES or site in by_site:
                raise RuntimeError(f"{what}: invalid or duplicate client site {site!r}")
            by_site[site] = metrics
        if set(by_site) != set(SITES):
            raise RuntimeError(f"{what}: expected replies from {sorted(SITES)}, got {sorted(by_site)}")
        expected = self.backend.expected_data_binding()
        if expected is not None:
            for site, metrics in sorted(by_site.items()):
                reported = metrics.get(DATA_BINDING)
                if reported != expected:
                    raise RuntimeError(
                        f"{what}: {site} " + (
                            "did not report which federated dataset manifest it used" if reported is None else
                            f"used a different federated dataset manifest ({str(reported)[:12]}..., expected {expected[:12]}...); "
                            "its scaler and data would not match the other sites'"
                        )
                    )
        return by_site

    def _record(self, round: int, server_seconds: float) -> bool:
        """Record the round; True when training should stop."""
        train = self._by_site["train"]
        validation = self.backend.combine_validation(self._by_site["evaluate"])
        weights = sum(float(m[WEIGHT_KEY]) for m in train.values())
        train_loss = sum(float(m["train_loss"]) * float(m[WEIGHT_KEY]) for m in train.values()) / weights
        timing = {"elapsed_seconds": time.time() - self._started, "server_seconds": self._server_seconds + server_seconds}
        client = [float(m["local_fit_wall_seconds"]) for m in train.values() if "local_fit_wall_seconds" in m]
        if client:
            timing["client_seconds"] = max(client)  # the slowest client is the round's compute critical path
        self.history.append(
            RoundRecord(round, train_loss, validation.loss, {"wall_seconds": timing["elapsed_seconds"], **timing,
                                                             "train": train, "validation": validation.summary})
        )
        return self.stopper.update(round, validation.loss, self.arrays, validation.summary)


class CompassRun(Strategy):
    """A file-based COMPASS run: a prepare exchange, then `CompassStrategy`.

    Round 0 ("prepare"): each site gets its data assignment, prepares its own data, and reports its scaler statistics.
    The server aggregates them into the global scaler and federated manifest, writes them where its backend expects
    them, and only then builds the training: the real server backend needs the scaler to exist before it can be created.
    """

    def __init__(
        self, *, sites: Mapping[int, str], assignments: Mapping[int, "Manifest"], code_root: Path, expected_identity: dict,
        run_config: Mapping, rounds: int, patience: int, out_dir: Path, server_site_root: Path, compass_source: Path | None = None,
    ):
        if set(sites) != set(assignments):
            raise ValueError("every node needs exactly one assignment")
        for node, site in sites.items():
            if site not in SITES or "source" not in {f.name for f in assignments[node].files}:
                raise ValueError(f"node {node}: site must be one of {sorted(SITES)} and its assignment needs a file named 'source'")
        self.sites, self.assignments, self.code_root = dict(sites), dict(assignments), Path(code_root)
        self.identity, self.run_config = json.dumps(expected_identity, sort_keys=True), dict(run_config)
        self.rounds, self.patience, self.out_dir = rounds, patience, Path(out_dir)
        self.server_site_root, self.compass_source = Path(server_site_root), compass_source
        self.training: CompassStrategy | None = None

    @property
    def done(self) -> bool:
        return self.training is not None and self.training.done

    def position(self) -> tuple[int, str]:
        return (0, "prepare") if self.training is None else self.training.position()

    def plan(self) -> Plan:
        if self.training is not None:
            return self.training.plan()
        meta = {"seed": int(self.run_config["seed"]), "code_identity": self.identity}
        return Plan(0, "prepare", {node: Payload(meta={**meta, "assignment": a.to_json()}) for node, a in self.assignments.items()})

    def absorb(self, round: int, phase: str, replies: dict[int, Payload]) -> None:
        if phase != "prepare":
            self.training.absorb(round, phase, replies)
            return
        reports, gene_orders, sources = {}, {}, {}
        for node in sorted(replies):
            blobs, client_id = replies[node].blobs, SITES[self.sites[node]][0]
            reports[client_id] = ClientReport(blobs["manifest"], blobs["scaler_stats"])
            gene_orders[client_id] = blobs["gene_order"]
            sources[client_id] = next(f.sha256 for f in self.assignments[node].files if f.name == "source")
        fc = load_fc(self.code_root)
        result = aggregate(fc, reports, self.code_root / CONFIG, "global_scaler.npz", expected_source_sha256=sources)
        for client_id, genes in gene_orders.items():
            if hashlib.sha256(genes).hexdigest() != result.manifest["gene_order_sha256"]:
                raise ValueError(f"{client_id}: the gene order sent is not the one its manifest attests")
        root = self.server_site_root
        write_if_different(root / "global_scaler.npz", result.scaler_bytes)
        write_if_different(root / "federated_manifest.json", result.manifest_bytes)
        write_if_different(root / "gene_order.txt", next(iter(gene_orders.values())))
        fc.validate_federated_manifest_metadata(
            root / "federated_manifest.json", global_scaler=root / "global_scaler.npz", config_path=self.code_root / CONFIG
        )
        factory = server_backend_factory(self.run_config)
        if str(self.run_config["backend"]) != "stub":
            factory = EnvFactory(factory, self.code_root, self.compass_source, self.server_site_root)
        self.training = CompassStrategy(
            self.sites, self.rounds, self.patience, self.out_dir, backend_factory=factory,
            shared={"scaler_npz": result.scaler_bytes, "federated_manifest": result.manifest_bytes}, code_identity=self.identity,
        )
