"""The COMPASS site as a neutral client function (`fedlab_core.ClientFn`): prepare its data, train, evaluate.

`make_client(node_config, run_config)` is a client factory: both `fedlab-async client --client-factory` and the live
Flower runner call it with the site's own settings. The optimizer state between rounds is the node's state payload.

A node config with `code_root` describes a site that holds its own code and data (a file-based run): requests are
then checked against the code identity, the shared scaler and manifest they carry are installed, and the real
backends get their environment. Without `code_root` the node was staged by fedlab and its services set the environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedlab_core import ClientFn, Payload

from .apps_common import client_backend
from .contract import SITES, OptimizerState
from .site import check_identity, install_shared, site_environment
from .strategy import WEIGHT_KEY, named, ordered

OPTIMIZER = "optimizer"
SOURCES = None  # file:// and http(s)://, made on first use; set `neutral_client.SOURCES` to register more (S3, DRS)


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value  # payload metadata is JSON


def _reply(arrays: list[np.ndarray] | None, metrics: dict, weight: int) -> Payload:
    return Payload(named(arrays) if arrays is not None else {}, {"metrics": {**{k: _plain(v) for k, v in metrics.items()}, WEIGHT_KEY: weight}})


def make_client(node_config: dict, run_config: dict) -> ClientFn:
    managed = "code_root" in node_config
    if managed:
        compass = node_config.get("compass_source")
        code_root, site_root, compass = Path(node_config["code_root"]), Path(node_config["workdir"]) / "site_root", Path(compass) if compass else None

    def prepare(request: Payload) -> Payload:
        from fedlab_async.assets import AssetCache, Sources, load_assignment  # file-run sites only: live nodes never prepare

        from .prepare import prepare_site

        site = str(node_config["site"])
        cache = AssetCache(node_config["cache"], SOURCES or Sources())
        source = cache.fetch(load_assignment(request.meta["assignment"], cache))["source"]
        client_dir = site_root / "clients" / site
        report = prepare_site(code_root, site, source, client_dir, seed=int(request.meta["seed"]))
        blobs = {"manifest": report.manifest_bytes, "scaler_stats": report.scaler_stats_bytes, "gene_order": (client_dir / "gene_order.txt").read_bytes()}
        return Payload(blobs=blobs)

    def train_or_evaluate(phase: str, request: Payload, state: Payload) -> tuple[Payload, Payload]:
        backend = client_backend(run_config, node_config)
        arrays, config = ordered(request.arrays), dict(request.meta["config"])
        if phase == "train":
            saved = OptimizerState(ordered(state.arrays), state.meta[OPTIMIZER]) if OPTIMIZER in state.meta else None
            result = backend.fit(arrays, config, saved)
            if result.optimizer is not None:
                state = Payload(named(result.optimizer.arrays), {OPTIMIZER: result.optimizer.meta})
            return _reply(result.arrays, result.metrics, backend.weight), state
        return _reply(None, backend.evaluate(arrays, config), backend.weight), state

    def client(phase: str, request: Payload, state: Payload, node_id: int) -> tuple[Payload, Payload]:
        if phase not in ("prepare", "train", "evaluate"):
            raise ValueError(f"unknown phase {phase!r}")
        if not managed:
            return train_or_evaluate(phase, request, state)
        check_identity(request.meta.get("code_identity"), code_root, compass)
        if phase == "prepare":
            return prepare(request), state
        if "scaler_npz" in request.blobs:
            site = str(node_config["site"])
            install_shared(request.blobs, code_root, site_root, site, SITES[site][0], int(run_config["seed"]))
        if str(run_config["backend"]) == "stub":
            return train_or_evaluate(phase, request, state)
        with site_environment(code_root, compass, site_root):
            return train_or_evaluate(phase, request, state)

    return client
