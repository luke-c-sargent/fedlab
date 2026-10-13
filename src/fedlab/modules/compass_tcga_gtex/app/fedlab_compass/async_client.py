"""ClientApp for a file-based COMPASS run: three actions, one per phase of the run.

    query.prepare   fetch the assigned raw file, prepare this site's data, report scaler statistics
    train/evaluate  as in `client_app`, after installing the shared scaler and manifest and checking the code

Per-node settings come from `node_config`: `site`, `workdir`, `cache` (asset cache), `code_root`, and
optionally `compass_source`. Everything the site prepares lives under `<workdir>/site_root`.
"""

from __future__ import annotations

from pathlib import Path

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedlab_async.assets import AssetCache, Sources, load_assignment

from . import client_app
from .contract import SITES
from .prepare import prepare_site
from .site import check_identity, install_shared, site_environment

app = ClientApp()
SOURCES = Sources()  # file:// and http(s)://; register more (S3, DRS) before the app runs


def _paths(node_config) -> tuple[Path, Path, Path | None]:
    compass = node_config.get("compass_source")
    return Path(node_config["code_root"]), Path(node_config["workdir"]) / "site_root", Path(compass) if compass else None


@app.query("prepare")
def prepare(msg: Message, context: Context) -> Message:
    config, node = dict(msg.content["config"]), context.node_config
    site = str(node["site"])
    code_root, site_root, compass = _paths(node)
    check_identity(config.get("code_identity"), code_root, compass)
    cache = AssetCache(node["cache"], SOURCES)
    source = cache.fetch(load_assignment(config["assignment"], cache))["source"]
    client_dir = site_root / "clients" / site
    report = prepare_site(code_root, site, source, client_dir, seed=int(config["seed"]))
    reply = RecordDict(
        {
            "report": ConfigRecord(
                {
                    "manifest": report.manifest_bytes,
                    "scaler_stats": report.scaler_stats_bytes,
                    "gene_order": (client_dir / "gene_order.txt").read_bytes(),
                }
            ),
            "metrics": MetricRecord({"num-examples": 1}),
        }
    )
    return Message(reply, reply_to=msg)


def _ready(msg: Message, context: Context) -> None:
    config, node = dict(msg.content["config"]), context.node_config
    code_root, site_root, compass = _paths(node)
    check_identity(config.get("code_identity"), code_root, compass)
    if "scaler_npz" in config:
        site = str(node["site"])
        install_shared(config, code_root, site_root, site, SITES[site][0], int(context.run_config["seed"]))


def _run(action, msg: Message, context: Context) -> Message:
    _ready(msg, context)
    if str(context.run_config["backend"]) == "stub":
        return action(msg, context)
    code_root, site_root, compass = _paths(context.node_config)
    with site_environment(code_root, compass, site_root):
        return action(msg, context)


@app.train()
def train(msg: Message, context: Context) -> Message:
    return _run(client_app.train, msg, context)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    return _run(client_app.evaluate, msg, context)
