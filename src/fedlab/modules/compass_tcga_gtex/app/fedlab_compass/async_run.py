"""Program factory for a file-based COMPASS run (`fedlab-async init`): the strategy is `strategy.CompassRun`."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fedlab_async.assets import Asset, Manifest, measure

from .identity import code_identity
from .strategy import CompassRun


def build(options: Mapping, run_dir: Path) -> CompassRun:
    """Program factory for `fedlab-async init`. Options (all paths may use ~):

        code_root:       directory holding centralized_test/ and federated_test/ (the installed training code)
        nodes:           {node id: {site: tcga|gtex, source: URI of its raw file, sha256: .., bytes: ..}}
                         sha256/bytes are measured by reading the file if you leave them out
        rounds, patience, seed, backend (compass|stub), and the other run settings: as in `module_options`
        compass_commit / compass_source: pin the COMPASS commit in the code identity (given, or read from a checkout)
    """
    code_root = Path(options["code_root"]).expanduser()
    sites, assignments = {}, {}
    for node, spec in options["nodes"].items():
        asset = (
            Asset("source", spec["source"], spec["sha256"], int(spec["bytes"]))
            if "sha256" in spec else measure("source", str(spec["source"]))
        )
        sites[int(node)], assignments[int(node)] = spec["site"], Manifest((asset,))
    compass_source = Path(options["compass_source"]).expanduser() if options.get("compass_source") else None
    identity = code_identity(code_root, compass_source)
    if options.get("compass_commit"):
        identity["compass_commit"] = options["compass_commit"]
    run_config = {"backend": options.get("backend", "compass"), "seed": int(options.get("seed", 42))}
    run_config |= {k: v for k, v in options.items() if k in RUN_SETTINGS}
    run_dir = Path(run_dir)
    return CompassRun(
        sites=sites, assignments=assignments, code_root=code_root, expected_identity=identity, run_config=run_config,
        rounds=int(options.get("rounds", 100)), patience=int(options.get("patience", 10)), out_dir=run_dir / "results",
        server_site_root=run_dir / "private" / "site_root", compass_source=compass_source,
    )


RUN_SETTINGS = ("local-epochs", "micro-batch-size", "num-workers", "cpu-threads", "device")
