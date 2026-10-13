"""A file-based COMPASS run: a prepare exchange, then the usual train/evaluate rounds.

Round 0 ("prepare"): each site gets its data assignment, prepares its own data, and reports its scaler
statistics. The server aggregates them into the global scaler and federated manifest (`aggregate`), writes them
where its backend expects them, and only then builds the training run: the real server backend needs the scaler
to exist before it can be created. Every later request carries the scaler, the manifest and the code identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict

from fedlab_async.adapters.flower_server import FlowerProgram, _decode, build_request, check_reply
from fedlab_async.assets import Asset, Manifest, measure
from fedlab_async.server import Plan, Program, Server, Slot, Timing

from . import orchestrator
from .aggregate import ClientReport, aggregate
from .apps_common import server_backend_factory
from .contract import SITES
from .identity import code_identity
from .site import CONFIG, load_fc, write_if_different


class CompassHooks:
    """The orchestrator's `Federation`, plus the shared files and code identity in every request's config."""

    def __init__(self, federation: orchestrator.Federation, shared: dict):
        self.federation, self.shared = federation, shared

    def config(self, server_round: int, phase: str) -> ConfigRecord:
        return ConfigRecord({**dict(self.federation.config(server_round, phase)), **self.shared})

    def round_started(self, server_round: int) -> None:
        self.federation.round_started(server_round)

    def check_replies(self, server_round: int, phase: str, replies: list[Message]) -> None:
        self.federation.check_replies(server_round, phase, replies)

    def train_aggregated(self, server_round, replies, aggregated):
        return self.federation.train_aggregated(server_round, replies, aggregated)

    def evaluated(self, server_round, replies, arrays, timing) -> bool:
        return self.federation.evaluated(server_round, replies, arrays, timing)

    def finished(self) -> None:
        self.federation.finished()


class EnvFactory:
    """Build the real server backend with the environment it reads. The factory is rebuilt after every pickle load."""

    def __init__(self, inner, code_root: Path, compass_source: Path | None, site_root: Path):
        self.inner, self.code_root, self.compass_source, self.site_root = inner, code_root, compass_source, site_root

    def __call__(self):
        os.environ["FEDLAB_COMPASS_CODE"] = str(self.code_root)
        os.environ["FEDLAB_COMPASS_SITE_ROOT"] = str(self.site_root)
        if self.compass_source is not None:
            os.environ["COMPASS_SOURCE_DIR"] = str(self.compass_source)
        return self.inner()


class CompassProgram(Program):
    def __init__(
        self, *, sites: Mapping[int, str], assignments: Mapping[int, Manifest], code_root: Path, expected_identity: dict,
        run_config: Mapping, rounds: int, patience: int, out_dir: Path, server_site_root: Path,
        compass_source: Path | None = None,
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
        self.flower: FlowerProgram | None = None

    @property
    def done(self) -> bool:
        return self.flower is not None and self.flower.done

    def position(self) -> tuple[int, str]:
        return (0, "prepare") if self.flower is None else self.flower.position()

    def plan(self, run_id: str) -> Plan:
        if self.flower is not None:
            return self.flower.plan(run_id)
        requests = {}
        for node, assignment in self.assignments.items():
            config = ConfigRecord(
                {"assignment": assignment.to_json(), "seed": int(self.run_config["seed"]), "code_identity": self.identity}
            )
            requests[node] = build_request(run_id, 0, "prepare", node, "query.prepare", RecordDict({"config": config}))
        return Plan(0, "prepare", requests)

    def validate(self, slot: Slot, payload: bytes) -> None:
        check_reply(slot, payload)

    def absorb(self, run_id: str, round: int, phase: str, responses: dict[int, bytes], timing: Timing) -> None:
        if phase != "prepare":
            self.flower.absorb(run_id, round, phase, responses, timing)
            return
        reports, gene_orders, sources = {}, {}, {}
        for node in sorted(responses):
            reply = _decode(responses[node])
            if reply.has_error():
                raise RuntimeError(f"node {node} ({self.sites[node]}) could not prepare its data: {reply.error.reason}")
            client_id = SITES[self.sites[node]][0]
            content = dict(reply.content["report"])
            reports[client_id] = ClientReport(bytes(content["manifest"]), bytes(content["scaler_stats"]))
            gene_orders[client_id] = bytes(content["gene_order"])
            sources[client_id] = next(f.sha256 for f in self.assignments[node].files if f.name == "source")
        fc = load_fc(self.code_root)
        result = aggregate(fc, reports, self.code_root / CONFIG, "global_scaler.npz", expected_source_sha256=sources)
        for client_id, genes in gene_orders.items():
            if hashlib.sha256(genes).hexdigest() != result.manifest["gene_order_sha256"]:
                raise ValueError(f"{client_id}: the gene order sent is not the one its manifest attests")
        self._install_on_server(fc, result, next(iter(gene_orders.values())))
        self._start_training(result.scaler_bytes, result.manifest_bytes)

    def _install_on_server(self, fc, result, gene_order: bytes) -> None:
        root = self.server_site_root
        write_if_different(root / "global_scaler.npz", result.scaler_bytes)
        write_if_different(root / "federated_manifest.json", result.manifest_bytes)
        write_if_different(root / "gene_order.txt", gene_order)
        fc.validate_federated_manifest_metadata(
            root / "federated_manifest.json", global_scaler=root / "global_scaler.npz", config_path=self.code_root / CONFIG
        )

    def _start_training(self, scaler: bytes, manifest: bytes) -> None:
        factory = server_backend_factory(self.run_config)
        if str(self.run_config["backend"]) != "stub":
            factory = EnvFactory(factory, self.code_root, self.compass_source, self.server_site_root)
        federation = orchestrator.Federation(None, self.rounds, self.patience, self.out_dir, backend_factory=factory)
        hooks = CompassHooks(federation, {"scaler_npz": scaler, "federated_manifest": manifest, "code_identity": self.identity})
        self.flower = FlowerProgram(
            orchestrator.make_strategy(), ArrayRecord(federation.backend.initial_arrays()), sorted(self.sites), self.rounds,
            hooks=hooks,
        )


class CompassServer(Server):
    @classmethod
    def create(cls, state_path: Path, exchange, **program_args) -> CompassServer:  # type: ignore[override]
        return super().create(state_path, exchange, CompassProgram(**program_args))

    @property
    def federation(self) -> orchestrator.Federation:
        return self.state.program.flower.hooks.federation


def build(options: Mapping, run_dir: Path) -> CompassProgram:
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
    return CompassProgram(
        sites=sites, assignments=assignments, code_root=code_root, expected_identity=identity, run_config=run_config,
        rounds=int(options.get("rounds", 100)), patience=int(options.get("patience", 10)), out_dir=run_dir / "results",
        server_site_root=run_dir / "private" / "site_root", compass_source=compass_source,
    )


RUN_SETTINGS = ("local-epochs", "micro-batch-size", "num-workers", "cpu-threads", "device")
