"""Real training backend: COMPASS (pinned source) + torch, ported from the original Flower 1.8 scripts.

Ported from `federated_test/scripts/{03_flower_client,04_flower_server}.py` for the default path only
(historical_v1 protocol, scratch initialization, random-within-context negatives, full GTEx).
Everything that is not Flower-specific is reused from `federated_common` and `centralized_test`
unchanged. Imports are lazy so the rest of the app works without torch.

Differences from the original, all forced by the Message API running each message in a fresh process:
  * the Adam state travels in `context.state` instead of living in the client process;
  * the torch RNG (dropout) is re-seeded per round instead of running on from process start;
  * the per-process audit receipts (hashes of every payload) are dropped; the prepared data is
    verified once on each node at deploy time (see `verify.py`).
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .contract import SITES, BestModel, ClientBackend, FitResult, OptimizerState, RoundRecord, ServerBackend, Validation
from .logic import pack_optimizer_state, unpack_optimizer_state

CODE_ENV = "FEDLAB_COMPASS_CODE"  # dir holding centralized_test/ and federated_test/
SOURCE_ENV = "COMPASS_SOURCE_DIR"  # the pinned COMPASS git checkout
SITE_ENV = "FEDLAB_COMPASS_SITE_ROOT"  # prepared data: clients/<site>/, global_scaler.npz, federated_manifest.json
OVERRIDES_NONE = ("negative_sampling", "hard_negative_fraction", "gtex_donor_fraction", "gtex_dose_seed", "anchors_per_round")


class Env:
    """Paths, configuration and imports shared by the client and server backends."""

    def __init__(self, run_config):
        code, self.source, self.site_root = Path(os.environ[CODE_ENV]), Path(os.environ[SOURCE_ENV]), Path(os.environ[SITE_ENV])
        for path in (code / "federated_test" / "scripts", code / "centralized_test" / "scripts", self.source):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import federated_common as fc
        import torch

        self.fc, self.torch = fc, torch
        self.central = fc.load_centralized_training()
        self.config_path = code / "centralized_test" / "config" / "paper_pretraining.json"
        self.federated_path = code / "federated_test" / "config" / "federated_pretraining.json"
        self.config = self.central.load_config(self.config_path)
        self.federated = fc.load_federated_config(self.federated_path)
        self.seed = int(run_config["seed"])
        self.local_epochs = int(run_config["local-epochs"])
        self.controls = fc.protocol_controls(self.federated, self.seed)
        self.streams = self.controls["streams"]
        self.qualification = fc.resolve_federated_qualification(self.federated, {k: None for k in OVERRIDES_NONE})
        self._require_default_path()
        self.source_commit = self.central.verify_compass_source(self.source)
        self.scaler_path = self.site_root / "global_scaler.npz"
        self.manifest = fc.validate_federated_manifest_metadata(
            self.site_root / "federated_manifest.json", global_scaler=self.scaler_path,
            config_path=self.config_path, expected_split_seed=self.streams["split"],
        )
        self.aggregation_policy = self.federated["flower"]["aggregation_contract"]

    def _require_default_path(self) -> None:
        q = self.qualification
        unported = {
            "protocol": self.controls["training_protocol"] != "historical_v1",
            "initialization": q["initialization"] != "scratch",
            "negative_sampling": q["negative_sampling"] != "random_within_context",
            "gtex_donor_fraction": q["gtex_donor_fraction"] != 1.0,
            "anchors_per_round": q["anchors_per_round"] is not None,
        }
        bad = [name for name, flag in unported.items() if flag]
        if bad:
            raise NotImplementedError(f"only the default training path is ported; unsupported: {bad}")

    def build_model(self):
        from compass.model.model import Compass

        self.central.set_reproducible_seed(self.streams["initialization"], self.torch)
        return self.central.build_model(self.config, 34, self.seed, Compass)


def _loader_epoch_seed(seed: int, server_round: int, local_epoch: int) -> int:
    return seed * 10_000_019 + server_round * 101 + local_epoch  # historical_v1 formula


def _seed_worker(worker_id: int) -> None:
    import torch

    np.random.seed(torch.initial_seed() % (2**32))


class RealClient(ClientBackend):
    def __init__(self, site: str, run_config, node_config):
        from sklearn.preprocessing import MinMaxScaler

        env = self.env = Env(run_config)
        fc, central, torch = env.fc, env.central, env.torch
        self.site = site
        self.client_id, self.expected_contexts = SITES[site]
        self.client_dir = env.site_root / "clients" / site
        contract = env.federated["flower"]["clients"][self.client_id]
        self.weight = (
            int(env.manifest["clients"][self.client_id]["train_samples"])
            if env.aggregation_policy == "sample_count_weighted"
            else int(contract["aggregation_weight"])
        )
        self.training = env.config["training"]
        self.micro_batch = int(run_config["micro-batch-size"])
        self.num_workers = int(run_config["num-workers"])
        effective_batch = int(self.training["batch_size"])
        if self.micro_batch > effective_batch or effective_batch % self.micro_batch:
            raise ValueError("micro-batch size must divide the configured effective batch size")
        self.accumulation_steps = effective_batch // self.micro_batch
        torch.set_num_threads(int(run_config["cpu-threads"]))
        self.device = torch.device(str(run_config["device"]))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this node")

        self.records, self.train_indices, self.validation_indices = fc.client_split(
            self.client_dir, self.training["validation_fraction"], env.streams["split"]
        )
        self.contexts = np.load(self.client_dir / "context.int16.npy", mmap_mode="r")
        self.group_ids = [record["group_id"] for record in self.records]
        self.scaler = fc.build_sklearn_scaler(env.scaler_path, MinMaxScaler)
        self.anchor_budget, _ = fc.resolve_local_anchor_budget(
            selected_training_samples=int(env.manifest["total_training_samples"]),
            client_count=len(env.federated["flower"]["clients"]),
            effective_batch_size=effective_batch,
            requested_anchors_per_round=env.qualification["anchors_per_round"],
            local_epochs=env.local_epochs,
        )
        self.local_sampling = env.federated["flower"].get("local_sampling", "context_balanced")
        self._model = None

    # ---- lazily built pieces (a train message never needs validation data, and vice versa) ------
    @property
    def model(self):
        if self._model is None:
            self._model = self.env.build_model().to(self.device)
        return self._model

    def _dataset(self, indices, deterministic_seed):
        return self.env.central.TripletDataset(
            expression_path=self.client_dir / "expression.float32.npy",
            contexts=self.contexts,
            group_ids=self.group_ids,
            global_indices=indices,
            scaler=self.scaler,
            mask_probability=self.training["mask_probability"],
            jitter_std=self.training["jitter_standard_deviation"],
            deterministic_seed=deterministic_seed,
            sample_keys=None,
            training_protocol=self.env.controls["training_protocol"],
            frozen_negative_pools=None,
        )

    def _make_optimizer(self):
        torch, model = self.env.torch, self.model
        return torch.optim.Adam(
            list(model.inputencoder.parameters()) + list(model.latentprojector.parameters()),
            lr=self.training["learning_rate"],
            weight_decay=self.training["weight_decay"],
        )

    # ---- ClientBackend -----------------------------------------------------------------------
    def fit(self, arrays, config, optimizer_state: OptimizerState | None) -> FitResult:
        from torch.utils.data import DataLoader

        env = self.env
        fc, central, torch = env.fc, env.central, env.torch
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        server_round = int(config["server-round"])
        local_epochs = int(config["local_epochs"])
        model = self.model
        fc.set_foundation_arrays(model, arrays, torch)
        received = [np.asarray(a).copy() for a in arrays]
        optimizer = self._make_optimizer()
        if optimizer_state is not None:
            saved = unpack_optimizer_state(optimizer_state)
            saved["state"] = {i: {k: torch.as_tensor(v) for k, v in s.items()} for i, s in saved["state"].items()}
            optimizer.load_state_dict(saved)

        # Each round runs in a new process: re-seed so dropout differs between rounds but stays reproducible.
        torch.manual_seed(env.streams["optimization"] * 1_000_003 + server_round)
        train_dataset = self._dataset(self.train_indices, None)
        losses = []
        for local_epoch in range(local_epochs):
            sampler = fc.make_local_sampler(
                dataset=train_dataset, contexts=self.contexts, sample_count=self.anchor_budget,
                server_round=server_round, local_epoch=local_epoch, seed=env.streams["sampler"],
                torch=torch, policy=self.local_sampling,
            )
            train_dataset.set_epoch((server_round - 1) * local_epochs + local_epoch)
            generator = torch.Generator()
            generator.manual_seed(_loader_epoch_seed(env.streams["loader"], server_round, local_epoch))
            loader = DataLoader(
                train_dataset, batch_size=self.micro_batch, sampler=sampler, drop_last=True,
                pin_memory=self.device.type == "cuda", num_workers=self.num_workers,
                worker_init_fn=_seed_worker, generator=generator,
            )
            losses.append(
                central.train_epoch(
                    loader, model, optimizer, self.device, self.training["triplet_margin"], torch,
                    self.accumulation_steps, inputencoder_trainable=True,
                )
            )
        updated = fc.get_foundation_arrays(model)
        l2 = float(np.sqrt(sum(float(np.sum((u.astype(np.float64) - r.astype(np.float64)) ** 2)) for u, r in zip(updated, received))))
        metrics = {
            "site": self.site,
            "client_id": self.client_id,
            "train_loss": float(np.mean(losses)),
            "anchors_processed": self.anchor_budget * local_epochs,
            "update_l2_norm": l2,
            "local_fit_wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
        }
        return FitResult(updated, pack_optimizer_state(optimizer.state_dict()), metrics)

    def evaluate(self, arrays, config) -> dict:
        from torch.utils.data import DataLoader

        env = self.env
        fc, central, torch = env.fc, env.central, env.torch
        fc.set_foundation_arrays(self.model, arrays, torch)
        dataset = self._dataset(self.validation_indices, env.streams["triplet"])
        loader = DataLoader(
            dataset, batch_size=self.micro_batch, shuffle=False, drop_last=False,
            pin_memory=self.device.type == "cuda", num_workers=self.num_workers, worker_init_fn=_seed_worker,
        )
        summary = central.validation_epoch(
            loader, self.model, self.device, self.training["triplet_margin"], torch, sorted(self.expected_contexts)
        )
        loss = float(summary[fc.VALIDATION_CONTEXT_MACRO_FIELD])
        return {
            "site": self.site,
            "client_id": self.client_id,
            "checkpoint_selection_metric": fc.CHECKPOINT_SELECTION_METRIC,
            "validation_loss": loss,
            fc.VALIDATION_CONTEXT_MACRO_FIELD: loss,
            fc.VALIDATION_ROW_MACRO_FIELD: float(summary[fc.VALIDATION_ROW_MACRO_FIELD]),
            fc.VALIDATION_CONTEXT_STATISTICS_METRIC: json.dumps(
                summary["validation_per_context"], sort_keys=True, separators=(",", ":")
            ),
            "actual_validation_samples": len(self.validation_indices),
        }


class RealServer(ServerBackend):
    def __init__(self, run_config):
        from sklearn.preprocessing import MinMaxScaler

        env = self.env = Env(run_config)
        self.local_epochs = env.local_epochs
        self.genes = (env.site_root / "gene_order.txt").read_text(encoding="utf-8").splitlines()
        if len(self.genes) != env.config["input"]["expected_genes"]:
            raise ValueError("gene order violates the COMPASS contract")
        self.scaler = env.fc.build_sklearn_scaler(env.scaler_path, MinMaxScaler)
        self.model = env.build_model().cpu()

    def initial_arrays(self):
        return self.env.fc.get_foundation_arrays(self.model)

    def fit_config(self, server_round: int) -> dict:
        return {"server_round": server_round, "local_epochs": self.local_epochs}

    def aggregate(self, updates):
        return self.env.fc.aggregate_arrays(updates)  # same float/int handling as the original

    def combine_validation(self, by_site) -> Validation:
        fc = self.env.fc
        summary, _ = fc.aggregate_client_validation_metrics(
            {SITES[site][0]: metrics for site, metrics in by_site.items()},
            {client_id: contexts for client_id, contexts in SITES.values()},
        )
        return Validation(float(summary[fc.VALIDATION_CONTEXT_MACRO_FIELD]), summary)

    def finalize(self, best: BestModel, history: list[RoundRecord], out_dir: Path) -> None:
        from compass import PreTrainer
        from compass.model.saver import SaveBestModel
        from compass.model.scaler import Datascaler
        from sklearn.preprocessing import MinMaxScaler

        env = self.env
        fc, central, torch = env.fc, env.central, env.torch
        out_dir.mkdir(parents=True, exist_ok=True)
        fc.set_foundation_arrays(self.model, best.arrays, torch)
        model = self.model.cpu().eval()
        history_rows = [
            {
                "epoch": r.round,
                "train_loss": r.train_loss,
                "validation_loss": r.validation_loss,
                fc.VALIDATION_CONTEXT_MACRO_FIELD: r.extra["validation"][fc.VALIDATION_CONTEXT_MACRO_FIELD],
                fc.VALIDATION_ROW_MACRO_FIELD: r.extra["validation"][fc.VALIDATION_ROW_MACRO_FIELD],
            }
            for r in history
        ]
        central.write_history(out_dir / "history.tsv", history_rows)
        (out_dir / "rounds.json").write_text(json.dumps([r.__dict__ for r in history], indent=2, default=str))
        portable = {
            "epoch": best.round,
            "model_args": copy.deepcopy(model.model_args),
            "model_state_dict": central.cpu_state_dict(model.state_dict()),
            "optimizer_state_dict": {},
            "scaler": self.scaler,
            "feature_names": list(self.genes),
            "best_validation_loss": best.loss,
            "checkpoint_selection_metric": fc.CHECKPOINT_SELECTION_METRIC,
            "best_validation_context_macro_loss": best.validation[fc.VALIDATION_CONTEXT_MACRO_FIELD],
            "best_validation_row_macro_loss": best.validation[fc.VALIDATION_ROW_MACRO_FIELD],
            "best_validation_per_context": best.validation["validation_per_context"],
            "scope": "federated_tcga_gtex",
            "compass_git_commit": env.source_commit,
        }
        central.atomic_torch_save(torch, portable, out_dir / "best_model.pth")
        wrapper = central.make_official_pretrainer(
            PreTrainer=PreTrainer, SaveBestModel=SaveBestModel, Datascaler=Datascaler, MinMaxScaler=MinMaxScaler,
            model=model, scaler=self.scaler, genes=self.genes, config=env.config, context_count=34, seed=env.seed,
            best_epoch=best.round, best_validation_loss=best.loss, best_validation_summary=best.validation,
            best_optimizer_state={}, history=history_rows, run_dir=out_dir,
        )
        central.atomic_torch_save(torch, wrapper, out_dir / env.federated["output"]["official_pretrainer"])
