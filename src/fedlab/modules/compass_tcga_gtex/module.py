"""Federated COMPASS foundation-model training on TCGA (33 cancer contexts) and GTEx (normal tissue)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from ...config import Node
from ..base import Deployment, Site
from ..flower_module import APP_DIR, ENV_DIR, FlowerModule

HERE = Path(__file__).parent
COMPASS_URL = "https://github.com/mims-harvard/COMPASS.git"
COMPASS_COMMIT = "9ecacfb72d63feec6434d336b702eb22a68c13a2"  # pinned by the original experiment
TRAIN_DIR = "compass_hpc_foundation_model_train"  # inside the user's checkout
CODE_DIR, SOURCE_DIR, SITE_DIR = "compass_code", "compass_src", "site"  # on every node, under ~
SITES = ("tcga", "gtex")
DEFAULTS = {
    "backend": "compass", "seed": 42, "rounds": 100, "local_epochs": 1, "patience": 10,
    "micro_batch_size": 64, "num_workers": 0, "cpu_threads": 8, "device": "cuda",
}


class CompassTcgaGtexModule(FlowerModule):
    name = "compass_tcga_gtex"
    description = "Federated COMPASS pretraining: TCGA + GTEx clients, context-weighted FedAvg, early stopping."

    def __init__(self, cfg, options):
        super().__init__(cfg, options)
        self.opt = {**DEFAULTS, **options}

    # ---- local paths ------------------------------------------------------------------------
    @property
    def stub(self) -> bool:
        return self.opt["backend"] == "stub"

    @property
    def train_root(self) -> Path:
        return Path(str(self.opt["compass_repo"])).expanduser() / TRAIN_DIR

    @property
    def prepared(self) -> Path:
        return Path(str(self.opt.get("prepared_root", ".fedlab/compass/prepared"))).expanduser()

    @property
    def app_path(self) -> Path:
        return HERE / "app"

    @property
    def requirements(self) -> Path:
        return HERE / ("requirements_stub.txt" if self.stub else "requirements.txt")

    # ---- preflight and preparation --------------------------------------------------------------
    def validate(self, nodes: list[Node]) -> list[str]:
        problems = []
        servers = [n for n in nodes if n.role == "server"]
        sites = sorted(n.site or "" for n in nodes if n.role == "client")
        if len(servers) != 1:
            problems.append("exactly one node must have role: server")
        if sites != sorted(SITES):
            problems.append(f"clients must have exactly the sites {list(SITES)}; found {sites}")
        if self.opt["backend"] not in ("compass", "stub"):
            problems.append("module_options.backend must be 'compass' or 'stub'")
        if self.stub:
            return problems
        if "compass_repo" not in self.opt:
            return problems + ["module_options.compass_repo must point at your federated-learning-model checkout"]
        for sub in ("centralized_test", "federated_test"):
            if not (self.train_root / sub).is_dir():
                problems.append(f"{self.train_root / sub} not found")
        missing = [
            str(p.relative_to(self.prepared))
            for p in self._data_files()
            if not p.exists()
        ]
        if missing:
            problems.append(f"prepared data incomplete under {self.prepared} (missing {missing}); run `fedlab prepare`")
        return problems

    def _data_files(self) -> list[Path]:
        return [
            self.prepared / "federated_manifest.json", self.prepared / "global_scaler.npz",
            self.prepared / "gene_order.txt",
            *(self.prepared / "clients" / s / "manifest.json" for s in SITES),
        ]

    def prepare(self, log: Callable[[str], None]) -> None:
        """Build the per-site caches and the shared scaler locally (raw data never leaves this machine)."""
        for key in ("compass_repo", "tcga_tsv", "gtex_tsv"):
            if key not in self.opt:
                raise ValueError(f"module_options.{key} is required for `fedlab prepare`")
        python = str(self.opt.get("prep_python", sys.executable))
        scripts = self.train_root / "federated_test" / "scripts"
        config = self.train_root / "centralized_test" / "config" / "paper_pretraining.json"
        clients = self.prepared / "clients"
        steps = [
            [python, str(scripts / "01_prepare_clients.py"), "--tcga", str(Path(self.opt["tcga_tsv"]).expanduser()),
             "--gtex", str(Path(self.opt["gtex_tsv"]).expanduser()), "--config", str(config),
             "--output-root", str(clients), "--seed", str(self.opt["seed"])],
            [python, str(scripts / "02_aggregate_scaler.py"), "--clients-root", str(clients),
             "--output", str(self.prepared / "global_scaler.npz"),
             "--manifest", str(self.prepared / "federated_manifest.json"), "--config", str(config)],
        ]
        for step in steps:
            log(f"prepare: {' '.join(step[1:3])} ...")
            subprocess.run(step, check=True)
        shutil.copyfile(clients / "tcga" / "gene_order.txt", self.prepared / "gene_order.txt")
        log(f"prepare: done; prepared data is in {self.prepared}")

    # ---- Flower configuration ---------------------------------------------------------------------
    def run_config(self) -> dict:
        o = self.opt
        return {
            "backend": o["backend"], "num-server-rounds": int(o["rounds"]), "local-epochs": int(o["local_epochs"]),
            "patience": int(o["patience"]), "seed": int(o["seed"]), "micro-batch-size": int(o["micro_batch_size"]),
            "num-workers": int(o["num_workers"]), "cpu-threads": int(o["cpu_threads"]),
            "device": "cpu" if self.stub else str(o["device"]),
        }

    def service_env(self, d: Deployment, site: Site) -> dict[str, str]:
        env = super().service_env(d, site)
        env["FEDLAB_STATE_DIR"] = f"{site.host.home}/state"  # per-node optimizer state (too large for context.state)
        if not self.stub:
            home = site.host.home
            env |= {
                "FEDLAB_COMPASS_CODE": f"{home}/{CODE_DIR}", "COMPASS_SOURCE_DIR": f"{home}/{SOURCE_DIR}",
                "FEDLAB_COMPASS_SITE_ROOT": f"{home}/{SITE_DIR}", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            }
        return env

    # ---- staging ------------------------------------------------------------------------------------
    def stage_site(self, d: Deployment, site: Site) -> None:
        if self.stub:
            return
        host, name = site.host, site.node.name
        home = host.home
        d.log(f"stage {name}: code")
        for sub in ("centralized_test", "federated_test"):
            host.put(self.train_root / sub, f"{home}/{CODE_DIR}/{sub}")
        src = f"{home}/{SOURCE_DIR}"
        host.run(  # the pinned commit only (the full history is ~800 MB); fall back to a full fetch
            f"test -d {src}/.git || {{ git init -q {src} && git -C {src} remote add origin {COMPASS_URL}; }}\n"
            f"git -C {src} fetch -q --depth 1 origin {COMPASS_COMMIT} || git -C {src} fetch -q origin\n"
            f"git -C {src} checkout -q --detach {COMPASS_COMMIT}"
        )
        d.log(f"stage {name}: prepared data")
        root = f"{home}/{SITE_DIR}"
        for file in ("federated_manifest.json", "global_scaler.npz", "gene_order.txt"):
            host.put(self.prepared / file, f"{root}/{file}")
        if site.node.role == "client":
            host.put(self.prepared / "clients" / site.node.site, f"{root}/clients/{site.node.site}")
        host.put(self.app_path, f"{home}/{APP_DIR}")  # for the verification below; the server also runs it from here
        which = site.node.site if site.node.role == "client" else "server"
        env = self.service_env(d, site)
        d.log(f"stage {name}: verifying prepared data")
        out = host.run(
            f"cd {home}/{APP_DIR} && PYTHONPATH={home}/{APP_DIR} {home}/{ENV_DIR}/bin/python -m fedlab_compass.verify "
            f"--site {which} --seed {self.opt['seed']}",
            env=env,
        )
        d.log(f"  {name}: {out.strip().splitlines()[-1]}")
