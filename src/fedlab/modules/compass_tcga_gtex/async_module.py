"""The same COMPASS experiment, run as files instead of Flower connections (see `fedlab-async`).

fedlab plays the part of whoever moves the files, over the SSH connections it already has to every node. That is
the right thing for a fleet you own and for testing on real machines. It is not how file-based runs are meant to
work between institutions that do not trust each other; there, use `fedlab-async` directly and move the files
yourself.

    stage   environment and code on every node, the fedlab_core/fedlab_async/fedlab_compass packages, each site's raw file
    start   write the run and site configuration, `fedlab-async init` on the server
    run     repeat: emit on the server, copy requests to the sites, run the clients, copy responses back, collect
    collect results and the audit log

The sites prepare their own data (nothing is staged but the raw TSVs, which stand in for a data source the site
can reach), so `fedlab prepare` is not needed.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml

import fedlab_core
from fedlab_async.assets import measure

from ...config import Node
from ..base import Deployment, Site
from ..flower_module import ENV_DIR
from .module import CODE_DIR, NODE_IDS, PYLIB, SITES, SOURCE_DIR, CompassTcgaGtexModule

RUN_DIR = "run"  # the server's run directory
STUB_REQUIREMENTS = "flwr==1.40.0\nnumpy>=1.26,<2\npandas==1.5.3\n"  # the real prepare scripts still run
EXTRA_REQUIREMENTS = "typer\npyyaml\n"  # for the fedlab-async commands
ASYNC_PACKAGE = Path(__file__).resolve().parents[3] / "fedlab_async"
CORE_PACKAGE = Path(fedlab_core.__file__).parent


class CompassAsyncModule(CompassTcgaGtexModule):
    name = "compass_tcga_gtex_async"
    description = "Federated COMPASS pretraining as files: sites prepare their own data; fedlab moves the files over SSH."

    # ---- preflight ---------------------------------------------------------------------------
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
        for key in ("compass_repo", "tcga_tsv", "gtex_tsv"):
            if key not in self.opt:
                problems.append(f"module_options.{key} is required (the sites prepare their own data from the raw files)")
        if problems:
            return problems
        for sub in ("centralized_test", "federated_test"):
            if not (self.train_root / sub).is_dir():
                problems.append(f"{self.train_root / sub} not found")
        for key in ("tcga_tsv", "gtex_tsv"):
            if not Path(str(self.opt[key])).expanduser().is_file():
                problems.append(f"{key}: {self.opt[key]} is not a file")
        return problems

    def prepare(self, log: Callable[[str], None]) -> None:
        log("nothing to prepare here: in a file-based run each site prepares its own data during the first step")

    # ---- staging ----------------------------------------------------------------------------------
    def environment(self) -> str:
        base = STUB_REQUIREMENTS if self.stub else (self.requirements.read_text().rstrip() + "\n")
        return base + EXTRA_REQUIREMENTS

    def stage(self, d: Deployment) -> None:
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda site: self._stage_node(d, site), d.sites))

    def _stage_node(self, d: Deployment, site: Site) -> None:
        host, name = site.host, site.node.name
        if self.opt.get("install_environment", True):
            self.install_environment(d, site, self.environment())
        else:
            d.log(f"stage {name}: python environment (install_environment: false, using ~/{ENV_DIR} as it is)")
        d.log(f"stage {name}: fedlab packages")
        host.put(CORE_PACKAGE, f"{host.home}/{PYLIB}/fedlab_core")
        host.put(ASYNC_PACKAGE, f"{host.home}/{PYLIB}/fedlab_async")
        host.put(self.app_path / "fedlab_compass", f"{host.home}/{PYLIB}/fedlab_compass")
        self.stage_code(d, site, compass_source=not self.stub)  # the stub needs the scripts, not the COMPASS source
        if site.node.role == "client":
            d.log(f"stage {name}: raw data")
            host.put(self._tsv(site.node.site), f"{host.home}/data/{site.node.site}.tsv")

    def _tsv(self, site: str) -> Path:
        return Path(str(self.opt[f"{site}_tsv"])).expanduser()

    # ---- the run ------------------------------------------------------------------------------------
    def _python(self, site: Site) -> str:
        return f"{site.host.home}/{ENV_DIR}/bin/python"

    def _env(self, site: Site) -> dict[str, str]:
        return {"PYTHONPATH": f"{site.host.home}/{PYLIB}"}

    def _command(self, site: Site, args: str, *, check: bool = True) -> tuple[int, str]:
        code, output = site.host.execute(f"{self._python(site)} -m fedlab_async {args}", env=self._env(site))
        if check and code:
            raise RuntimeError(f"fedlab-async {args.split()[0]} failed on {site.node.name} (exit {code}):\n{output.strip()}")
        return code, output

    def _options(self, d: Deployment) -> dict:
        o = self.opt
        home = d.server.host.home
        nodes = {}
        for site, node_id in NODE_IDS.items():
            asset = measure("source", self._tsv(site).resolve().as_uri())  # hash the local file once
            client = next(c for c in d.clients if c.node.site == site)
            nodes[node_id] = {
                "site": site, "source": f"file://{client.host.home}/data/{site}.tsv", "sha256": asset.sha256, "bytes": asset.bytes,
            }
        options = {
            "code_root": f"{home}/{CODE_DIR}", "backend": o["backend"], "seed": int(o["seed"]), "rounds": int(o["rounds"]),
            "patience": int(o["patience"]), "local-epochs": int(o["local_epochs"]), "nodes": nodes,
        }
        if not self.stub:
            options["compass_source"] = f"{home}/{SOURCE_DIR}"
        return options

    def _site_file(self, site: Site) -> dict:
        home = site.host.home
        node = {"site": site.node.site, "workdir": f"{home}/site", "cache": f"{home}/cache", "code_root": f"{home}/{CODE_DIR}"}
        if not self.stub:
            node["compass_source"] = f"{home}/{SOURCE_DIR}"
        run = {k: v for k, v in self.run_config().items() if k not in ("num-server-rounds", "patience")}
        return {"node_id": NODE_IDS[site.node.site], "node": node, "run": run}

    def start(self, d: Deployment) -> None:
        server = d.server
        d.log("start: run and site configuration")
        server.host.put_text(
            yaml.safe_dump({"program": "fedlab_compass.async_run:build", "options": self._options(d)}), f"{server.host.home}/run.yaml"
        )
        for site in d.clients:
            site.host.put_text(yaml.safe_dump(self._site_file(site)), f"{site.host.home}/site.yaml")
        d.log("start: fedlab-async init")
        self._command(server, f"init {server.host.home}/{RUN_DIR} --config {server.host.home}/run.yaml --force")

    def run(self, d: Deployment) -> str:
        server = d.server
        run = f"{server.host.home}/{RUN_DIR}"
        scratch = d.state_dir / "async"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        step = 0
        while "finished" not in self._command(server, f"status {run}")[1]:
            step += 1
            d.log(f"step {step}: emit")
            requests = [Path(line.strip()) for line in self._command(server, f"emit {run}")[1].splitlines() if line.strip().endswith(".req")]
            for site in d.clients:  # the server's request files go to the sites they are for
                for request in (r for r in requests if r.name.endswith(f"-n{NODE_IDS[site.node.site]}.req")):
                    server.host.get(str(request), scratch / "requests")
                    site.host.put(scratch / "requests" / request.name, f"{site.host.home}/exchange/requests/{request.name}")
            d.log(f"step {step}: {len(d.clients)} site(s) run their requests")
            with ThreadPoolExecutor() as pool:
                outputs = list(pool.map(lambda site: self._run_client(d, site), d.clients))
            for site, output in zip(d.clients, outputs):  # the response files come back to the server
                for line in output.splitlines():
                    if line.startswith("wrote "):
                        name = Path(line.split(" ", 1)[1].strip()).name
                        site.host.get(f"{site.host.home}/exchange/responses/{name}", scratch / "responses" / site.node.site)
                        server.host.put(scratch / "responses" / site.node.site / name, f"{run}/exchange/responses/{name}")
            code, output = self._command(server, f"collect {run}", check=False)
            for line in output.splitlines():
                d.log(f"  {line}")
            if code:
                raise RuntimeError(f"step {step}: the server could not advance:\n{output.strip()}")
        return self._command(server, f"status {run}")[1].split()[1]

    def _run_client(self, d: Deployment, site: Site) -> str:
        home = site.host.home
        args = f"client --exchange {home}/exchange --state-dir {home}/state --site-file {home}/site.yaml --client-factory fedlab_compass.neutral_client:make_client"
        code, output = self._command(site, args)
        return output

    # ---- results -------------------------------------------------------------------------------------
    def collect(self, d: Deployment, dest: Path) -> list[Path]:
        server = d.server
        run = f"{server.host.home}/{RUN_DIR}"
        server.host.mkdir(f"{run}/results")
        server.host.get(f"{run}/results", dest)
        audit = f"{run}/private/state.pkl.audit.jsonl"
        if server.host.execute(f"test -f {audit}")[0] == 0:
            server.host.get(audit, dest)
        return sorted(p for p in dest.rglob("*") if p.is_file())

    def stop(self, d: Deployment) -> None:
        """Nothing runs between steps, so there is nothing to stop."""
