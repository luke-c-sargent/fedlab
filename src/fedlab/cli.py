"""fedlab CLI: check / ping / up / deploy / run / collect / experiment / status / start / stop / ssh / destroy."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.markup import escape
from rich.console import Console
from rich.table import Table

from . import ssh as sshmod
from .config import Node, Settings, load_settings
from .modules import REGISTRY, Deployment, FLModule, Site, get_module, module_class
from .providers import NodeInfo, Provider, get_provider
from .readiness import wait_ready
from .remote import SshHost

app = typer.Typer(no_args_is_help=True, help="Launch and tear down multi-cloud VMs for Flower tests.")
console = Console()
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config YAML (default: ./config.yaml)")
NodeOpt = typer.Option(None, "--node", "-n", help="Limit to these node names (repeatable)")


class Ctx:
    def __init__(self, config: Optional[Path], only: Optional[list[str]] = None):
        self.cfg: Settings = load_settings(config)
        self.key = sshmod.ensure_keypair(self.cfg)
        self.nodes = self.cfg.resolved_nodes()
        self.all_nodes = self.nodes
        if only:
            unknown = set(only) - {n.name for n in self.nodes}
            if unknown:
                raise typer.BadParameter(f"unknown node(s): {sorted(unknown)}; known: {[n.name for n in self.nodes]}")
            self.nodes = [n for n in self.nodes if n.name in only]
        self._providers: dict[str, Provider] = {}

    def provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = get_provider(name, self.cfg, self.key)
        return self._providers[name]

    def for_node(self, node: Node) -> Provider:
        return self.provider(node.provider)

    def describe_all(self) -> list[NodeInfo]:
        with ThreadPoolExecutor() as ex:
            return list(ex.map(lambda n: self.for_node(n).describe(n), self.all_nodes))

    def refresh_local_files(self) -> list[NodeInfo]:
        """Regenerate nodes.json + SSH config from the clouds' current state."""
        infos = self.describe_all()
        sshmod.write_nodes_json(self.cfg, infos, self.key)
        sshmod.write_ssh_config(self.cfg, infos, self.key)
        if not sshmod.ssh_include_present(self.cfg):
            line = sshmod.include_line(self.cfg)
            if typer.confirm(f"Add '{line}' to the top of ~/.ssh/config so `ssh <node>` works?", default=True):
                sshmod.add_ssh_include(self.cfg)
        return infos


def _run_parallel(ctx: Ctx, label: str, fn: Callable[[Node], None]) -> list[str]:
    errors: list[str] = []

    def work(node: Node) -> None:
        try:
            console.print(f"[cyan]{label}[/] {node.name}")
            fn(node)
            console.print(f"[green]done[/] {node.name}")
        except Exception as e:  # report per-node; don't abort the others
            errors.append(f"{node.name}: {type(e).__name__}: {e}")

    with ThreadPoolExecutor() as ex:
        list(ex.map(work, ctx.nodes))
    return errors


def _finish(ctx: Ctx, errors: list[str], wait_ssh: bool = False) -> None:
    infos = ctx.refresh_local_files()
    if wait_ssh:
        waiting = [i for i in infos if i.name in {n.name for n in ctx.nodes} and i.public_ip]
        with ThreadPoolExecutor() as ex:
            results = list(ex.map(lambda i: sshmod.wait_for_login(i.public_ip, ctx.cfg.ssh_user, ctx.key), waiting))
        for i, ok in zip(waiting, results):
            console.print(f"ssh {i.name}: {'[green]reachable' if ok else '[red]not reachable yet'}")
    _print_status(ctx.cfg, infos)
    if errors:
        for e in errors:
            console.print(f"[red]ERROR[/] {e}")
        raise typer.Exit(1)


def _fmt_uptime(since: datetime | None) -> str:
    if not since:
        return "-"
    secs = (datetime.now(timezone.utc) - since).total_seconds()
    d, rem = divmod(int(secs), 86400)
    return f"{d}d {rem // 3600}h" if d else f"{rem // 3600}h {rem % 3600 // 60}m"


def _print_status(cfg: Settings, infos: list[NodeInfo]) -> None:
    table = Table()
    for col in ("node", "role", "state", "public ip", "uptime", "ssh"):
        table.add_column(col)
    roles = {n.name: n.role for n in cfg.resolved_nodes()}
    for i in infos:
        table.add_row(i.name, roles.get(i.name, "-"), i.state, i.public_ip or "-", _fmt_uptime(i.running_since), f"ssh {i.name}" if i.public_ip else "-")
    console.print(table)
    for i in infos:
        if i.running_since:
            days = (datetime.now(timezone.utc) - i.running_since).total_seconds() / 86400
            if days > cfg.warn_after_days:
                console.print(f"[yellow]warning:[/] {i.name} has been running {days:.1f} days (billing!)")


@app.command()
def check(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Verify credentials, permissions and requirements. Read-only; creates nothing in any cloud."""
    cfg = load_settings(config)
    nodes = cfg.resolved_nodes()
    if node:
        unknown = set(node) - {n.name for n in nodes}
        if unknown:
            raise typer.BadParameter(f"unknown node(s): {sorted(unknown)}; known: {[n.name for n in nodes]}")
        nodes = [n for n in nodes if n.name in node]
    key = sshmod.existing_or_ephemeral_keypair(cfg)  # don't write a keypair just to check

    failed = 0
    by_provider: dict[str, list[Node]] = {}
    for n in nodes:
        by_provider.setdefault(n.provider, []).append(n)
    with ThreadPoolExecutor() as ex:
        futures = {p: ex.submit(get_provider(p, cfg, key).check, ns) for p, ns in by_provider.items()}
        for prov, fut in futures.items():
            for r in fut.result():
                failed += not r.ok
                mark = "[green]ok  [/]" if r.ok else "[red]FAIL[/]"
                console.print(f"{mark} {r.name}" + (f" [dim]{r.detail}[/]" if r.detail else ""), highlight=False)
    if cfg.module:
        try:
            problems = get_module(cfg).validate(cfg.resolved_nodes())
        except ValueError as e:
            problems = [str(e)]
        if problems:
            for p in problems:
                console.print(f"[red]FAIL[/] module {cfg.module}: {p}", highlight=False)
            failed += len(problems)
        else:
            console.print(f"[green]ok  [/] module {cfg.module}: configuration and local data", highlight=False)
    if not shutil.which("ssh"):
        console.print("[red]FAIL[/] local: `ssh` not found on PATH")
        failed += 1
    if failed:
        console.print(f"[red]{failed} check(s) failed[/]")
        raise typer.Exit(1)
    console.print("[green]All checks passed.[/]")


@app.command()
def ping(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Check SSH reachability and key auth on each node of a running stack. Read-only."""
    ctx = Ctx(config, node)

    def probe(n: Node) -> tuple[Node, NodeInfo, tuple[bool, str]]:
        info = ctx.for_node(n).describe(n)
        if not info.public_ip:
            return n, info, (False, f"no public IP (state: {info.state})")
        return n, info, sshmod.probe_ssh(info.public_ip, ctx.cfg.ssh_user, ctx.key)

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(probe, ctx.nodes))
    failed = 0
    for n, info, (ok, detail) in results:
        failed += not ok
        mark = "[green]ok  [/]" if ok else "[red]FAIL[/]"
        console.print(f"{mark} {n.name} [dim]{info.public_ip or '-'} {detail}[/]", highlight=False)
    if failed:
        raise typer.Exit(1)


def _up(ctx: Ctx) -> None:
    errors = _run_parallel(ctx, "up", lambda n: ctx.for_node(n).up(n))
    _finish(ctx, errors, wait_ssh=True)


@app.command()
def up(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Create the VMs (idempotent; starts any that are stopped) and wait for SSH."""
    _up(Ctx(config, node))


@app.command()
def start(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Start stopped VMs and refresh nodes.json / SSH config with the new IPs."""
    ctx = Ctx(config, node)
    errors = _run_parallel(ctx, "start", lambda n: ctx.for_node(n).start(n))
    _finish(ctx, errors, wait_ssh=True)


@app.command()
def stop(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Stop VMs (disks keep billing)."""
    ctx = Ctx(config, node)
    errors = _run_parallel(ctx, "stop", lambda n: ctx.for_node(n).stop(n))
    _finish(ctx, errors)


@app.command()
def status(config: Optional[Path] = ConfigOpt):
    """Show node states and IPs; regenerates nodes.json and the SSH config."""
    ctx = Ctx(config)
    _print_status(ctx.cfg, ctx.refresh_local_files())


@app.command(name="ssh")
def ssh_cmd(
    node: str = typer.Argument(..., help="Node name"),
    command: Optional[list[str]] = typer.Argument(None, help="Optional remote command"),
    config: Optional[Path] = ConfigOpt,
):
    """SSH into a node using the repo keypair."""
    ctx = Ctx(config, [node])
    info = ctx.for_node(ctx.nodes[0]).describe(ctx.nodes[0])
    if not info.public_ip:
        raise typer.BadParameter(f"{node} has no public IP (state: {info.state})")
    cmd = [
        "ssh", "-i", str(ctx.key.private_path), "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        f"{ctx.cfg.ssh_user}@{info.public_ip}", *(command or []),
    ]
    os.execvp("ssh", cmd)


def _destroy(ctx: Ctx, yes: bool = False, dry_run: bool = False) -> None:
    by_provider: dict[str, list[Node]] = {}
    for n in ctx.nodes:
        by_provider.setdefault(n.provider, []).append(n)

    plan: list[str] = []
    for prov, nodes in by_provider.items():
        plan += ctx.provider(prov).destroy(nodes, dry_run=True)
    if not plan:
        console.print("Nothing to delete.")
    else:
        console.print("[bold]Will delete:[/]")
        for line in plan:
            console.print(f"  {escape(line)}")  # lines start with "[aws region]", which rich would read as markup
    if dry_run or not plan:
        return
    if not yes and not typer.confirm(f"Delete all of the above (run '{ctx.cfg.run_name}')?"):
        raise typer.Abort()

    _phase(1, 2, f"delete {len(plan)} resource(s) ({', '.join(sorted(by_provider))} in parallel)")
    started = time.monotonic()
    with ThreadPoolExecutor() as ex:  # the clouds are independent, and each deletion mostly waits
        list(ex.map(lambda item: ctx.provider(item[0]).destroy(item[1], dry_run=False, log=_log), by_provider.items()))
    _phase(2, 2, "verify that nothing is left")
    leftovers: list[str] = []
    for prov, nodes in by_provider.items():
        leftovers += ctx.provider(prov).destroy(nodes, dry_run=True)
    infos = [NodeInfo(n.name, n.provider, n.location, "absent") for n in ctx.nodes]
    sshmod.write_nodes_json(ctx.cfg, infos, ctx.key)
    sshmod.write_ssh_config(ctx.cfg, infos, ctx.key)
    took = int(time.monotonic() - started)
    if leftovers:
        console.print(f"[yellow]Destroy finished in {took}s but these remain; run `fedlab destroy` again:[/]")
        for line in leftovers:
            console.print(f"  {escape(line)}")
        raise typer.Exit(1)
    console.print(f"[green]Destroyed in {took}s; nothing tagged for '{ctx.cfg.run_name}' remains.[/] (The local keypair in .fedlab/ was kept.)")


@app.command()
def destroy(
    config: Optional[Path] = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be deleted"),
):
    """Delete ALL resources tagged for this run (instances, disks, firewalls, keys)."""
    _destroy(Ctx(config), yes, dry_run)  # always the whole fleet


# ---- federated-learning modules ---------------------------------------------------


_START = time.monotonic()


def _log(message: str) -> None:
    """Print a progress line prefixed with the time since fedlab started (so a quiet step is visibly alive)."""
    elapsed = int(time.monotonic() - _START)
    console.print(f"[+{elapsed // 60:02d}:{elapsed % 60:02d}] {message}", markup=False, highlight=False)


def _phase(number: int, total: int, title: str) -> None:
    console.rule(f"[bold]{number}/{total} {title}", style="cyan")


def _module(ctx: Ctx) -> FLModule:
    try:
        module = get_module(ctx.cfg)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    problems = module.validate(ctx.all_nodes)
    for p in problems:
        console.print(f"[red]FAIL[/] {p}", highlight=False)
    if problems:
        raise typer.Exit(1)
    return module


def _deployment(ctx: Ctx) -> Deployment:
    infos = {i.name: i for i in ctx.describe_all()}
    sites = []
    for n in ctx.all_nodes:
        info = infos[n.name]
        if info.state != "running" or not info.public_ip:
            raise typer.BadParameter(f"{n.name} is not running (state: {info.state}); run `fedlab up`")
        sites.append(Site(n, SshHost(n.name, info.public_ip, ctx.cfg.ssh_user, ctx.key)))
    server = next(s for s in sites if s.node.role == "server")
    clients = [s for s in sites if s.node.role == "client"]
    return Deployment(ctx.cfg, server, clients, ctx.cfg.state_dir / "deploy", log=_log)


def _wait_ready(d: Deployment) -> None:
    with ThreadPoolExecutor() as ex:
        list(ex.map(lambda s: wait_ready(s.host, s.node.gpu, log=d.log), d.sites))


def _collect(ctx: Ctx, module: FLModule, d: Deployment, out: Optional[Path] = None) -> Path:
    dest = out or ctx.cfg.results_dir / ctx.cfg.run_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    files = module.collect(d, dest)
    console.print(f"[green]Collected {len(files)} file(s)[/] into {dest}")
    return dest


@app.command()
def modules():
    """List the available federated-learning modules."""
    table = Table()
    for col in ("module", "description"):
        table.add_column(col)
    for name in sorted(REGISTRY):
        table.add_row(name, module_class(name).description)
    console.print(table)


@app.command()
def prepare(config: Optional[Path] = ConfigOpt):
    """Run the module's local preparation step (for example, build data caches). Uses no cloud."""
    cfg = load_settings(config)
    try:
        module = get_module(cfg)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    try:
        module.prepare(_log)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        console.print("[red]prepare failed:[/]", escape(str(e)))
        raise typer.Exit(1)


@app.command()
def deploy(config: Optional[Path] = ConfigOpt):
    """Install the module's software and data on the running nodes, then start the federation."""
    ctx = Ctx(config)
    module = _module(ctx)
    d = _deployment(ctx)
    _phase(1, 3, "wait for first boot (cloud-init, GPU driver)")
    _wait_ready(d)
    _phase(2, 3, "install software and data")
    module.stage(d)
    _phase(3, 3, "start the federation")
    module.start(d)
    console.print("[green]Deployed.[/] Next: `fedlab run`")


@app.command()
def run(config: Optional[Path] = ConfigOpt):
    """Run the deployed experiment to completion."""
    ctx = Ctx(config)
    module = _module(ctx)
    run_id = module.run(_deployment(ctx))
    console.print(f"[green]Run {run_id} completed.[/] Next: `fedlab collect`")


@app.command()
def collect(config: Optional[Path] = ConfigOpt, out: Optional[Path] = typer.Option(None, "--out", "-o", help="Local directory for results")):
    """Copy results (and service logs) from the nodes to this machine."""
    ctx = Ctx(config)
    module = _module(ctx)
    _collect(ctx, module, _deployment(ctx), out)


@app.command()
def logs(
    node: str = typer.Argument(..., help="Node name"),
    service: str = typer.Argument("auto", help="superlink, supernode, or auto"),
    lines: int = typer.Option(80, "--lines", "-n"),
    config: Optional[Path] = ConfigOpt,
):
    """Show the Flower service log on a node."""
    ctx = Ctx(config, [node])
    n = ctx.nodes[0]
    site = next(s for s in _deployment(ctx).sites if s.node.name == n.name)
    name = ("superlink" if n.role == "server" else "supernode") if service == "auto" else service
    console.print(site.host.service_log(name, lines), markup=False, highlight=False)


@app.command()
def experiment(
    config: Optional[Path] = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
    keep: bool = typer.Option(False, "--keep", help="Do not destroy the fleet afterwards"),
    destroy_on_failure: bool = typer.Option(False, "--destroy-on-failure", help="Also destroy the fleet if the experiment fails"),
):
    """up -> deploy -> run -> collect -> destroy. The fleet is destroyed only after results are collected."""
    ctx = Ctx(config)
    module = _module(ctx)
    if not yes and not typer.confirm(
        f"Create {len(ctx.all_nodes)} VM(s), run module '{ctx.cfg.module}', collect results"
        f"{'' if keep else ', and destroy everything'}?"
    ):
        raise typer.Abort()
    d: Deployment | None = None
    collected = False
    try:
        _phase(1, 6, "create the VMs and wait for SSH")
        _up(ctx)
        d = _deployment(ctx)
        _phase(2, 6, "wait for first boot (cloud-init, GPU driver)")
        _wait_ready(d)
        _phase(3, 6, "install software and data")
        module.stage(d)
        _phase(4, 6, "start the federation")
        module.start(d)
        _phase(5, 6, "run the experiment")
        module.run(d)
        _phase(6, 6, "collect results")
        _collect(ctx, module, d)
        collected = True
    except (Exception, KeyboardInterrupt, typer.Exit) as e:
        console.print(f"[red]Experiment failed:[/] {type(e).__name__}: {e}", markup=False)
        if d is not None and not collected:
            try:
                _collect(ctx, module, d)  # logs are worth having
            except Exception as ce:
                console.print(f"[yellow]could not collect after failure:[/] {ce}", markup=False)
    finally:
        if d is not None:
            try:
                module.stop(d)
            except Exception:
                pass
    if not collected:
        if destroy_on_failure:
            _destroy(ctx, yes=True)
        else:
            console.print("[yellow]The fleet is still running (billing). Debug with `fedlab logs`, then `fedlab destroy`.[/]")
        raise typer.Exit(1)
    if keep:
        console.print("[yellow]--keep: fleet left running. `fedlab destroy` when done.[/]")
    else:
        _destroy(ctx, yes=True)


if __name__ == "__main__":
    app()
