"""fedlab CLI: check / ping / up / status / start / stop / ssh / destroy."""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import ssh as sshmod
from .config import Node, Settings, load_settings
from .providers import NodeInfo, Provider, get_provider

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
        for i in infos:
            if i.name in {n.name for n in ctx.nodes} and i.public_ip:
                ok = sshmod.wait_for_ssh(i.public_ip)
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
    for col in ("node", "state", "public ip", "uptime", "ssh"):
        table.add_column(col)
    for i in infos:
        table.add_row(i.name, i.state, i.public_ip or "-", _fmt_uptime(i.running_since), f"ssh {i.name}" if i.public_ip else "-")
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


@app.command()
def up(config: Optional[Path] = ConfigOpt, node: Optional[list[str]] = NodeOpt):
    """Create the VMs (idempotent; starts any that are stopped) and wait for SSH."""
    ctx = Ctx(config, node)
    errors = _run_parallel(ctx, "up", lambda n: ctx.for_node(n).up(n))
    _finish(ctx, errors, wait_ssh=True)


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


@app.command()
def destroy(
    config: Optional[Path] = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be deleted"),
):
    """Delete ALL resources tagged for this run (instances, disks, firewalls, keys)."""
    ctx = Ctx(config)  # always the whole fleet
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
            console.print(f"  {line}")
    if dry_run or not plan:
        return
    if not yes and not typer.confirm(f"Delete all of the above (run '{ctx.cfg.run_name}')?"):
        raise typer.Abort()
    for prov, nodes in by_provider.items():
        ctx.provider(prov).destroy(nodes, dry_run=False)
    infos = [NodeInfo(n.name, n.provider, n.location, "absent") for n in ctx.nodes]
    sshmod.write_nodes_json(ctx.cfg, infos, ctx.key)
    sshmod.write_ssh_config(ctx.cfg, infos, ctx.key)
    console.print("[green]Destroyed.[/] (Local keypair in .fedlab/ was kept; delete it manually if desired.)")


if __name__ == "__main__":
    app()
