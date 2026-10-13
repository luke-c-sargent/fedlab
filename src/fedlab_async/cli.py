"""fedlab-async: run a federated-learning loop as files that move between a server and its sites.

Server side, in a run directory:
    init      create a run from a config naming a program factory
    emit      write this step's request files (put them where the sites can reach them)
    collect   read the response files; once every site has answered, aggregate and advance
    status    where the run is and who is outstanding
    reissue   restore request files that were lost or damaged in transit

Site side:
    client    run a request file and write the response file
    inspect   show the header of any exchange file
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from .envelope import EnvelopeError, peek, read
from .layout import Exchange
from .run import RunDir, add_paths, load_config, load_object


class _App(typer.Typer):
    """Prints a failure as one short line; FEDLAB_ASYNC_DEBUG=1 shows the full traceback."""

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except (typer.Exit, typer.Abort, SystemExit, KeyboardInterrupt):
            raise
        except Exception as error:
            if os.environ.get("FEDLAB_ASYNC_DEBUG"):
                raise
            typer.echo(f"error: {type(error).__name__}: {error}", err=True)
            typer.echo("(set FEDLAB_ASYNC_DEBUG=1 for the full traceback)", err=True)
            raise SystemExit(1)


app = _App(no_args_is_help=True, pretty_exceptions_enable=False, help=__doc__)
RunArg = typer.Argument(..., help="The run directory")


@app.callback()
def main(
    path: Optional[list[str]] = typer.Option(
        None, "--path", help="Add a directory to the import path (repeatable). Needed so the run's program and client "
        "classes can be found. Also read from FEDLAB_ASYNC_PATH (separated by ':').",
    ),
):
    add_paths([*(path or []), *os.environ.get("FEDLAB_ASYNC_PATH", "").split(os.pathsep)])


@app.command()
def init(
    run: Path = RunArg,
    config: Path = typer.Option(..., "--config", "-c", help="YAML/JSON with `program: package.module:factory` and `options:`"),
    force: bool = typer.Option(False, "--force", help="Replace an existing run's state"),
):
    """Create a run. The factory is called as factory(options: dict, run_dir: Path) and returns a Program."""
    from .server import Server

    spec = load_config(config)
    if "program" not in spec:
        raise ValueError(f"{config}: missing `program: package.module:factory`")
    run_dir = RunDir(run)
    if run_dir.exists() and not force:
        raise RuntimeError(f"{run} already holds a run; use --force to replace it")
    program = load_object(spec["program"])(spec.get("options", {}), run_dir.root)
    Server.create(run_dir.state, run_dir.exchange, program)
    status = run_dir.server().status()
    typer.echo(f"created run {status.run_id}")
    typer.echo(f"  next: {status.phase} (round {status.round}); `emit` writes the requests into {run_dir.exchange.root}/requests/")


@app.command()
def emit(run: Path = RunArg):
    """Write this step's request files. Calling it again lists the same files."""
    files = RunDir(run).server().emit()
    if not files:
        typer.echo("nothing to emit: the run is finished")
    for path in files:
        typer.echo(str(path))


@app.command()
def collect(run: Path = RunArg):
    """Read the responses; aggregate and advance once every site has answered."""
    result = RunDir(run).server().collect()
    for name, reason in result.rejected.items():
        typer.echo(f"rejected {name}: {reason}")
    if result.advanced:
        status = RunDir(run).server().status()
        typer.echo("all responses valid: advanced" + (" (run finished)" if status.done else f"; next: {status.phase} (round {status.round})"))
    else:
        typer.echo(f"not advanced; still waiting on node(s) {result.missing}" if result.missing else "nothing to collect: emit first")
        raise typer.Exit(2)


@app.command()
def status(run: Path = RunArg):
    """Where the run is and who is outstanding."""
    s = RunDir(run).server().status()
    typer.echo(f"run {s.run_id}")
    if s.done:
        typer.echo("finished")
        return
    typer.echo(f"round {s.round}, {s.phase}: " + ("requests are out" if s.emitted else "not emitted yet"))
    if s.emitted:
        typer.echo(f"  waiting on: {s.waiting_on or 'nobody (ready to collect)'}")
        typer.echo(f"  response files present: {s.received or 'none'}")


@app.command()
def reissue(
    run: Path = RunArg,
    node: Optional[int] = typer.Option(None, "--node", "-n", help="Only this node"),
    force: bool = typer.Option(False, "--force", help="Replace request files that look intact too"),
):
    """Restore outstanding request files that are missing or damaged, from the server's private copies."""
    restored = RunDir(run).server().reissue(node, force)
    for path in restored:
        typer.echo(f"restored {path}")
    if not restored:
        typer.echo("nothing to restore: every outstanding request file is intact")


@app.command()
def client(
    requests: Optional[list[Path]] = typer.Argument(None, help="Request files to run; default: this node's pending requests"),
    exchange: Path = typer.Option(..., "--exchange", "-e", help="The exchange directory (requests/ in, responses/ out)"),
    state_dir: Path = typer.Option(..., "--state-dir", help="This site's private state between requests"),
    site_file: Optional[Path] = typer.Option(None, "--site-file", "-s", help="YAML/JSON: node_id, node (node config), run (run config)"),
    app_spec: Optional[str] = typer.Option(None, "--app", help="Flower ClientApp as package.module:name"),
    plain: Optional[str] = typer.Option(None, "--plain", help="Plain client function as package.module:name"),
):
    """Run request files through this site's client and write the response files."""
    if bool(app_spec) == bool(plain):
        raise ValueError("give exactly one of --app (Flower ClientApp) or --plain (client function)")
    site = load_config(site_file) if site_file else {}
    ex = Exchange(exchange)
    todo = list(requests or [])
    if not todo:
        if "node_id" not in site:
            raise ValueError("no request files given and no node_id in the site file to find them by")
        todo = pending_requests(ex, int(site["node_id"]))
        if not todo:
            typer.echo("nothing to do: no pending request for this node")
            return
    for request in todo:
        if app_spec:
            from .adapters.flower_client import run_client

            out = run_client(request, load_object(app_spec), ex, state_dir, site.get("node", {}), site.get("run", {}))
        else:
            from .adapters.plain import run_plain_client

            out = run_plain_client(request, load_object(plain), ex, state_dir)
        typer.echo(f"wrote {out}")


def pending_requests(exchange: Exchange, node_id: int) -> list[Path]:
    """This node's request files that have no response yet, oldest exchange first."""
    found = []
    for path in sorted((exchange.root / "requests").glob("*.req")):
        header = peek(path)
        if header.node_id == node_id and not exchange.response(header.round, header.phase, node_id).exists():
            found.append((header.seq, path))
    return [path for _, path in sorted(found)]


@app.command()
def inspect(file: Path = typer.Argument(..., help="Any request, response or state file")):
    """Show the header of an exchange file and whether its payload is intact."""
    header = peek(file)
    for key, value in asdict(header).items():
        typer.echo(f"{key}: {value}")
    try:
        _, payload = read(file)
        typer.echo(f"payload: {len(payload):,} bytes, hash ok")
    except EnvelopeError as error:
        typer.echo(f"payload: DAMAGED ({error})")
        raise typer.Exit(1)
