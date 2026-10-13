"""Drive a whole file-based COMPASS run through the `fedlab-async` commands, however the commands are run.

`run(*args) -> (exit code, stdout, stderr)` is the only thing that differs between the in-process test, the
subprocess test and the opt-in real-backend test, so the driving logic is tested wherever it can run cheaply.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import synth_compass


def subprocess_runner(python: str, *pythonpath: Path):
    """Run `python -m fedlab_async ...` as a real process, as a site or server operator would."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}

    def run(*args):
        done = subprocess.run([python, "-m", "fedlab_async", *map(str, args)], capture_output=True, text=True, env=env)
        return done.returncode, done.stdout, done.stderr

    return run


def drive(run, tmp: Path, code_root: Path, *, backend: str, rounds: int, run_settings: dict | None = None,
          compass_source: Path | None = None, patience: int = 10, seed: int = 42) -> Path:
    """Write synthetic data, init a run, and loop emit / client / collect until it is finished. Returns the run dir."""
    data = tmp / "remote"
    data.mkdir()
    tcga, gtex = synth_compass.write_tsvs(data)
    settings = {"backend": backend, "seed": seed, **(run_settings or {})}
    options = {"code_root": str(code_root), "rounds": rounds, "patience": patience,
               "nodes": {1: {"site": "tcga", "source": tcga.as_uri()}, 2: {"site": "gtex", "source": gtex.as_uri()}}, **settings}
    if compass_source:
        options["compass_source"] = str(compass_source)
    (tmp / "run.yaml").write_text(yaml.safe_dump({"program": "fedlab_compass.async_run:build", "options": options}))
    run_dir = tmp / "run"
    code, out, err = run("init", run_dir, "--config", tmp / "run.yaml")
    assert code == 0, err

    sites = {}
    for node, site in ((1, "tcga"), (2, "gtex")):
        node_config = {"site": site, "workdir": str(tmp / f"workdir{node}"), "cache": str(tmp / f"cache{node}"), "code_root": str(code_root)}
        if compass_source:
            node_config["compass_source"] = str(compass_source)
        sites[node] = tmp / f"site{node}.yaml"
        sites[node].write_text(yaml.safe_dump({"node_id": node, "node": node_config, "run": settings}))

    steps = 0
    while "finished" not in run("status", run_dir)[1]:
        code, out, err = run("emit", run_dir)
        assert code == 0, err
        for node, site_file in sites.items():
            code, out, err = run("client", "--exchange", run_dir / "exchange", "--state-dir", tmp / f"state{node}",
                                 "--site-file", site_file, "--app", "fedlab_compass.async_client:app")
            assert code == 0 and "wrote" in out, (node, err)
        code, out, err = run("collect", run_dir)
        assert code == 0, (out, err)
        steps += 1
        assert steps <= 1 + 2 * rounds, "the run did not finish in the expected number of steps"
    assert steps == 1 + 2 * rounds  # prepare, then train and evaluate per round; pass patience >= rounds
    return run_dir


def audit_lines(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "private" / "state.pkl.audit.jsonl").read_text().splitlines()]
