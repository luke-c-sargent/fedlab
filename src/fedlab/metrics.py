"""Per-host resource metrics: a stdlib sampler runs on every node as a service and records cumulative counters.

CPU, memory and GPU are instantaneous; disk and network are cumulative byte counters, so totals are exact and
rates come from differences. `collect` stops the samplers, copies their files to `<dest>/metrics/`, and writes
`summary.json` (one entry per host, for the whole sampling span and for the run window marked by `mark`).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import shlex
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from .modules.base import Deployment

INTERVAL = 1  # seconds between samples
SERVICE = "metrics"
REMOTE_DIR = "metrics"
SAMPLER_PATH = ".fedlab/metrics_sampler.py"
COLUMNS = ["epoch", "cpu_pct", "hot_core_pct", "mem_used_mb", "disk_read_b", "disk_write_b", "net_rx_b", "net_tx_b",
           "gpu_util_pct", "gpu_mem_used_mb", "gpu_power_w"]

SAMPLER = r'''
import os, re, subprocess, sys, time

def read(path):
    try:
        return open(path).read()
    except OSError:
        return ""

def cpu():
    out = {}
    for line in read("/proc/stat").splitlines():
        if line.startswith("cpu"):
            f = line.split()
            v = list(map(int, f[1:9]))
            out[f[0]] = (sum(v), v[3] + v[4])  # total, idle + iowait
    return out

def mem():
    m = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/meminfo").splitlines() if l.strip()}
    return ((m["MemTotal"] - m["MemAvailable"]) // 1024, m["MemTotal"] // 1024) if m else ("", "")

DISK = re.compile(r"^(sd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+)$")  # whole disks, so partitions are not counted twice

def disk():
    r = w = 0
    for line in read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) > 9 and DISK.match(f[2]):
            r, w = r + int(f[5]) * 512, w + int(f[9]) * 512
    return r, w

SKIP = re.compile(r"^(lo|docker\d*|veth.*|br-.*)$")  # container bridges would count the same bytes twice

def net():
    rx = tx = 0
    for line in read("/proc/net/dev").splitlines()[2:]:
        name, _, rest = line.partition(":")
        if rest and not SKIP.match(name.strip()):
            f = rest.split()
            rx, tx = rx + int(f[0]), tx + int(f[8])
    return rx, tx

def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0].split(",")
        return [x.strip() for x in out] if len(out) == 5 else [""] * 5
    except Exception:
        return [""] * 5

interval, path = float(sys.argv[1]), sys.argv[2]
os.makedirs(os.path.dirname(path), exist_ok=True)
g = gpu()
out = open(path, "w", buffering=1)
print("#meta,%s,%d,%s,%s,%s" % (os.uname().nodename, max(1, len(cpu()) - 1), mem()[1], g[2], g[4].replace(",", " ")), file=out)
print("epoch,cpu_pct,hot_core_pct,mem_used_mb,disk_read_b,disk_write_b,net_rx_b,net_tx_b,gpu_util_pct,gpu_mem_used_mb,gpu_power_w", file=out)
prev = cpu()
while True:
    time.sleep(interval)
    cur = cpu()
    busy = {k: 100.0 * (1 - (cur[k][1] - prev[k][1]) / max(1, cur[k][0] - prev[k][0])) for k in cur}
    prev = cur
    hot = max([v for k, v in busy.items() if k != "cpu"] or [0])
    g = gpu()
    print("%.1f,%.1f,%.1f,%s,%d,%d,%d,%d,%s,%s,%s" % (time.time(), busy.get("cpu", 0), hot, mem()[0], *disk(), *net(), g[0], g[1], g[3]), file=out)
'''


def start(d: Deployment) -> None:
    """Start a sampler on every node (also run again after a reboot: it appends to a fresh file)."""
    for site in [*d.sites, *d.observers]:
        host = site.host
        host.put_text(SAMPLER, f"~/{SAMPLER_PATH}")
        host.start_service(SERVICE, f"python3 {host.home}/{SAMPLER_PATH} {INTERVAL} {host.home}/{REMOTE_DIR}/{site.node.name}.csv", {})


def mark_in(state_dir: Path, event: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "events.jsonl", "a") as f:
        f.write(json.dumps({"event": event, "epoch": time.time()}) + "\n")


def mark(d: Deployment, event: str) -> None:
    """Record a named moment (for example `run_start`) in this deployment's event log."""
    mark_in(d.state_dir, event)


@contextmanager
def phase(state_dir: Path, name: str):
    """Record `<name>_start` and `<name>_end`; the summary reports the difference as `phases[name]`."""
    mark_in(state_dir, f"{name}_start")
    try:
        yield
    finally:
        mark_in(state_dir, f"{name}_end")


def reset(state_dir: Path) -> None:
    """Forget the events and transfers of an earlier run."""
    for name in ("events.jsonl", "transfers.jsonl", "rtt.jsonl", "steps.jsonl", "pulls.jsonl"):
        (state_dir / name).unlink(missing_ok=True)


PROBE = ("import socket, sys, time, statistics; ms = []\n"
         "for _ in range(5):\n"
         "    t = time.time(); socket.create_connection((sys.argv[1], int(sys.argv[2])), 10).close(); ms.append((time.time() - t) * 1000)\n"
         "print('%.1f %.1f %.1f' % (min(ms), statistics.median(ms), max(ms)))")


def probe_rtt(d: Deployment, address: str, port: int) -> dict:
    """TCP connect time from every client to `address:port` (min, median, max ms over 5 tries). Also logged as an event."""
    out = {}
    for site in d.clients:
        code, text = site.host.execute(f"python3 -c {shlex.quote(PROBE)} {shlex.quote(address)} {port}")
        try:
            lo, mid, hi = (float(x) for x in text.split()[-3:])
            out[site.node.name] = {"min": lo, "median": mid, "max": hi}
        except ValueError:
            d.log(f"metrics: rtt probe from {site.node.name} failed ({text.strip()[-120:]})")
    with open(d.state_dir / "rtt.jsonl", "a") as f:
        f.write(json.dumps({"epoch": time.time(), "target": f"{address}:{port}", "ms": out}) + "\n")
    return out


def run_info(d: Deployment) -> dict:
    """What produced these results: versions, commit, configuration and nodes."""
    root = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip() or None  # noqa: E731
    cfg = json.dumps(d.cfg.model_dump(mode="json") if hasattr(d.cfg, "model_dump") else repr(d.cfg), sort_keys=True, default=str)
    return {
        "flwr": version("flwr"), "python": platform.python_version(), "fedlab_commit": git("rev-parse", "HEAD"),
        "fedlab_dirty": bool(git("status", "--porcelain")), "module": d.cfg.module,
        "mode": "async" if "async" in (d.cfg.module or "") else d.cfg.federation,  # async, direct or orchestrated
        "config_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
        "nodes": [{"name": s.node.name, "role": s.node.role, "site": s.node.site, "provider": s.node.provider,
                   "location": s.node.location, "machine_type": s.node.machine_type, "gpu": s.node.gpu} for s in d.sites],
    }


def transfer(d: Deployment, *, step: int, direction: str, name: str, nbytes: int, seconds: float, src: str, dst: str) -> None:
    """Record one application-level file transfer (a request or response file moved between hosts)."""
    row = {"step": step, "direction": direction, "file": name, "bytes": nbytes, "seconds": round(seconds, 3), "src": src, "dst": dst}
    d.state_dir.mkdir(parents=True, exist_ok=True)
    with open(d.state_dir / "transfers.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def pull(d: Deployment, *, host: str, image: str, seconds: float, size_bytes: int | None) -> None:
    """Record how long a node took to pull an image and how large the image is on disk."""
    d.state_dir.mkdir(parents=True, exist_ok=True)
    with open(d.state_dir / "pulls.jsonl", "a") as f:
        f.write(json.dumps({"host": host, "image": image, "seconds": round(seconds, 1), "size_bytes": size_bytes}) + "\n")


def step(d: Deployment, step: int, *, emit: float, collect: float, sites: dict[str, float]) -> None:
    """Record how one async step's time was spent: the server's emit and collect, and each site's compute."""
    row = {"step": step, "emit_s": round(emit, 3), "collect_s": round(collect, 3), "sites_s": sites}
    d.state_dir.mkdir(parents=True, exist_ok=True)
    with open(d.state_dir / "steps.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def parse(text: str) -> tuple[dict, list[dict]]:
    meta, rows = {}, []
    lines = text.splitlines()
    if lines and lines[0].startswith("#meta"):
        _, host, vcpus, ram, gpu_mem, gpu_name = (lines[0].split(",") + [""] * 6)[:6]
        meta = {"hostname": host, "vcpus": int(vcpus or 0), "ram_mb": int(ram) if ram else None,
                "gpu_mem_mb": float(gpu_mem) if gpu_mem else None, "gpu": gpu_name or None}
        lines = lines[1:]
    for fields in list(csv.reader(io.StringIO("\n".join(lines))))[1:]:
        if len(fields) != len(COLUMNS):
            continue  # a row cut short when the sampler was stopped
        try:
            rows.append({k: (float(v) if v else None) for k, v in zip(COLUMNS, fields)})
        except ValueError:
            continue
    return meta, rows


def summarize(rows: list[dict], t0: float | None = None, t1: float | None = None) -> dict | None:
    rows = [r for r in rows if (t0 is None or r["epoch"] >= t0) and (t1 is None or r["epoch"] <= t1)]
    if len(rows) < 2:
        return None
    col = lambda k: [r[k] for r in rows if r.get(k) is not None]
    span = rows[-1]["epoch"] - rows[0]["epoch"]
    out = {"seconds": round(span, 1), "samples": len(rows), "cpu_avg_pct": round(statistics.mean(col("cpu_pct")), 1),
           "cpu_peak_pct": round(max(col("cpu_pct")), 1), "hottest_core_peak_pct": round(max(col("hot_core_pct")), 1)}
    if col("mem_used_mb"):
        out["mem_peak_mb"] = max(col("mem_used_mb"))
    for name in ("disk_read_b", "disk_write_b", "net_rx_b", "net_tx_b"):
        v = col(name)
        if v:
            out[name.replace("_b", "_mb")] = round((v[-1] - v[0]) / 1e6, 2)  # counters are cumulative
    for name, label in (("net_rx_b", "net_rx_peak_mbps"), ("net_tx_b", "net_tx_peak_mbps")):
        pairs = [(a, b) for a, b in zip(rows, rows[1:]) if a.get(name) is not None and b.get(name) is not None and b["epoch"] > a["epoch"]]
        if pairs:
            out[label] = round(max((b[name] - a[name]) * 8 / 1e6 / (b["epoch"] - a["epoch"]) for a, b in pairs), 2)
    util = col("gpu_util_pct")
    if util:
        power = col("gpu_power_w")
        out.update(gpu_avg_pct=round(statistics.mean(util), 1), gpu_peak_pct=max(util), gpu_busy_over_50_pct=round(100 * sum(u > 50 for u in util) / len(util), 1),
                   gpu_mem_peak_mb=max(col("gpu_mem_used_mb")))
        if power:
            out["gpu_power_avg_w"] = round(statistics.mean(power), 1)
            out["gpu_energy_wh"] = round(statistics.mean(power) * span / 3600, 2)
    return out


def collect(d: Deployment, dest: Path) -> dict:
    """Stop the samplers, copy their files to `dest/metrics/`, write and return the summary."""
    events = {}
    log = d.state_dir / "events.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            e = json.loads(line)
            events[e["event"]] = e["epoch"]  # the latest of each
    out = dest / REMOTE_DIR
    out.mkdir(parents=True, exist_ok=True)
    phases = {k[:-6]: round(events[k[:-6] + "_end"] - v, 1) for k, v in events.items() if k.endswith("_start") and k[:-6] + "_end" in events}
    summary: dict = {"events": events, "phases": phases, "run_info": run_info(d), "hosts": {}}
    for site in [*d.sites, *d.observers]:
        site.host.stop_service(SERVICE)
        name = site.node.name
        try:
            site.host.get(f"~/{REMOTE_DIR}/{name}.csv", out)  # `get` copies into a directory
        except Exception as e:  # noqa: BLE001  a missing file must not lose the other hosts' metrics
            d.log(f"metrics: {name}: could not copy ({e})")
            continue
        meta, rows = parse((out / f"{name}.csv").read_text())
        entry = {"role": site.node.role, "site": site.node.site, **meta, "whole": summarize(rows)}
        if "run_start" in events and "run_end" in events:
            entry["run"] = summarize(rows, events["run_start"], events["run_end"])
        summary["hosts"][name] = entry
    # Upper bound on billed egress: bytes sent by hosts of each provider (the counters cannot tell cross-cloud from same-cloud).
    egress: dict = {}
    for site in [*d.sites, *d.observers]:
        whole = (summary["hosts"].get(site.node.name) or {}).get("whole") or {}
        egress[site.node.provider] = round(egress.get(site.node.provider, 0) + whole.get("net_tx_mb", 0), 2)
    summary["egress_by_provider_mb"] = egress
    steps_file = d.state_dir / "steps.jsonl"
    if steps_file.exists():
        shutil.copyfile(steps_file, out / "steps.jsonl")
        steps = [json.loads(line) for line in steps_file.read_text().splitlines()]
        idle: dict = {}
        for s in steps:  # a site idles while the slowest site of the step is still working
            slowest = max(s["sites_s"].values(), default=0)
            for name, seconds in s["sites_s"].items():
                idle[name] = round(idle.get(name, 0) + slowest - seconds, 3)
        summary["async_steps"] = {"steps": len(steps), "emit_s": round(sum(s["emit_s"] for s in steps), 3),
                                  "collect_s": round(sum(s["collect_s"] for s in steps), 3),
                                  "site_compute_s": {n: round(sum(s["sites_s"].get(n, 0) for s in steps), 3) for n in idle},
                                  "site_idle_s": idle}
    pulls = d.state_dir / "pulls.jsonl"
    if pulls.exists():
        summary["image_pulls"] = [json.loads(line) for line in pulls.read_text().splitlines()]
    probes = d.state_dir / "rtt.jsonl"
    if probes.exists():
        summary["rtt"] = [json.loads(line) for line in probes.read_text().splitlines()]
    ledger = d.state_dir / "transfers.jsonl"
    if ledger.exists():
        shutil.copyfile(ledger, out / "transfers.jsonl")
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        totals: dict = {}
        for r in rows:
            t = totals.setdefault(f"{r['src']} -> {r['dst']}", {"files": 0, "bytes": 0, "seconds": 0.0})
            t["files"] += 1
            t["bytes"] += r["bytes"]
            t["seconds"] = round(t["seconds"] + r["seconds"], 3)
        summary["transfers"] = totals
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
