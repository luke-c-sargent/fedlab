"""Compare runs: read each run's `metrics/summary.json` and write one table and a standalone chart page.

Rows are runs; columns are the numbers worth comparing across execution styles (direct, orchestrated, async).
Writes `comparison.csv`, `comparison.json`, `comparison.md` and `comparison.html` (no scripts, no network needed)."""

from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path

# (column, heading, unit) for the charts, in display order
CHARTS = [
    ("total_s", "Total time (without collect)", "s"), ("run_s", "Run time", "s"), ("net_tx_run_mb", "Data sent by all hosts during the run", "MB"),
    ("orchestrator_in_mb", "Orchestrator: received from nodes", "MB"), ("orchestrator_out_mb", "Orchestrator: sent to nodes", "MB"),
    ("transfer_mb", "Async files moved between server and sites", "MB"), ("gpu_avg_pct", "GPU utilization (average over GPU hosts)", "%"),
    ("gpu_energy_wh", "GPU energy during the run", "Wh"), ("rtt_ms", "Round-trip to the server or orchestrator (median, worst client)", "ms"),
    ("pull_s", "Slowest image pull", "s"),
]
COLUMNS = ["run", "mode", "module", "flwr", "commit", "rounds", "train_loss", "validation_loss", "total_s", "run_s",
           "orchestrator_up_s", "up_s", "first_boot_s", "stage_s", "start_s", "net_tx_run_mb", "net_rx_run_mb", "net_tx_whole_mb",
           "disk_read_run_mb", "disk_write_run_mb", "cpu_avg_pct", "mem_peak_mb", "gpu_avg_pct", "gpu_energy_wh", "rtt_ms",
           "pull_s", "image_gb", "orchestrator_in_mb", "orchestrator_out_mb", "transfer_mb", "async_idle_s", "egress_aws_mb", "egress_gcp_mb"]
MODE_COLORS = {"direct": "#2b6cb0", "orchestrated": "#c05621", "async": "#2f855a"}


def _final_loss(run_dir: Path) -> tuple[int | None, float | None, float | None]:
    for history in sorted(run_dir.rglob("history.tsv")):
        rows = list(csv.DictReader(history.open(), delimiter="\t"))
        if rows:
            last = rows[-1]
            get = lambda *keys: next((float(last[k]) for k in keys if last.get(k) not in (None, "")), None)  # noqa: E731
            return len(rows), get("train_loss", "fit_train_loss"), get("validation_loss", "val_loss")
    return None, None, None


def extract(run_dir: Path) -> dict | None:
    """One row of numbers for the run in `run_dir` (a `results/<name>/<timestamp>` directory), or None if it has no metrics."""
    file = run_dir / "metrics" / "summary.json"
    if not file.exists():
        return None
    s = json.loads(file.read_text())
    info, phases, hosts = s.get("run_info", {}), s.get("phases", {}), s.get("hosts", {})
    mode = info.get("mode") or ("orchestrated" if "orchestrator_traffic" in s else "async" if "async_steps" in s else "direct")
    window = [h.get("run") or h.get("whole") or {} for h in hosts.values()]
    whole = [h.get("whole") or {} for h in hosts.values()]
    total = lambda key, ws=window: round(sum(w.get(key, 0) for w in ws), 2)  # noqa: E731
    gpus = [w for w in window if "gpu_avg_pct" in w]
    rounds, train, val = _final_loss(run_dir)
    traffic = (s.get("orchestrator_traffic") or {}).get("total", {})
    rtt = [r["median"] for probe in s.get("rtt", []) for r in probe["ms"].values()]
    pulls = s.get("image_pulls", [])
    row = {
        "run": f"{run_dir.parent.name}/{run_dir.name}", "mode": mode, "module": info.get("module"), "flwr": info.get("flwr"),
        "commit": (info.get("fedlab_commit") or "")[:8] + ("+" if info.get("fedlab_dirty") else ""), "rounds": rounds,
        "train_loss": train, "validation_loss": val,
        "total_s": round(sum(v for k, v in phases.items() if k in ("orchestrator_up", "up", "first_boot", "stage", "start", "run")), 1),
        "run_s": phases.get("run"), "orchestrator_up_s": phases.get("orchestrator_up"), "up_s": phases.get("up"),
        "first_boot_s": phases.get("first_boot"), "stage_s": phases.get("stage"), "start_s": phases.get("start"),
        "net_tx_run_mb": total("net_tx_mb"), "net_rx_run_mb": total("net_rx_mb"), "net_tx_whole_mb": total("net_tx_mb", whole),
        "disk_read_run_mb": total("disk_read_mb"), "disk_write_run_mb": total("disk_write_mb"),
        "cpu_avg_pct": round(statistics.mean([w["cpu_avg_pct"] for w in window if "cpu_avg_pct" in w] or [0]), 1),
        "mem_peak_mb": max([w.get("mem_peak_mb", 0) for w in window] or [0]),
        "gpu_avg_pct": round(statistics.mean(w["gpu_avg_pct"] for w in gpus), 1) if gpus else None,
        "gpu_energy_wh": round(sum(w.get("gpu_energy_wh", 0) for w in gpus), 2) if gpus else None,
        "rtt_ms": max(rtt) if rtt else None, "pull_s": max((p["seconds"] for p in pulls), default=None),
        "image_gb": round(max((p["size_bytes"] or 0 for p in pulls), default=0) / 1e9, 2) or None,
        "orchestrator_in_mb": round(traffic["bytes_in"] / 1e6, 2) if traffic else None,
        "orchestrator_out_mb": round(traffic["bytes_out"] / 1e6, 2) if traffic else None,
        "transfer_mb": round(sum(t["bytes"] for t in s.get("transfers", {}).values()) / 1e6, 2) if s.get("transfers") else None,
        "async_idle_s": round(sum(s["async_steps"]["site_idle_s"].values()), 1) if s.get("async_steps") else None,
        "egress_aws_mb": s.get("egress_by_provider_mb", {}).get("aws"), "egress_gcp_mb": s.get("egress_by_provider_mb", {}).get("gcp"),
    }
    row["_phases"] = {k: v for k, v in phases.items() if k in ("orchestrator_up", "up", "first_boot", "stage", "start", "run")}
    return row


def find_runs(roots: list[Path]) -> list[Path]:
    """Run directories under `roots`: each root is a run directory itself, or holds them (one or two levels down)."""
    found = []
    for root in roots:
        found += [root] if (root / "metrics" / "summary.json").exists() else sorted(p.parent.parent for p in root.glob("*/*/metrics/summary.json")) + sorted(p.parent.parent for p in root.glob("*/metrics/summary.json"))
    return sorted(set(found), key=lambda p: (p.parent.name, p.name))


def write(rows: list[dict], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    public = [{k: r.get(k) for k in COLUMNS} for r in rows]
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(public)
    (out / "comparison.json").write_text(json.dumps(public, indent=2))
    fmt = lambda v: "" if v is None else f"{v:g}" if isinstance(v, float) else str(v)  # noqa: E731
    shown = ["run", "mode", "rounds", "validation_loss", "total_s", "run_s", "net_tx_run_mb", "gpu_avg_pct", "gpu_energy_wh", "rtt_ms", "orchestrator_in_mb", "orchestrator_out_mb", "transfer_mb"]
    md = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)] + ["| " + " | ".join(fmt(r.get(c)) for c in shown) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    (out / "comparison.html").write_text(_page(rows))
    return [out / n for n in ("comparison.csv", "comparison.json", "comparison.md", "comparison.html")]


def _page(rows: list[dict]) -> str:
    e = html.escape
    charts = []
    for key, title, unit in CHARTS:
        vals = [(r["run"], r["mode"], r.get(key)) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        top = max(v for _, _, v in vals) or 1
        bars = "".join(
            f'<div class="row"><span class="lab">{e(run.split("/")[0])} <small>{e(mode)}</small></span>'
            f'<span class="bar" style="width:{max(1.0, 100 * v / top):.1f}%;background:{MODE_COLORS.get(mode, "#718096")}"></span>'
            f'<span class="val">{v:g} {unit}</span></div>' for run, mode, v in vals)
        charts.append(f"<section><h2>{e(title)}</h2>{bars}</section>")
    phase_colors = {"orchestrator_up": "#9f7aea", "up": "#4299e1", "first_boot": "#38b2ac", "stage": "#ecc94b", "start": "#ed8936", "run": "#48bb78"}
    stacks = []
    for r in rows:
        total = sum(r["_phases"].values()) or 1
        segs = "".join(f'<span style="width:{100 * v / total:.1f}%;background:{phase_colors[k]}" title="{k}: {v:g}s"></span>' for k, v in r["_phases"].items())
        stacks.append(f'<div class="row"><span class="lab">{e(r["run"].split("/")[0])} <small>{e(r["mode"])}</small></span><span class="stack">{segs}</span><span class="val">{total:.0f} s</span></div>')
    legend = " ".join(f'<i style="background:{c}"></i>{k}' for k, c in phase_colors.items())
    head = "".join(f"<th>{e(c)}</th>" for c in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{e('' if r.get(c) is None else f'{r[c]:g}' if isinstance(r.get(c), float) else str(r.get(c)))}</td>" for c in COLUMNS) + "</tr>" for r in rows)
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>fedlab run comparison</title>
<style>:root{{--bg:#fff;--fg:#1a202c;--mut:#718096;--line:#e2e8f0}}@media(prefers-color-scheme:dark){{:root{{--bg:#171923;--fg:#e2e8f0;--mut:#a0aec0;--line:#2d3748}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:0 16px 48px}}
h1{{font-size:1.5rem}}h2{{font-size:1rem;margin:24px 0 8px}}small{{color:var(--mut)}}.row{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.lab{{flex:0 0 220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.bar{{height:18px;border-radius:3px;min-width:2px}}.val{{color:var(--mut);white-space:nowrap}}
.stack{{display:flex;flex:1;height:18px;border-radius:3px;overflow:hidden}}.stack span{{display:block}}i{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}}
@media(max-width:600px){{.lab{{flex-basis:110px}}}}</style>
<h1>fedlab run comparison</h1><p><small>Bars are coloured by execution style: <b style="color:{MODE_COLORS['direct']}">direct</b>, <b style="color:{MODE_COLORS['orchestrated']}">orchestrated</b>, <b style="color:{MODE_COLORS['async']}">async</b>.</small></p>
<section><h2>Where the time went</h2>{''.join(stacks)}<p><small>{legend}</small></p></section>
{''.join(charts)}<section><h2>All numbers</h2><div class="scroll"><table><tr>{head}</tr>{body}</table></div></section>"""
