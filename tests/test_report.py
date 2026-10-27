import csv
import json

from fedlab import report


def make_run(root, name, stamp, mode, run_s, tx, gpu=None, orch=None, transfers=None, loss=0.5):
    d = root / name / stamp
    (d / "metrics").mkdir(parents=True)
    host = {"whole": {"net_tx_mb": tx * 2, "net_rx_mb": tx}, "run": {"cpu_avg_pct": 20.0, "mem_peak_mb": 3000, "net_tx_mb": tx, "net_rx_mb": tx / 2,
                                                                       "disk_read_mb": 1, "disk_write_mb": 2, **({"gpu_avg_pct": gpu, "gpu_energy_wh": 1.5} if gpu else {})}}
    summary = {"phases": {"up": 50.0, "first_boot": 30.0, "stage": 100.0, "start": 40.0, "run": run_s}, "run_info": {"flwr": "1.42.0", "module": "m", "mode": mode, "fedlab_commit": "abcdef123456"},
               "hosts": {"a": host, "b": host}, "egress_by_provider_mb": {"aws": 5.0, "gcp": 3.0}, "rtt": [{"target": "x:1", "ms": {"a": {"median": 2.0}, "b": {"median": 40.0}}}]}
    if orch:
        summary["orchestrator_traffic"] = {"total": {"bytes_in": orch[0], "bytes_out": orch[1]}}
        summary["image_pulls"] = [{"host": "a", "seconds": 60.0, "size_bytes": 1_470_000_000}]
    if transfers:
        summary["transfers"] = {"s -> c": {"files": 2, "bytes": transfers, "seconds": 1.0}}
        summary["async_steps"] = {"site_idle_s": {"c1": 3.0, "c2": 0.0}}
    (d / "metrics" / "summary.json").write_text(json.dumps(summary))
    (d / "run-1").mkdir()
    (d / "run-1" / "history.tsv").write_text(f"epoch\ttrain_loss\tvalidation_loss\n1\t0.9\t0.7\n2\t0.6\t{loss}\n")
    return d


def test_rows_carry_the_numbers_that_distinguish_the_styles(tmp_path):
    make_run(tmp_path, "direct", "t1", "direct", 300.0, 100.0, gpu=40.0)
    make_run(tmp_path, "orch", "t1", "orchestrated", 320.0, 100.0, gpu=38.0, orch=(2_000_000, 8_000_000))
    make_run(tmp_path, "async", "t1", "async", 500.0, 50.0, gpu=20.0, transfers=12_000_000)
    rows = {r["mode"]: r for r in (report.extract(d) for d in report.find_runs([tmp_path]))}
    assert set(rows) == {"direct", "orchestrated", "async"}
    assert rows["direct"]["total_s"] == 520.0 and rows["direct"]["run_s"] == 300.0 and rows["direct"]["net_tx_run_mb"] == 200.0
    assert rows["orchestrated"]["orchestrator_in_mb"] == 2.0 and rows["orchestrated"]["orchestrator_out_mb"] == 8.0 and rows["orchestrated"]["image_gb"] == 1.47
    assert rows["async"]["transfer_mb"] == 12.0 and rows["async"]["async_idle_s"] == 3.0 and rows["direct"]["orchestrator_in_mb"] is None
    assert rows["direct"]["validation_loss"] == 0.5 and rows["direct"]["rounds"] == 2 and rows["direct"]["rtt_ms"] == 40.0
    assert rows["direct"]["gpu_avg_pct"] == 40.0 and rows["direct"]["gpu_energy_wh"] == 3.0 and rows["direct"]["commit"] == "abcdef12"


def test_write_makes_every_format_and_the_page_has_a_bar_for_each_run(tmp_path):
    make_run(tmp_path, "direct", "t1", "direct", 300.0, 100.0)
    make_run(tmp_path, "async", "t1", "async", 500.0, 50.0, transfers=12_000_000)
    rows = [report.extract(d) for d in report.find_runs([tmp_path])]
    files = report.write(rows, tmp_path / "out")
    assert [f.name for f in files] == ["comparison.csv", "comparison.json", "comparison.md", "comparison.html"] and all(f.exists() for f in files)
    assert len(list(csv.DictReader((tmp_path / "out" / "comparison.csv").open()))) == 2
    page = (tmp_path / "out" / "comparison.html").read_text()
    assert page.count('class="bar"') >= 4 and "<script" not in page and "Where the time went" in page


def test_a_directory_without_metrics_is_skipped(tmp_path):
    (tmp_path / "x" / "t").mkdir(parents=True)
    assert report.find_runs([tmp_path]) == [] and report.extract(tmp_path / "x" / "t") is None
