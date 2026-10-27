import json

from fedlab import metrics
from fedlab.config import Node, Settings
from fedlab.modules import Deployment, Site
from fedlab.remote import LocalHost

CSV = """#meta,node1,4,16000,24000,NVIDIA A10G
epoch,cpu_pct,hot_core_pct,mem_used_mb,disk_read_b,disk_write_b,net_rx_b,net_tx_b,gpu_util_pct,gpu_mem_used_mb,gpu_power_w
100.0,10.0,20.0,1000,0,0,1000,1000,0,100,50
103.0,50.0,90.0,2000,3000000,6000000,4001000,7001000,80,12000,150
106.0,30.0,40.0,1500,6000000,6000000,10001000,7001000,60,12000,100
"""


def test_summary_totals_come_from_the_counters_and_rates_from_the_differences():
    meta, rows = metrics.parse(CSV)
    s = metrics.summarize(rows)
    assert meta["gpu"] == "NVIDIA A10G" and meta["vcpus"] == 4
    assert s["seconds"] == 6 and s["mem_peak_mb"] == 2000
    assert (s["disk_read_mb"], s["disk_write_mb"], s["net_rx_mb"], s["net_tx_mb"]) == (6.0, 6.0, 10.0, 7.0)
    assert s["net_rx_peak_mbps"] == 16.0  # 6 MB in 3 s
    assert s["gpu_peak_pct"] == 80 and s["gpu_energy_wh"] == round(100 * 6 / 3600, 2)


def test_a_window_selects_only_its_samples():
    _, rows = metrics.parse(CSV)
    assert metrics.summarize(rows, 103.0, 106.0)["net_rx_mb"] == 6.0
    assert metrics.summarize(rows, 200.0, 300.0) is None


def test_a_row_cut_short_is_ignored():
    _, rows = metrics.parse(CSV + "109.0,5.0,5.")
    assert len(rows) == 3


def test_the_sampler_runs_collects_and_summarizes(tmp_path):
    host = LocalHost("h", tmp_path / "h")
    d = Deployment(Settings(), Site(Node("n", "aws", "local", "server", "t", 10), host), [], tmp_path)
    d.state_dir.mkdir(exist_ok=True)
    metrics.INTERVAL, old = 0.2, metrics.INTERVAL
    try:
        metrics.start(d)
        metrics.mark(d, "run_start")
        import time

        time.sleep(1.5)
        metrics.mark(d, "run_end")
        summary = metrics.collect(d, tmp_path / "out")
    finally:
        metrics.INTERVAL = old
    assert json.loads((tmp_path / "out" / "metrics" / "summary.json").read_text())["hosts"].keys() == {"n"}
    assert summary["hosts"]["n"]["whole"]["samples"] >= 3 and summary["hosts"]["n"]["run"]["samples"] >= 2


def test_phases_rtt_run_info_and_egress_are_in_the_summary(tmp_path):
    host = LocalHost("h", tmp_path / "h")
    d = Deployment(Settings(), Site(Node("n", "aws", "local", "server", "t", 10), host), [Site(Node("c", "gcp", "local", "client", "t", 10, site="x"), LocalHost("c", tmp_path / "c"))], tmp_path / "state")
    metrics.reset(d.state_dir)
    with metrics.phase(d.state_dir, "stage"):
        pass
    import socket

    with socket.socket() as s:  # something to connect to
        s.bind(("127.0.0.1", 0))
        s.listen()
        rtt = metrics.probe_rtt(d, "127.0.0.1", s.getsockname()[1])
    assert rtt["c"]["min"] <= rtt["c"]["median"] <= rtt["c"]["max"]
    summary = metrics.collect(d, tmp_path / "out")
    assert "stage" in summary["phases"] and summary["rtt"][0]["target"].startswith("127.0.0.1:")
    assert summary["run_info"]["flwr"] and {n["name"] for n in summary["run_info"]["nodes"]} == {"n", "c"}
    assert set(summary["egress_by_provider_mb"]) == {"aws", "gcp"}
