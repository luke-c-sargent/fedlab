from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound

from fedlab import ssh
from fedlab.providers import gcp as gcpmod
from fedlab.providers.gcp import GcpProvider


def _prov(cfg):
    cfg.gcp_project = "proj"
    return GcpProvider(cfg, ssh.ensure_keypair(cfg))


def test_up_creates_firewall_and_instance_with_expected_shape(cfg):
    prov = _prov(cfg)
    node = cfg.resolved_nodes()[2]
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(
        gcpmod.compute_v1, "FirewallsClient"
    ) as fc:
        ic.return_value.get.side_effect = NotFound("x")
        fc.return_value.get.side_effect = NotFound("x")
        prov.up(node)
    fw = fc.return_value.insert.call_args.kwargs["firewall_resource"]
    assert fw.name == "fedlearn-client" and list(fw.target_tags) == ["fedlearn-client"]
    assert list(fw.allowed[0].ports) == ["22"]  # clients expose SSH only
    assert list(fw.source_ranges) == ["0.0.0.0/0"]
    inst = ic.return_value.insert.call_args.kwargs["instance_resource"]
    assert list(inst.tags.items) == ["fedlearn-client"]
    assert inst.name == "fedlearn-client-gcp-us-central1-a"
    assert inst.machine_type.endswith("n1-highmem-8")
    (acc,) = inst.guest_accelerators
    assert acc.accelerator_type.endswith("us-central1-a/acceleratorTypes/nvidia-tesla-t4") and acc.accelerator_count == 1
    assert inst.scheduling.on_host_maintenance == "TERMINATE"
    assert inst.disks[0].initialize_params.disk_size_gb == 200
    assert inst.labels["fedlab-run"] == "fedlearn"
    keys = {i.key: i.value for i in inst.metadata.items}
    assert keys["ssh-keys"].startswith("ubuntu:ssh-ed25519 ") and "#cloud-config" in keys["user-data"]
    assert "ubuntu-drivers install" in keys["user-data"]


def test_describe_maps_state_and_ip(cfg):
    prov = _prov(cfg)
    node = cfg.resolved_nodes()[2]
    inst = MagicMock(status="TERMINATED")
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic:
        ic.return_value.get.return_value = inst
        inst.network_interfaces = []
        info = prov.describe(node)
    assert info.state == "stopped" and info.public_ip is None


def test_destroy_dry_run_lists_by_label_using_request_objects(cfg):
    prov = _prov(cfg)
    node = cfg.resolved_nodes()[2]
    inst, disk = MagicMock(), MagicMock()
    inst.name, disk.name = "vm1", "disk1"
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(
        gcpmod.compute_v1, "DisksClient"
    ) as dc, patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        ic.return_value.list.return_value = [inst]
        dc.return_value.list.return_value = [disk]
        fc.return_value.get.side_effect = NotFound("x")
        actions = prov.destroy([node], dry_run=True)
    # list() has no flattened `filter` kwarg; it must go through a request object
    req = ic.return_value.list.call_args.kwargs["request"]
    assert req.filter == "labels.fedlab-run = fedlearn" and req.zone == node.location
    assert dc.return_value.list.call_args.kwargs["request"].filter == req.filter
    assert actions == ["[gcp us-central1-a] delete instance vm1", "[gcp us-central1-a] delete leftover disk disk1"]
    ic.return_value.delete.assert_not_called()


def test_server_firewall_opens_flower_ports(cfg):
    prov = _prov(cfg)
    with patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        fc.return_value.get.side_effect = NotFound("x")
        prov._ensure_firewall("server")
    fw = fc.return_value.insert.call_args.kwargs["firewall_resource"]
    assert fw.name == "fedlearn-server" and list(fw.allowed[0].ports) == ["22", "9092"]


def test_await_logs_while_an_operation_is_slow(monkeypatch):
    import concurrent.futures

    class Slow:
        calls = 0

        def result(self, timeout=None):
            Slow.calls += 1
            if Slow.calls < 3:
                raise concurrent.futures.TimeoutError()

    messages = []
    GcpProvider._await(Slow(), "deleting instance vm1", messages.append, "[gcp z]", beat=15)
    assert len(messages) == 2 and "still deleting instance vm1" in messages[0]


def test_up_refuses_to_reuse_a_vm_of_a_different_type(cfg):
    import dataclasses

    import pytest

    prov = _prov(cfg)
    node = dataclasses.replace(cfg.resolved_nodes()[2], machine_type="g2-standard-4")
    existing = MagicMock(machine_type="https://www.googleapis.com/compute/v1/projects/p/zones/z/machineTypes/n1-highmem-8")
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic:
        ic.return_value.get.return_value = existing
        with pytest.raises(RuntimeError, match=r"already exists as n1-highmem-8.*g2-standard-4.*fedlab destroy"):
            prov.up(node)


# ---- zone fallback: a node may name more zones to try when the first has no capacity --------------------------

import dataclasses  # noqa: E402

import pytest  # noqa: E402
from google.api_core.exceptions import Forbidden, ServiceUnavailable  # noqa: E402

STOCKOUT = "503 SERVICE UNAVAILABLE ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS: The zone does not have enough resources"
ZONES = ("us-central1-a", "us-central1-b", "us-central1-c")


def _node(cfg):
    return dataclasses.replace(cfg.resolved_nodes()[2], fallback_locations=ZONES[1:])


def _created_in(ic):
    return [c.kwargs["zone"] for c in ic.return_value.insert.call_args_list]


def test_a_stockout_moves_on_to_the_next_zone_and_builds_the_node_for_that_zone(cfg):
    prov, node, messages = _prov(cfg), _node(cfg), []
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        ic.return_value.get.side_effect = NotFound("x")
        fc.return_value.get.side_effect = NotFound("x")
        ic.return_value.insert.return_value.result.side_effect = [ServiceUnavailable(STOCKOUT), None]
        prov.up(node, log=messages.append)
    assert _created_in(ic) == ["us-central1-a", "us-central1-b"]
    inst = ic.return_value.insert.call_args.kwargs["instance_resource"]  # the second attempt, built for zone b
    assert "zones/us-central1-b/machineTypes/" in inst.machine_type
    assert inst.guest_accelerators[0].accelerator_type.startswith("zones/us-central1-b/acceleratorTypes/")
    assert "zones/us-central1-b/diskTypes/" in inst.disks[0].initialize_params.disk_type
    assert inst.name == node.name  # the name keeps its primary zone: it identifies the node, not where it landed
    assert messages == [
        f"{node.name}: no capacity for n1-highmem-8 in us-central1-a; trying us-central1-b",
        f"{node.name}: created in us-central1-b (no capacity in us-central1-a)",
    ]
    assert fc.return_value.insert.call_count == 1  # the firewall is created once, not per attempt


def test_when_every_zone_is_out_the_error_names_them_all(cfg):
    prov, node = _prov(cfg), _node(cfg)
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        ic.return_value.get.side_effect = NotFound("x")
        fc.return_value.get.side_effect = NotFound("x")
        ic.return_value.insert.return_value.result.side_effect = ServiceUnavailable(STOCKOUT)
        with pytest.raises(RuntimeError, match=r"no capacity for n1-highmem-8 in any of \['us-central1-a', 'us-central1-b', 'us-central1-c'\]"):
            prov.up(node)
    assert _created_in(ic) == list(ZONES)


def test_other_errors_do_not_fall_back(cfg):
    prov, node = _prov(cfg), _node(cfg)
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        ic.return_value.get.side_effect = NotFound("x")
        fc.return_value.get.side_effect = NotFound("x")
        ic.return_value.insert.return_value.result.side_effect = Forbidden("Quota 'NVIDIA_T4_GPUS' exceeded")
        with pytest.raises(Forbidden):
            prov.up(node)
    assert _created_in(ic) == ["us-central1-a"]  # a quota problem would follow the node to every zone


def test_without_fallback_zones_a_stockout_is_still_a_clear_error(cfg):
    prov, node = _prov(cfg), cfg.resolved_nodes()[2]
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(gcpmod.compute_v1, "FirewallsClient") as fc:
        ic.return_value.get.side_effect = NotFound("x")
        fc.return_value.get.side_effect = NotFound("x")
        ic.return_value.insert.return_value.result.side_effect = ServiceUnavailable(STOCKOUT)
        with pytest.raises(RuntimeError, match=r"in any of \['us-central1-a'\].*fallback_locations"):
            prov.up(node)


def _only_in(zone, status="RUNNING"):
    """An InstancesClient().get that finds the VM in `zone` only."""
    inst = MagicMock(status=status, network_interfaces=[], machine_type="zones/z/machineTypes/n1-highmem-8",
                     last_start_timestamp="2026-10-12T12:00:00+00:00")

    def get(project, zone, instance):  # noqa: ANN001
        if zone == _only_in.target:
            return inst
        raise NotFound("x")

    _only_in.target = zone
    return get


def test_describe_finds_a_vm_in_a_fallback_zone_and_reports_where_it_is(cfg):
    prov, node = _prov(cfg), _node(cfg)
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic:
        ic.return_value.get.side_effect = _only_in("us-central1-c")
        info = prov.describe(node)
    assert info.state == "running" and info.location == "us-central1-c"
    assert [c.kwargs["zone"] for c in ic.return_value.get.call_args_list] == list(ZONES)


def test_up_reuses_a_vm_that_landed_in_a_fallback_zone(cfg):
    prov, node = _prov(cfg), _node(cfg)
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic:
        ic.return_value.get.side_effect = _only_in("us-central1-b")
        prov.up(node)
    ic.return_value.insert.assert_not_called()


def test_start_and_stop_act_in_the_zone_the_vm_is_in(cfg):
    prov, node = _prov(cfg), _node(cfg)
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic:
        ic.return_value.get.side_effect = _only_in("us-central1-b", status="TERMINATED")
        prov.start(node)
        assert ic.return_value.start.call_args.kwargs["zone"] == "us-central1-b"
        ic.return_value.get.side_effect = _only_in("us-central1-c", status="RUNNING")
        prov.stop(node)
        assert ic.return_value.stop.call_args.kwargs["zone"] == "us-central1-c"


def test_destroy_looks_in_every_candidate_zone(cfg):
    prov, node = _prov(cfg), _node(cfg)
    stray = MagicMock()
    stray.name = "fedlearn-client-gcp-us-central1-a"
    with patch.object(gcpmod.compute_v1, "InstancesClient") as ic, patch.object(gcpmod.compute_v1, "DisksClient") as dc, patch.object(
        gcpmod.compute_v1, "FirewallsClient"
    ) as fc:
        ic.return_value.list.side_effect = lambda request: [stray] if request.zone == "us-central1-c" else []
        dc.return_value.list.return_value = []
        fc.return_value.get.side_effect = NotFound("x")
        actions = prov.destroy([node], dry_run=True)
    assert sorted(c.kwargs["request"].zone for c in ic.return_value.list.call_args_list) == list(ZONES)
    assert actions == ["[gcp us-central1-c] delete instance fedlearn-client-gcp-us-central1-a"]  # found where it landed


def test_check_verifies_every_candidate_zone(cfg):
    prov, node = _prov(cfg), _node(cfg)
    quota = MagicMock(metric="NVIDIA_T4_GPUS", limit=4, usage=0)
    with patch("google.auth.default", return_value=(None, "proj")), patch.object(gcpmod.compute_v1, "FirewallsClient"), patch.object(
        gcpmod.compute_v1, "ImagesClient"
    ), patch.object(gcpmod.compute_v1, "InstancesClient"), patch.object(gcpmod.compute_v1, "AcceleratorTypesClient"), patch.object(
        gcpmod.compute_v1, "RegionsClient"
    ) as rc, patch.object(gcpmod.compute_v1, "MachineTypesClient") as mc:
        rc.return_value.get.return_value.quotas = [quota]
        results = prov.check([node])
    assert all(r.ok for r in results)
    assert [c.kwargs["zone"] for c in mc.return_value.get.call_args_list] == list(ZONES)
    assert {r.name.split(":")[0] for r in results if "machine type" in r.name} == {f"gcp {z}" for z in ZONES}
    assert sum("list instances" in r.name for r in results) == 3  # the permission check still runs per zone
