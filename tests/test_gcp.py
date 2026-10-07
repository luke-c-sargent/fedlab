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
    assert list(fw.allowed[0].ports) == ["22", "9091", "9092", "9093"]
    assert list(fw.source_ranges) == ["0.0.0.0/0"]
    inst = ic.return_value.insert.call_args.kwargs["instance_resource"]
    assert inst.name == "fedlearn-gcp-us-central1-a"
    assert inst.machine_type.endswith("n2-standard-4")
    assert inst.disks[0].initialize_params.disk_size_gb == 200
    assert inst.labels["fedlab-run"] == "fedlearn"
    keys = {i.key: i.value for i in inst.metadata.items}
    assert keys["ssh-keys"].startswith("ubuntu:ssh-ed25519 ") and "#cloud-config" in keys["user-data"]


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
