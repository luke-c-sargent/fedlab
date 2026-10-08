import boto3
import pytest
from moto import mock_aws

from fedlab import ssh
from fedlab.providers.aws import AwsProvider


@pytest.fixture
def aws(cfg, monkeypatch):
    with mock_aws():
        prov = AwsProvider(cfg, ssh.ensure_keypair(cfg))
        monkeypatch.setattr(prov, "_ami", lambda ec2, gpu=False: ec2.describe_images()["Images"][0]["ImageId"])
        yield prov


def _node(cfg, i=0):
    return cfg.resolved_nodes()[i]


def test_lifecycle_and_destroy(aws, cfg):
    node = _node(cfg)
    ec2 = boto3.client("ec2", region_name=node.location)

    aws.up(node)
    aws.up(node)  # idempotent
    info = aws.describe(node)
    assert info.state == "running" and info.public_ip
    reservations = ec2.describe_instances()["Reservations"]
    assert len(reservations) == 1
    inst = reservations[0]["Instances"][0]
    assert inst["InstanceType"] == "r6i.large"
    vol = ec2.describe_volumes()["Volumes"][0]
    assert vol["Size"] == 200 and vol["VolumeType"] == "gp3"
    sg = aws._find_sg(ec2, "server")
    ports = sorted(p["FromPort"] for p in sg["IpPermissions"])
    assert ports == [22, 9092]

    aws.stop(node)
    assert aws.describe(node).state == "stopped"
    aws.start(node)
    assert aws.describe(node).state == "running"

    dry = aws.destroy([node], dry_run=True)
    assert any("terminate" in a for a in dry) and aws.describe(node).state == "running"

    aws.destroy([node])
    assert aws.describe(node).state == "absent"
    assert aws._find_sg(ec2, "server") is None
    assert ec2.describe_key_pairs()["KeyPairs"] == []
    assert aws.destroy([node]) == []  # idempotent


def test_destroy_ignores_untagged_resources(aws, cfg):
    node = _node(cfg)
    ec2 = boto3.client("ec2", region_name=node.location)
    other = ec2.run_instances(ImageId=aws._ami(ec2), MinCount=1, MaxCount=1)["Instances"][0]["InstanceId"]
    aws.up(node)
    aws.destroy([node])
    states = {i["InstanceId"]: i["State"]["Name"] for r in ec2.describe_instances()["Reservations"] for i in r["Instances"]}
    assert states[other] == "running"


def test_gpu_nodes_use_the_nvidia_driver_ami(cfg):
    prov = AwsProvider(cfg, ssh.ensure_keypair(cfg))
    ec2 = type("E", (), {})()
    seen = {}
    ec2.describe_images = lambda **kw: seen.update(kw) or {"Images": [{"ImageId": "ami-1", "CreationDate": "x"}]}
    assert prov._ami(ec2, gpu=True) == "ami-1" and seen["Owners"] == ["amazon"]
    assert "Nvidia Driver" in seen["Filters"][0]["Values"][0]
    prov._ami(ec2)
    assert seen["Owners"] == ["099720109477"]


def test_clients_only_open_ssh(aws, cfg):
    node = _node(cfg, 1)  # client
    ec2 = boto3.client("ec2", region_name=node.location)
    aws.up(node)
    sg = aws._find_sg(ec2, "client")
    assert [p["FromPort"] for p in sg["IpPermissions"]] == [22]
    assert aws._find_sg(ec2, "server") is None
    assert any("(client)" in a for a in aws.destroy([node], dry_run=True))
    aws.destroy([node])
    assert aws._find_sg(ec2, "client") is None


def _capacity_error():
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "InsufficientInstanceCapacity", "Message": "full"}}, "RunInstances")


class _FakeEc2:
    """run_instances fails unless the request names a subnet in `good_zone`."""

    def __init__(self, good_zone):
        self.good_zone, self.calls = good_zone, []

    def run_instances(self, **kw):
        self.calls.append(kw.get("SubnetId"))
        if kw.get("SubnetId") == f"subnet-{self.good_zone}":
            return {"Instances": [{"InstanceId": "i-1"}]}
        raise _capacity_error()

    def describe_subnets(self, **kw):
        return {"Subnets": [{"SubnetId": f"subnet-{z}", "AvailabilityZone": z} for z in ("zone-c", "zone-b")]}


def test_launch_tries_each_zone_after_a_capacity_error():
    ec2 = _FakeEc2("zone-c")
    assert AwsProvider._launch(ec2, InstanceType="g4dn.4xlarge")["Instances"][0]["InstanceId"] == "i-1"
    assert ec2.calls == [None, "subnet-zone-b", "subnet-zone-c"]  # AWS's own pick, then each zone in order


def test_launch_explains_when_every_zone_is_full():
    with pytest.raises(RuntimeError, match=r"zones tried: zone-b, zone-c.*another region or instance type"):
        AwsProvider._launch(_FakeEc2("none"), InstanceType="g4dn.4xlarge")


def test_launch_does_not_swallow_other_errors():
    from botocore.exceptions import ClientError

    class Denied(_FakeEc2):
        def run_instances(self, **kw):
            raise ClientError({"Error": {"Code": "UnauthorizedOperation", "Message": "no"}}, "RunInstances")

    with pytest.raises(ClientError, match="UnauthorizedOperation"):
        AwsProvider._launch(Denied("x"), InstanceType="t")
