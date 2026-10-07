from __future__ import annotations

import time

import boto3
from botocore.exceptions import ClientError

from .. import cloud_init
from ..config import TAG_KEY, Node
from .base import CheckResult, NodeInfo, Provider

LIVE_STATES = ["pending", "running", "stopping", "stopped"]


class AwsProvider(Provider):
    # ---- helpers -------------------------------------------------------
    def _ec2(self, region: str):
        return boto3.client("ec2", region_name=region)

    @property
    def _tag_filter(self) -> dict:
        return {"Name": f"tag:{TAG_KEY}", "Values": [self.cfg.run_name]}

    def _tags(self, resource_type: str, name: str) -> dict:
        return {
            "ResourceType": resource_type,
            "Tags": [{"Key": TAG_KEY, "Value": self.cfg.run_name}, {"Key": "Name", "Value": name}],
        }

    def _find(self, ec2, node: Node, states=LIVE_STATES) -> dict | None:
        res = ec2.describe_instances(
            Filters=[
                self._tag_filter,
                {"Name": "tag:Name", "Values": [node.name]},
                {"Name": "instance-state-name", "Values": states},
            ]
        )
        found = [i for r in res["Reservations"] for i in r["Instances"]]
        return found[0] if found else None

    def _ami(self, ec2, gpu: bool = False) -> str:
        if gpu:  # Ubuntu 22.04 with NVIDIA driver + CUDA preinstalled
            owner, pattern = "amazon", "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
        else:
            owner, pattern = "099720109477", "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"  # Canonical
        imgs = ec2.describe_images(
            Owners=[owner],
            Filters=[{"Name": "name", "Values": [pattern]}, {"Name": "state", "Values": ["available"]}],
        )["Images"]
        if not imgs:
            raise RuntimeError(f"no AMI matching {pattern!r}")
        return max(imgs, key=lambda i: i["CreationDate"])["ImageId"]

    def _sg_name(self, role: str) -> str:
        return f"{self.cfg.run_name}-{role}-sg"

    @property
    def _key_name(self) -> str:
        return f"{self.cfg.run_name}-key"

    def _find_sg(self, ec2, role: str) -> dict | None:
        sgs = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [self._sg_name(role)]}, self._tag_filter]
        )["SecurityGroups"]
        return sgs[0] if sgs else None

    def _ensure_sg(self, ec2, role: str) -> str:
        sg = self._find_sg(ec2, role)
        if sg:
            sg_id = sg["GroupId"]
        else:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
            if not vpcs:
                raise RuntimeError("no default VPC in this region")
            sg_id = ec2.create_security_group(
                GroupName=self._sg_name(role),
                Description=f"fedlab {self.cfg.run_name} {role}",
                VpcId=vpcs[0]["VpcId"],
                TagSpecifications=[self._tags("security-group", self._sg_name(role))],
            )["GroupId"]
        for port in self.cfg.ports_for(role):
            try:
                ec2.authorize_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[
                        {
                            "IpProtocol": "tcp",
                            "FromPort": port,
                            "ToPort": port,
                            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "fedlab"}],
                        }
                    ],
                )
            except ClientError as e:
                if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                    raise
        return sg_id

    def _ensure_key(self, ec2) -> None:
        try:
            ec2.import_key_pair(
                KeyName=self._key_name,
                PublicKeyMaterial=self.key.public_key.encode(),
                TagSpecifications=[self._tags("key-pair", self._key_name)],
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidKeyPair.Duplicate":
                raise

    # ---- Provider API --------------------------------------------------
    def up(self, node: Node) -> None:
        ec2 = self._ec2(node.location)
        if self._find(ec2, node):
            self.start(node)
            return
        sg_id = self._ensure_sg(ec2, node.role)
        self._ensure_key(ec2)
        res = ec2.run_instances(
            ImageId=self._ami(ec2, node.gpu),
            InstanceType=node.machine_type,
            MinCount=1,
            MaxCount=1,
            KeyName=self._key_name,
            SecurityGroupIds=[sg_id],
            UserData=cloud_init.render(self.cfg.ssh_user),
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": node.disk_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
                }
            ],
            MetadataOptions={"HttpTokens": "required"},
            TagSpecifications=[self._tags("instance", node.name), self._tags("volume", node.name)],
        )
        ec2.get_waiter("instance_running").wait(InstanceIds=[res["Instances"][0]["InstanceId"]])

    def start(self, node: Node) -> None:
        ec2 = self._ec2(node.location)
        inst = self._find(ec2, node)
        if not inst:
            raise RuntimeError(f"{node.name} does not exist; run `fedlab up`")
        iid = inst["InstanceId"]
        if inst["State"]["Name"] in ("stopped", "stopping"):
            if inst["State"]["Name"] == "stopping":
                ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
            ec2.start_instances(InstanceIds=[iid])
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])

    def stop(self, node: Node) -> None:
        ec2 = self._ec2(node.location)
        inst = self._find(ec2, node)
        if not inst:
            return
        iid = inst["InstanceId"]
        if inst["State"]["Name"] in ("pending", "running"):
            ec2.stop_instances(InstanceIds=[iid])
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])

    def describe(self, node: Node) -> NodeInfo:
        inst = self._find(self._ec2(node.location), node)
        if not inst:
            return NodeInfo(node.name, "aws", node.location, "absent")
        state = inst["State"]["Name"]
        return NodeInfo(
            node.name,
            "aws",
            node.location,
            state,
            public_ip=inst.get("PublicIpAddress"),
            running_since=inst["LaunchTime"] if state == "running" else None,
        )

    def destroy(self, nodes: list[Node], dry_run: bool = False) -> list[str]:
        actions: list[str] = []
        for region in sorted({n.location for n in nodes}):
            ec2 = self._ec2(region)
            res = ec2.describe_instances(
                Filters=[self._tag_filter, {"Name": "instance-state-name", "Values": LIVE_STATES}]
            )
            ids = [i["InstanceId"] for r in res["Reservations"] for i in r["Instances"]]
            if ids:
                actions.append(f"[aws {region}] terminate instances {ids}")
                if not dry_run:
                    ec2.terminate_instances(InstanceIds=ids)
                    ec2.get_waiter("instance_terminated").wait(InstanceIds=ids)

            for role in ("server", "client"):
                sg = self._find_sg(ec2, role)
                if sg:
                    actions.append(f"[aws {region}] delete security group {sg['GroupId']} ({role})")
                    if not dry_run:
                        self._delete_sg(ec2, sg["GroupId"])

            keys = ec2.describe_key_pairs(Filters=[self._tag_filter])["KeyPairs"]
            for k in keys:
                actions.append(f"[aws {region}] delete key pair {k['KeyName']}")
                if not dry_run:
                    ec2.delete_key_pair(KeyName=k["KeyName"])

            vols = ec2.describe_volumes(Filters=[self._tag_filter])["Volumes"]
            for v in vols:
                actions.append(f"[aws {region}] delete leftover volume {v['VolumeId']}")
                if not dry_run:
                    try:
                        ec2.delete_volume(VolumeId=v["VolumeId"])
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "InvalidVolume.NotFound":
                            raise
        return actions

    def check(self, nodes: list[Node]) -> list[CheckResult]:
        results: list[CheckResult] = []

        def run(name: str, fn) -> None:
            try:
                results.append(CheckResult(name, True, fn() or ""))
            except Exception as e:
                msg = e.response["Error"]["Code"] if isinstance(e, ClientError) else f"{type(e).__name__}: {e}"
                results.append(CheckResult(name, False, msg))

        def dry(call, **kw) -> str:
            """EC2 DryRun: DryRunOperation means 'would have been allowed'."""
            try:
                call(DryRun=True, **kw)
            except ClientError as e:
                if e.response["Error"]["Code"] != "DryRunOperation":
                    raise
            return "permitted (dry run)"

        def identity() -> str:
            ident = boto3.client("sts").get_caller_identity()
            return ident["Arn"]

        run("aws credentials", identity)
        if not results[0].ok:
            return results

        for region in sorted({n.location for n in nodes}):
            ec2 = self._ec2(region)
            tag = f"aws {region}"

            def vpc() -> str:
                vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
                if not vpcs:
                    raise RuntimeError("no default VPC in this region")
                return vpcs[0]["VpcId"]

            run(f"{tag}: default VPC", vpc)
            for n in (n for n in nodes if n.location == region):

                def offered(n=n) -> str:
                    r = ec2.describe_instance_type_offerings(
                        LocationType="region", Filters=[{"Name": "instance-type", "Values": [n.machine_type]}]
                    )
                    if not r["InstanceTypeOfferings"]:
                        raise RuntimeError(f"{n.machine_type} not offered in {region}")
                    return n.machine_type

                def can_launch(n=n) -> str:
                    # Also surfaces vCPU quota problems (VcpuLimitExceeded), common for G/VT GPU types.
                    return dry(
                        ec2.run_instances,
                        ImageId=self._ami(ec2, n.gpu),
                        InstanceType=n.machine_type,
                        MinCount=1,
                        MaxCount=1,
                    )

                run(f"{tag}: {n.name} AMI", lambda n=n: self._ami(ec2, n.gpu))
                run(f"{tag}: {n.name} machine type", offered)
                run(f"{tag}: {n.name} launch permission + quota", can_launch)
            run(
                f"{tag}: create security group",
                lambda: dry(ec2.create_security_group, GroupName=self._sg_name("check"), Description="fedlab check", VpcId=vpc()),
            )
            run(
                f"{tag}: import key pair",
                lambda: dry(ec2.import_key_pair, KeyName=self._key_name, PublicKeyMaterial=self.key.public_key.encode()),
            )
        return results

    @staticmethod
    def _delete_sg(ec2, sg_id: str) -> None:
        # ENIs from just-terminated instances can linger briefly.
        for attempt in range(12):
            try:
                ec2.delete_security_group(GroupId=sg_id)
                return
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "InvalidGroup.NotFound":
                    return
                if code != "DependencyViolation" or attempt == 11:
                    raise
                time.sleep(5)
