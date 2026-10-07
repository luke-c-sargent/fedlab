from __future__ import annotations

from datetime import datetime

from google.api_core.exceptions import NotFound
from google.cloud import compute_v1

from .. import cloud_init
from ..config import TAG_KEY, Node
from .base import CheckResult, NodeInfo, Provider

IMAGE = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
_STATE = {
    "PROVISIONING": "pending",
    "STAGING": "pending",
    "RUNNING": "running",
    "STOPPING": "stopping",
    "SUSPENDING": "stopping",
    "TERMINATED": "stopped",
    "SUSPENDED": "stopped",
}


class GcpProvider(Provider):
    # ---- helpers -------------------------------------------------------
    @property
    def project(self) -> str:
        if not self.cfg.gcp_project:
            raise RuntimeError("GCP project not set: export GCP_PROJECT or set gcp_project in config")
        return self.cfg.gcp_project

    @property
    def _fw_name(self) -> str:
        return f"{self.cfg.run_name}-allow"

    @property
    def _net_tag(self) -> str:
        return self.cfg.run_name

    def _get(self, node: Node):
        try:
            return compute_v1.InstancesClient().get(
                project=self.project, zone=node.location, instance=node.name
            )
        except NotFound:
            return None

    def _ensure_firewall(self) -> None:
        fw = compute_v1.FirewallsClient()
        rule = compute_v1.Firewall(
            name=self._fw_name,
            network="global/networks/default",
            direction="INGRESS",
            source_ranges=["0.0.0.0/0"],
            target_tags=[self._net_tag],
            allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=[str(p) for p in self.cfg.open_ports])],
            description=f"fedlab {self.cfg.run_name}",
        )
        try:
            fw.get(project=self.project, firewall=self._fw_name)
            fw.update(project=self.project, firewall=self._fw_name, firewall_resource=rule).result()
        except NotFound:
            fw.insert(project=self.project, firewall_resource=rule).result()

    # ---- Provider API --------------------------------------------------
    def up(self, node: Node) -> None:
        if self._get(node):
            self.start(node)
            return
        self._ensure_firewall()
        labels = {TAG_KEY: self.cfg.run_name}
        inst = compute_v1.Instance(
            name=node.name,
            machine_type=f"zones/{node.location}/machineTypes/{node.machine_type}",
            labels=labels,
            tags=compute_v1.Tags(items=[self._net_tag]),
            disks=[
                compute_v1.AttachedDisk(
                    boot=True,
                    auto_delete=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image=IMAGE,
                        disk_size_gb=node.disk_gb,
                        disk_type=f"zones/{node.location}/diskTypes/pd-balanced",
                        labels=labels,
                    ),
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    network="global/networks/default",
                    access_configs=[
                        compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
                    ],
                )
            ],
            metadata=compute_v1.Metadata(
                items=[
                    compute_v1.Items(
                        key="ssh-keys", value=f"{self.cfg.ssh_user}:{self.key.public_key} fedlab"
                    ),
                    compute_v1.Items(key="user-data", value=cloud_init.render(self.cfg.ssh_user)),
                ]
            ),
        )
        compute_v1.InstancesClient().insert(
            project=self.project, zone=node.location, instance_resource=inst
        ).result()

    def start(self, node: Node) -> None:
        inst = self._get(node)
        if not inst:
            raise RuntimeError(f"{node.name} does not exist; run `fedlab up`")
        if _STATE.get(inst.status) == "stopped":
            compute_v1.InstancesClient().start(
                project=self.project, zone=node.location, instance=node.name
            ).result()

    def stop(self, node: Node) -> None:
        inst = self._get(node)
        if inst and _STATE.get(inst.status) in ("pending", "running"):
            compute_v1.InstancesClient().stop(
                project=self.project, zone=node.location, instance=node.name
            ).result()

    def describe(self, node: Node) -> NodeInfo:
        inst = self._get(node)
        if not inst:
            return NodeInfo(node.name, "gcp", node.location, "absent")
        state = _STATE.get(inst.status, inst.status.lower())
        ip = next(
            (ac.nat_i_p for ni in inst.network_interfaces for ac in ni.access_configs if ac.nat_i_p),
            None,
        )
        since = None
        if state == "running":
            ts = inst.last_start_timestamp or inst.creation_timestamp
            since = datetime.fromisoformat(ts) if ts else None
        return NodeInfo(node.name, "gcp", node.location, state, public_ip=ip, running_since=since)

    def destroy(self, nodes: list[Node], dry_run: bool = False) -> list[str]:
        actions: list[str] = []
        flt = f"labels.{TAG_KEY} = {self.cfg.run_name}"
        instances = compute_v1.InstancesClient()
        disks = compute_v1.DisksClient()
        for zone in sorted({n.location for n in nodes}):
            ops = []
            for i in instances.list(project=self.project, zone=zone, filter=flt):
                actions.append(f"[gcp {zone}] delete instance {i.name}")
                if not dry_run:
                    ops.append(
                        instances.delete(project=self.project, zone=zone, instance=i.name)
                    )
            for op in ops:
                op.result()
            for d in disks.list(project=self.project, zone=zone, filter=flt):
                actions.append(f"[gcp {zone}] delete leftover disk {d.name}")
                if not dry_run:
                    try:
                        disks.delete(project=self.project, zone=zone, disk=d.name).result()
                    except NotFound:
                        pass
        fw = compute_v1.FirewallsClient()
        try:
            fw.get(project=self.project, firewall=self._fw_name)
            actions.append(f"[gcp] delete firewall rule {self._fw_name}")
            if not dry_run:
                fw.delete(project=self.project, firewall=self._fw_name).result()
        except NotFound:
            pass
        return actions

    def check(self, nodes: list[Node]) -> list[CheckResult]:
        import google.auth

        results: list[CheckResult] = []

        def run(name: str, fn) -> bool:
            try:
                results.append(CheckResult(name, True, fn() or ""))
                return True
            except Exception as e:
                results.append(CheckResult(name, False, f"{type(e).__name__}: {e}"))
                return False

        def creds() -> str:
            _, proj = google.auth.default()
            return f"application default credentials (adc project: {proj or 'none'})"

        if not run("gcp credentials", creds) or not run("gcp project", lambda: self.project):
            return results

        run("gcp: compute API / list firewalls", lambda: next(iter(compute_v1.FirewallsClient().list(project=self.project)), None) and "ok")
        run(
            "gcp: Ubuntu 22.04 image",
            lambda: compute_v1.ImagesClient().get_from_family(project="ubuntu-os-cloud", family="ubuntu-2204-lts").name,
        )
        for n in nodes:
            run(
                f"gcp {n.location}: {n.name} machine type",
                lambda n=n: compute_v1.MachineTypesClient().get(
                    project=self.project, zone=n.location, machine_type=n.machine_type
                ).name,
            )
            run(
                f"gcp {n.location}: list instances",
                lambda n=n: next(iter(compute_v1.InstancesClient().list(project=self.project, zone=n.location)), None) and "ok",
            )
        return results
