"""Configuration: YAML file + .env + FEDLAB_* env vars (env wins over the file)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TAG_KEY = "fedlab-run"
_SLUG = re.compile(r"^[a-z][a-z0-9-]{0,30}$")


class NodeSpec(BaseModel):
    provider: Literal["aws", "gcp"]
    location: str  # AWS region or GCP zone
    role: Literal["server", "client"] = "client"
    site: str | None = None  # data site a client holds, e.g. "tcga"; interpreted by the module
    machine_type: str | None = None
    disk_gb: int | None = None
    gpu: bool = False  # AWS: use the NVIDIA-driver AMI. GCP: install the driver on first boot.
    accelerator: str | None = None  # GCP only: GPU to attach to an N1 machine, e.g. nvidia-tesla-t4


@dataclass(frozen=True)
class Node:
    """A fully-resolved node."""

    name: str
    provider: str
    location: str
    role: str
    machine_type: str
    disk_gb: int
    gpu: bool = False
    accelerator: str | None = None
    site: str | None = None


def _default_nodes() -> list[NodeSpec]:
    return [
        # server: 2 vCPU / 16 GB
        NodeSpec(role="server", provider="aws", location="eu-west-1", machine_type="r6i.large"),
        # GPU clients: T4, 52-64 GB RAM
        NodeSpec(role="client", site="tcga", provider="aws", location="us-east-1", machine_type="g4dn.4xlarge", gpu=True),  # 16 vCPU / 64 GB, T4
        NodeSpec(
            role="client", site="gtex", provider="gcp", location="us-central1-a", machine_type="n1-highmem-8",  # 8 vCPU / 52 GB
            gpu=True, accelerator="nvidia-tesla-t4",
        ),
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FEDLAB_", env_file=".env", extra="ignore", populate_by_name=True
    )

    run_name: str = "fedlearn"
    name_prefix: str = "fedlearn"
    aws_machine_type: str = "m5.xlarge"
    gcp_machine_type: str = "n2-standard-4"
    disk_gb: int = 200
    ports: list[int] = [22, 9092]  # opened on the server only (9092 = Flower Fleet API)
    warn_after_days: float = 3
    ssh_user: str = "ubuntu"
    gcp_project: str | None = Field(
        default=None, validation_alias=AliasChoices("gcp_project", "FEDLAB_GCP_PROJECT", "GCP_PROJECT")
    )
    state_dir: Path = Path(".fedlab")
    ssh_config_path: Path = Path("~/.ssh/config.d/fedlab")
    nodes: list[NodeSpec] = Field(default_factory=_default_nodes)
    module: str | None = None  # federated-learning module to deploy; see `fedlab modules`
    module_options: dict[str, Any] = Field(default_factory=dict)  # passed to the module
    results_dir: Path = Path("results")

    @field_validator("run_name", "name_prefix")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("must be lowercase letters, digits, hyphens; start with a letter")
        return v

    @property
    def open_ports(self) -> list[int]:
        """Ports opened on the server (`ports`, plus SSH)."""
        return sorted({22, *self.ports})

    def ports_for(self, role: str) -> list[int]:
        """Clients only expose SSH; they connect out to the server."""
        return self.open_ports if role == "server" else [22]

    @property
    def key_path(self) -> Path:
        return self.state_dir / "id_ed25519"

    @property
    def nodes_file(self) -> Path:
        return self.state_dir / "nodes.json"

    def resolved_nodes(self) -> list[Node]:
        out = []
        for n in self.nodes:
            default_mt = self.aws_machine_type if n.provider == "aws" else self.gcp_machine_type
            out.append(
                Node(
                    name=f"{self.name_prefix}-{n.role}-{n.provider}-{n.location}",
                    provider=n.provider,
                    location=n.location,
                    role=n.role,
                    machine_type=n.machine_type or default_mt,
                    disk_gb=n.disk_gb or self.disk_gb,
                    gpu=n.gpu,
                    accelerator=n.accelerator,
                    site=n.site,
                )
            )
        names = [n.name for n in out]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate node names: {names}")
        servers = [n.name for n in out if n.role == "server"]
        if len(servers) != 1:
            raise ValueError(f"exactly one node must have role: server (found {servers})")
        return out


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings. `.env` is also exported to os.environ so boto3/google auth see it."""
    load_dotenv()
    if config_path is None and Path("config.yaml").exists():
        config_path = Path("config.yaml")
    if config_path is None:
        return Settings()

    class _FileSettings(Settings):
        model_config = {**Settings.model_config, "yaml_file": str(config_path)}

        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
            from pydantic_settings import YamlConfigSettingsSource

            return (
                init_settings,
                env_settings,
                dotenv_settings,
                YamlConfigSettingsSource(settings_cls),
                file_secret_settings,
            )

    return _FileSettings()
