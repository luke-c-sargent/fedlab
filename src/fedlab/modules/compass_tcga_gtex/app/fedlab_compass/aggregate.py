"""Server-side scaler aggregation for clients that prepared their own data.

This is `02_aggregate_scaler.py` working from what clients send instead of from prepared directories on one
disk. A client reports its `manifest.json` and `scaler_stats.npz`, byte for byte. The server cannot re-hash a
client's expression arrays (it never sees them), so it checks what it can: the manifest's own identity, the
configuration, the split seed, and that the stats file is the one the manifest attests. It produces the same
`global_scaler.npz` and `federated_manifest.json` contents as the original script, so the clients' existing
metadata check (`validate_federated_manifest_metadata`) accepts them unchanged.

Two deliberate differences from the original: the scaler file is written deterministically (no timestamp in
the zip), so the same inputs always give the same scaler hash and prepared-data id; and `scaler.path` in the
manifest is whatever the caller says, because the server's path is not the clients'.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CLIENT_IDS = ("TCGA", "GTEx")


@dataclass(frozen=True)
class ClientReport:
    manifest_bytes: bytes  # the client's manifest.json, exactly as written (its hash goes into the federated manifest)
    scaler_stats_bytes: bytes  # the client's scaler_stats.npz, exactly as written


@dataclass(frozen=True)
class Aggregate:
    scaler_bytes: bytes
    manifest: dict
    manifest_bytes: bytes


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def npz_bytes(**arrays: np.ndarray) -> bytes:
    """`np.savez` without a timestamp, so equal arrays always give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            with archive.open(member, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asanyarray(array), allow_pickle=False)
    return buffer.getvalue()


def json_bytes(value: Any) -> bytes:
    """The format the original scripts write (`atomic_write_json`)."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _check_client_manifest(fc, manifest: Mapping[str, Any], client_id: str, config_hash: str, expected_source: str | None) -> None:
    """`validate_client_prepared_manifest` minus the artifact re-hash, which needs the client's files."""
    if manifest.get("schema_version") != fc.CLIENT_PREPARED_MANIFEST_SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError(f"{client_id}: prepared manifest is incomplete or obsolete")
    if manifest.get("client_id") != client_id:
        raise ValueError(f"{client_id}: manifest belongs to {manifest.get('client_id')!r}")
    if manifest.get("config_sha256") != config_hash:
        raise ValueError(f"{client_id}: prepared with a different configuration")
    if expected_source is not None and manifest.get("source_sha256") != expected_source:
        raise ValueError(f"{client_id}: prepared from a file other than the assigned one")
    if not isinstance(manifest.get("artifact_hashes"), dict):
        raise ValueError(f"{client_id}: manifest has no artifact hashes")
    if manifest.get("client_dataset_id") != fc.stable_json_hash(fc.client_prepared_identity(manifest)):
        raise ValueError(f"{client_id}: dataset identity does not match the manifest contents")


def aggregate(
    fc, reports: Mapping[str, ClientReport], config_path: Path, scaler_path: str,
    expected_source_sha256: Mapping[str, str] | None = None,
) -> Aggregate:
    """`fc` is the `federated_common` module. `expected_source_sha256` (client id -> hash) is what was assigned."""
    if set(reports) != set(CLIENT_IDS):
        raise ValueError(f"expected reports from {sorted(CLIENT_IDS)}, got {sorted(reports)}")
    expected_source_sha256 = expected_source_sha256 or {}
    config_hash = _sha256(Path(config_path).read_bytes())
    manifests: dict[str, dict] = {}
    manifest_hashes: dict[str, str] = {}
    statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gene_hashes: set[str] = set()
    for client_id in CLIENT_IDS:
        report = reports[client_id]
        manifest = json.loads(report.manifest_bytes)
        _check_client_manifest(fc, manifest, client_id, config_hash, expected_source_sha256.get(client_id))
        if manifest["artifact_hashes"].get("scaler_stats_sha256") != _sha256(report.scaler_stats_bytes):
            raise ValueError(f"{client_id}: the scaler statistics are not the ones its manifest attests")
        with np.load(io.BytesIO(report.scaler_stats_bytes), allow_pickle=False) as stats:
            minimum = np.asarray(stats["feature_min"], dtype=np.float64)
            maximum = np.asarray(stats["feature_max"], dtype=np.float64)
        if minimum.shape != maximum.shape or minimum.ndim != 1:
            raise ValueError(f"Invalid {client_id} scaler statistics")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(maximum < minimum):
            raise ValueError(f"Invalid extrema in {client_id} scaler statistics")
        manifests[client_id] = manifest
        manifest_hashes[client_id] = _sha256(report.manifest_bytes)
        statistics[client_id] = (minimum, maximum)
        gene_hashes.add(str(manifest["artifact_hashes"].get("gene_order_sha256")))

    if len(gene_hashes) != 1:
        raise ValueError("TCGA and GTEx client gene orders differ")
    split_seeds = {int(manifest["split_seed"]) for manifest in manifests.values()}
    if len(split_seeds) != 1:
        raise ValueError("TCGA and GTEx client caches use different split seeds")
    tcga_min, tcga_max = statistics["TCGA"]
    gtex_min, gtex_max = statistics["GTEx"]
    if tcga_min.shape != gtex_min.shape:
        raise ValueError("TCGA and GTEx scaler feature counts differ")
    data_min = np.minimum(tcga_min, gtex_min)
    data_max = np.maximum(tcga_max, gtex_max)
    data_range = data_max - data_min
    safe_range = data_range.copy()
    safe_range[safe_range == 0.0] = 1.0
    scale = 1.0 / safe_range
    offset = -data_min * scale
    sample_count = sum(int(manifest["samples"]) for manifest in manifests.values())
    training_sample_count = sum(int(manifest["train_samples"]) for manifest in manifests.values())

    scaler_bytes = npz_bytes(
        data_min=data_min, data_max=data_max, data_range=data_range, scale=scale, offset=offset,
        n_samples_seen=np.asarray([training_sample_count], dtype=np.int64),
    )
    scaler_hash = _sha256(scaler_bytes)
    output = {
        "schema_version": fc.FEDERATED_PREPARED_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "source_input_sha256": {name: manifests[name]["source_sha256"] for name in sorted(manifests)},
        "preparation_config_sha256": config_hash,
        "split_seed": next(iter(split_seeds)),
        "clients": {
            name: {
                "client_dataset_id": manifests[name]["client_dataset_id"],
                "manifest_sha256": manifest_hashes[name],
                "artifact_hashes": manifests[name]["artifact_hashes"],
                "samples": manifests[name]["samples"],
                "source_samples": manifests[name].get("source_samples", manifests[name]["samples"]),
                "split_seed": manifests[name]["split_seed"],
                "train_samples": manifests[name]["train_samples"],
                "validation_samples": manifests[name]["validation_samples"],
                "groups": manifests[name]["groups"],
                "contexts": manifests[name]["contexts"],
                "dose_control": manifests[name].get("dose_control"),
            }
            for name in sorted(manifests)
        },
        "genes": int(data_min.shape[0]),
        "gene_order_sha256": next(iter(gene_hashes)),
        "total_samples": sample_count,
        "total_training_samples": training_sample_count,
        "gtex_dose_control": manifests["GTEx"].get("dose_control"),
        "scaler": {
            "method": "global_minmax_from_client_extrema",
            "input_transform": "log2p1",
            "fit_scope": "union_of_client_training_partitions_only",
            "shared_statistics": ["per_gene_minimum", "per_gene_maximum"],
            "raw_sample_rows_shared": False,
            "path": str(scaler_path),
            "sha256": scaler_hash,
        },
    }
    output["prepared_artifact_hashes"] = {
        "global_scaler_sha256": scaler_hash,
        "client_manifest_sha256": {name: manifest_hashes[name] for name in sorted(manifest_hashes)},
        "clients": {name: manifests[name]["artifact_hashes"] for name in sorted(manifests)},
    }
    identity = fc.federated_prepared_identity(output)
    prepared_data_id = fc.stable_json_hash(identity)
    output["prepared_data_identity"] = identity
    output["prepared_data_id"] = prepared_data_id
    output["federated_dataset_id"] = prepared_data_id
    output["manifest_payload_sha256"] = fc.stable_json_hash(output)
    return Aggregate(scaler_bytes, output, json_bytes(output))
