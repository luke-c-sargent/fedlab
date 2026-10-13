"""What a site needs around its data in a file-based run: the code it imports, the shared files, the checks.

In the live setup fedlab uploads all of this over SSH and runs `verify` once. Here a site installs the code
itself (see `identity`), and the shared files (global scaler, federated manifest) arrive inside requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .identity import check_code_identity, code_identity
from .verify import verify_client

CONFIG = Path("centralized_test") / "config" / "paper_pretraining.json"
FEDERATED_CONFIG = Path("federated_test") / "config" / "federated_pretraining.json"


def load_fc(code_root: Path):
    """The original `federated_common` module, imported from the site's installed code."""
    for sub in (Path("federated_test") / "scripts", Path("centralized_test") / "scripts"):
        path = str(Path(code_root) / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import federated_common

    return federated_common


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_if_different(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


@contextmanager
def site_environment(code_root: Path, compass_source: Path | None, site_root: Path):
    """The environment variables the real backends read, set for the duration of the block."""
    wanted = {"FEDLAB_COMPASS_CODE": str(code_root), "FEDLAB_COMPASS_SITE_ROOT": str(site_root)}
    if compass_source is not None:
        wanted["COMPASS_SOURCE_DIR"] = str(compass_source)
    saved = {k: os.environ.get(k) for k in wanted}
    os.environ.update(wanted)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_identity(expected_json: str | None, code_root: Path, compass_source: Path | None) -> None:
    """Refuse to run on code other than the code the request was made for. No expectation, no check."""
    if not expected_json:
        return
    expected = json.loads(expected_json)
    actual = code_identity(code_root, compass_source if "compass_commit" in expected else None)
    check_code_identity(expected, actual)


def install_shared(config: dict, code_root: Path, site_root: Path, site: str, client_id: str, seed: int) -> dict:
    """Put the shared scaler and federated manifest where the backends read them, and verify them.

    The metadata check runs every time (it is cheap). Re-hashing the site's own prepared data is not, so that
    happens once per federated manifest, marked by a file next to it.
    """
    scaler, manifest = bytes(config["scaler_npz"]), bytes(config["federated_manifest"])
    scaler_path, manifest_path = site_root / "global_scaler.npz", site_root / "federated_manifest.json"
    client_dir = site_root / "clients" / site
    write_if_different(scaler_path, scaler)
    write_if_different(manifest_path, manifest)
    write_if_different(site_root / "gene_order.txt", (client_dir / "gene_order.txt").read_bytes())

    fc = load_fc(code_root)
    config_path = Path(code_root) / CONFIG
    split = fc.protocol_controls(fc.load_federated_config(Path(code_root) / FEDERATED_CONFIG), seed)["streams"]["split"]
    federated = fc.validate_federated_manifest_metadata(
        manifest_path, global_scaler=scaler_path, config_path=config_path, expected_split_seed=split
    )
    marker = site_root / f".data-verified-{hashlib.sha256(manifest).hexdigest()[:16]}"
    if not marker.exists():
        expected_genes = json.loads(config_path.read_text())["input"]["expected_genes"]
        verify_client(fc, config_path, split, federated, client_dir, client_id, expected_genes, _sha256_file)
        marker.write_text("ok\n")
    return federated
