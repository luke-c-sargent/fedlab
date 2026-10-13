"""Prepare one site's data on the site's own machine: the per-site half of `01_prepare_clients.py`.

The original script prepares both sites in one call. Its building block `prepare_client` is already per site,
so this calls it directly (loaded from the code directory the site has installed) and repeats only the parts of
the script's `main()` that apply to one site. Nothing about the preparation itself is reimplemented: the splits,
the hashes and the manifest come from the original function.

What it cannot do alone is the cross-site header comparison of the original; that is covered later, when the
server checks that both sites' gene-order hashes agree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .aggregate import ClientReport
from .contract import SITES
from .site import load_fc

CONFIG = Path("centralized_test") / "config" / "paper_pretraining.json"
SCRIPT = Path("federated_test") / "scripts" / "01_prepare_clients.py"


def load_prepare_script(code_root: Path):
    """The original script as a module (it is a script with a numeric name, so import it by path)."""
    load_fc(code_root)  # puts the script directories on sys.path: the script imports its siblings by name
    path = Path(code_root) / SCRIPT
    spec = importlib.util.spec_from_file_location("fedlab_prepare_clients", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its `main()` is guarded, so this only defines functions
    return module


def prepare_site(
    code_root: Path, site: str, source_path: Path, output_dir: Path, seed: int,
    chunk_rows: int = 128, gtex_donor_fraction: float = 1.0, gtex_dose_seed: int = 1729,
    training_protocol: str = "historical_v1",
) -> ClientReport:
    """Prepare `site` ("tcga" or "gtex") from its raw TSV into `output_dir`. Defaults are those of `01`."""
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {sorted(SITES)}")
    client_id = SITES[site][0]
    code_root, source_path, output_dir = Path(code_root), Path(source_path), Path(output_dir)
    script = load_prepare_script(code_root)
    config_path = code_root / CONFIG
    for path in (source_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = script.load_config(config_path)

    header = script.read_header(source_path)
    expected = int(config["input"]["expected_genes"])
    if tuple(header[:3]) != script.REQUIRED_PREFIX:
        raise ValueError(f"{client_id} prefix must be {script.REQUIRED_PREFIX}; got {header[:3]}")
    genes = header[3:]
    if len(genes) != expected:
        raise ValueError(f"Expected {expected:,} genes; found {len(genes):,}")
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate genes in the header")

    extra = {}
    if client_id == "GTEx":
        selected, receipt = script.select_gtex_donor_groups(source_path, gtex_donor_fraction, gtex_dose_seed, chunk_rows)
        extra = {"selected_gtex_groups": selected, "gtex_dose_receipt": receipt}
    script.prepare_client(
        source_path=source_path, source=client_id, genes=genes, config=config, config_sha256=script.sha256_file(config_path),
        output_dir=output_dir, chunk_rows=chunk_rows, seed=seed, training_protocol=training_protocol, **extra,
    )
    return ClientReport((output_dir / "manifest.json").read_bytes(), (output_dir / "scaler_stats.npz").read_bytes())
