"""Deploy-time check, run once on each node: the prepared data matches the federated manifest.

It runs once, not per message: hashing a ~1 GB array every round would dominate the round time.
"""

from __future__ import annotations

import argparse

from .contract import SITES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=[*SITES, "server"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from .real import Env

    env = Env({"seed": args.seed, "local-epochs": 1})  # also validates the federated manifest and scaler
    if args.site == "server":
        genes = (env.site_root / "gene_order.txt").read_text(encoding="utf-8").splitlines()
        if len(genes) != env.config["input"]["expected_genes"]:
            raise SystemExit("gene order violates the COMPASS contract")
        print("server data verified")
        return

    fc, central = env.fc, env.central
    client_id = SITES[args.site][0]
    client_dir = env.site_root / "clients" / args.site
    client = fc.validate_client_prepared_manifest(
        client_dir, expected_client_id=client_id,
        expected_config_sha256=central.sha256_file(env.config_path), expected_split_seed=env.streams["split"],
    )
    manifest, contract = env.manifest, env.manifest["clients"][client_id]
    client_sha = central.sha256_file(client_dir / "manifest.json")
    hashes = manifest.get("prepared_artifact_hashes", {})
    if (
        contract.get("client_dataset_id") != client["client_dataset_id"]
        or contract.get("manifest_sha256") != client_sha
        or contract.get("artifact_hashes") != client["artifact_hashes"]
        or hashes.get("client_manifest_sha256", {}).get(client_id) != client_sha
        or hashes.get("clients", {}).get(client_id) != client["artifact_hashes"]
        or manifest.get("source_input_sha256", {}).get(client_id) != client["source_sha256"]
        or manifest.get("gene_order_sha256") != client["artifact_hashes"]["gene_order_sha256"]
    ):
        raise SystemExit(f"federated manifest does not bind the prepared {client_id} data")
    if client["genes"] != env.config["input"]["expected_genes"]:
        raise SystemExit("client gene count differs from the COMPASS contract")
    print(f"{client_id} data verified ({client['samples']} samples)")


if __name__ == "__main__":
    main()
