"""Tiny synthetic TCGA/GTEx expression TSVs in the format the COMPASS prep scripts expect."""

from pathlib import Path

import numpy as np
import pandas as pd

GENES = 15_672  # the COMPASS input contract


def write_tsvs(directory: Path, per_context: int = 12, gtex: int = 40, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    genes = [f"G{i:05d}" for i in range(GENES)]

    def expression(rows: int) -> np.ndarray:
        return rng.lognormal(mean=1.0, sigma=1.5, size=(rows, GENES)).astype(np.float32)

    codes = np.repeat(np.arange(33), per_context)
    tcga = pd.DataFrame(expression(len(codes)), columns=genes)
    tcga.insert(0, "cancer_code", codes)
    tcga.insert(0, "cancer_type", [f"ctx{c}" for c in codes])
    tcga.insert(0, "sample_id", [f"TCGA-{i // 10000:02d}-{i % 10000:04d}-01A-11R" for i in range(len(codes))])
    donors = [f"GTEX-{i:04d}-{i:04d}-SM-ABCDE" for i in range(gtex)]
    gtex_df = pd.DataFrame(expression(gtex), columns=genes)
    gtex_df.insert(0, "cancer_code", np.nan)
    gtex_df.insert(0, "cancer_type", "normal")
    gtex_df.insert(0, "sample_id", donors)
    tcga_path, gtex_path = directory / "tcga.tsv", directory / "gtex.tsv"
    tcga.to_csv(tcga_path, sep="\t", index=False)
    gtex_df.to_csv(gtex_path, sep="\t", index=False)
    return tcga_path, gtex_path
