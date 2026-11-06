"""Prepare the published PoroNet H2 benchmark for MOF-H2 model validation.

The PoroNet repository publishes paired GCMC labels for the same 1,990 MOFs at
160 K / 5 bar and 77 K / 100 bar.  This script converts those spreadsheets
into one CSV and records the source metadata.  CIF retrieval is intentionally
separate because the structures are served by MOFDB rather than bundled with
the label repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("/tmp/PoroNet/GCMC_output")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data"


def load_labels(source: Path) -> pd.DataFrame:
    low = pd.read_excel(source / "160K_5bar-mof-gL.xlsx").rename(
        columns={"gas_loading_g/L": "uptake_160K_5bar_gL",
                 "standard_error": "stderr_160K_5bar_gL"}
    )
    high = pd.read_excel(source / "77K_100bar-mof-gL.xlsx").rename(
        columns={"gas_loading_g/L": "uptake_77K_100bar_gL",
                 "standard_error": "stderr_77K_100bar_gL"}
    )
    required = {
        "mof", "uptake_160K_5bar_gL", "stderr_160K_5bar_gL",
        "uptake_77K_100bar_gL", "stderr_77K_100bar_gL",
    }
    merged = low.merge(high, on="mof", validate="one_to_one")
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Missing expected PoroNet columns: {sorted(missing)}")
    merged["deliverable_77K100_to_160K5_gL"] = (
        merged["uptake_77K_100bar_gL"] - merged["uptake_160K_5bar_gL"]
    )
    return merged[
        [
            "mof",
            "uptake_160K_5bar_gL",
            "stderr_160K_5bar_gL",
            "uptake_77K_100bar_gL",
            "stderr_77K_100bar_gL",
            "deliverable_77K100_to_160K5_gL",
        ]
    ].sort_values("mof").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    labels = load_labels(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "poronet_h2_benchmark.csv"
    labels.to_csv(out, index=False)
    manifest = pd.DataFrame(
        [
            {
                "dataset": "PoroNet H2 benchmark",
                "paper": "Zheng, Gopalan, Shi, JCTC 2026",
                "doi": "10.1021/acs.jctc.6c00100",
                "repository": "https://github.com/Shi-Research-Group/PoroNet",
                "cif_source": "https://mof.tech.northwestern.edu/mofs/",
                "n_mofs": len(labels),
                "labels": "160 K / 5 bar; 77 K / 100 bar; deliverable difference",
                "units": "g/L",
            }
        ]
    )
    manifest.to_csv(args.output_dir / "poronet_h2_manifest.csv", index=False)
    print(f"saved {out} ({len(labels)} MOFs)")
    print(labels.describe().round(4).to_string())


if __name__ == "__main__":
    main()
