"""Export the CIF strings bundled in the PoroNet structure cache.

PoroNet's selected-MOF Feather file contains CIF text for 1,998 structures,
so MOFDB access is not required for this benchmark subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/tmp/PoroNet/Pore_Graph_Generation/Selected_MOFs_for_H2.feather"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/poronet_cif")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/poronet_h2_benchmark.csv",
    )
    args = parser.parse_args()

    labels = set(pd.read_csv(args.labels)["mof"])
    structures = pd.read_feather(args.source)
    required = {"name", "cif"}
    if not required.issubset(structures.columns):
        raise ValueError(f"Expected columns {sorted(required)}; found {list(structures)}")
    selected = structures[structures["name"].isin(labels)]
    args.output.mkdir(parents=True, exist_ok=True)
    for row in selected.itertuples(index=False):
        (args.output / f"{row.name}.cif").write_text(row.cif)
    print(f"exported {len(selected)} CIFs to {args.output}")
    print(f"labels without CIF: {len(labels - set(selected.name))}")


if __name__ == "__main__":
    main()
