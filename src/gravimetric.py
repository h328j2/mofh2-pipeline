"""Convert volumetric hydrogen uptake columns to gravimetric uptake.

The project feature caches store framework density as g/cm^3, while RASPA
reports it as kg/m^3. After normalizing density to kg/m^3:

    mg_H2/g_framework = uptake_gL / density_kg_m3 * 1000
    wt_percent        = uptake_gL / density_kg_m3 * 100

This conversion preserves the existing RASPA convention and avoids training a
second model for a quantity deterministically related to volumetric uptake.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def add_gravimetric_columns(
    table: pd.DataFrame,
    density: pd.Series | np.ndarray | None = None,
    suffix: str = "_gL",
    columns: list[str] | None = None,
    density_unit: str = "auto",
) -> pd.DataFrame:
    """Add mg/g and wt% columns for uptake-like volumetric columns."""
    out = table.copy()
    if density is None:
        for name in ("density", "density_kg_m3", "density_pmg"):
            if name in out:
                density = out[name]
                break
    if density is None:
        raise ValueError("A framework density column or density array is required")
    density = pd.to_numeric(pd.Series(density, index=out.index), errors="coerce")
    if (density.dropna() <= 0).any():
        raise ValueError("Every available framework density must be positive")
    if density_unit == "auto":
        density_unit = "kg_m3" if any(
            name in out.columns and name.endswith("kg_m3")
            for name in ("density_kg_m3",)
        ) else "g_cm3"
    if density_unit not in {"g_cm3", "kg_m3"}:
        raise ValueError("density_unit must be 'g_cm3', 'kg_m3', or 'auto'")
    density_kg_m3 = density if density_unit == "kg_m3" else density * 1000.0

    candidates = columns if columns is not None else list(out.columns)
    for column in candidates:
        if column not in out or (suffix and not column.endswith(suffix)):
            continue
        if any(token in column.lower() for token in ("error", "stderr", "std", "bias")):
            continue
        prefix = column[: -len(suffix)] if suffix else column
        values = pd.to_numeric(out[column], errors="coerce")
        out[f"{prefix}_mg_per_g"] = values / density_kg_m3 * 1000.0
        out[f"{prefix}_wt_percent"] = values / density_kg_m3 * 100.0
    return out


def _densities_from_cache(table: pd.DataFrame, cache: Path, id_column: str) -> pd.Series:
    values = []
    for name in table[id_column]:
        path = cache / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Feature cache is missing {path}")
        with path.open("rb") as handle:
            feature = pickle.load(handle)
        values.append(feature.get("density"))
    return pd.Series(values, index=table.index, name="density")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--id-column", default="mof")
    parser.add_argument("--density-column", default="density")
    parser.add_argument("--density-unit", choices=["auto", "g_cm3", "kg_m3"], default="auto")
    parser.add_argument(
        "--columns",
        help="Comma-separated uptake columns when names do not end in _gL",
    )
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    density = table[args.density_column] if args.density_column in table else None
    if density is None:
        if args.cache is None:
            raise ValueError(
                f"{args.input} has no {args.density_column!r}; provide --cache"
            )
        density = _densities_from_cache(table, args.cache, args.id_column)
        table["density"] = density
    columns = args.columns.split(",") if args.columns else None
    table = add_gravimetric_columns(
        table,
        density=density,
        suffix="" if columns else "_gL",
        columns=columns,
        density_unit=args.density_unit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(table)} rows)")


if __name__ == "__main__":
    main()
