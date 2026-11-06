"""Featurise the CIFs exported from the PoroNet literature benchmark."""

from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import sys

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
import predict_topo as P  # noqa: E402

KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")


def fast_zeo(cif: str):
    from pymatgen.core import Structure

    structure = Structure.from_file(cif)
    with tempfile.TemporaryDirectory() as tmp:
        out = {"density_pmg": float(structure.density)}
        subprocess.run(
            ["prlimit", "--as=1500000000", P.ZEO, "-res", f"{tmp}/r", cif],
            capture_output=True,
            timeout=300,
            check=True,
        )
        tokens = Path(f"{tmp}/r").read_text().split()
        out["lcd_zeo"] = 0.981 * float(tokens[1]) + 0.152
        out["pld_zeo"] = 0.987 * float(tokens[2]) + 0.186
        subprocess.run(
            ["prlimit", "--as=1500000000", P.ZEO, "-sa", "1.86", "1.86", "500", f"{tmp}/s", cif],
            capture_output=True,
            timeout=300,
            check=True,
        )
        out["sa_m2g_zeo"] = float(
            dict(KV.findall(Path(f"{tmp}/s").read_text().splitlines()[0]))["ASA_m^2/g"]
        )
        subprocess.run(
            ["prlimit", "--as=1500000000", P.ZEO, "-vol", "0.0", "0.0", "5000", f"{tmp}/v", cif],
            capture_output=True,
            timeout=300,
            check=True,
        )
        out["vf_zeo_0.0"] = float(
            dict(KV.findall(Path(f"{tmp}/v").read_text().splitlines()[0]))["AV_Volume_fraction"]
        )
        return structure, out


P.zeo = fast_zeo


def job(args: tuple[Path, Path]) -> tuple[str, str]:
    cif, cache = args
    out = cache / f"{cif.stem}.pkl"
    if not out.exists():
        pickle.dump(P._featurise(str(cif)), out.open("wb"))
    return cif.name, pickle.load(out.open("rb")).get("status", "missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", type=Path, default=S.parent / "data/poronet_cif")
    parser.add_argument("--cache", type=Path, default=S.parent / "data/poronet_cache")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cifs = sorted(args.cif_dir.glob("*.cif"))
    tasks = [(cif, args.cache) for cif in cifs]
    with Pool(args.workers, maxtasksperchild=20) as pool:
        counts: dict[str, int] = {}
        for i, (name, status) in enumerate(pool.imap_unordered(job, tasks, chunksize=1), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0:
                print(i, counts, flush=True)
    print(f"featurised {len(cifs)} structures: {counts}")


if __name__ == "__main__":
    main()
