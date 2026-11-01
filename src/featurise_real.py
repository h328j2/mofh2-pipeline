"""Featurise the 58 experimental library MOFs used at 130 K / 160 K (list: real_picks.csv, status ok).
Eight large cubic frameworks are featurised from their primitive cell, as in the original run (GCMC still uses the full CIF)."""
import os, sys, pickle, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
CIF = S + "/raspa/share/raspa/structures/cif"; OUT = S + "/real_cache"; PRIM = S + "/real_prim"
PRIMITIVE = {"Cu-BTC", "IRMOF-13", "IRMOF-14", "NU-108-Zn", "PCN-60", "PCN-61", "ZIF-5", "ZIF-96"}
def job(n):
    out = f"{OUT}/{n}.pkl"
    if os.path.exists(out): return n, "cached"
    src = f"{CIF}/{n}.cif"
    if n in PRIMITIVE:
        from pymatgen.core import Structure
        Structure.from_file(src).get_primitive_structure(tolerance=0.05).to(filename=f"{PRIM}/{n}.cif"); src = f"{PRIM}/{n}.cif"
    r = P._featurise(src); r["name"] = n; pickle.dump(r, open(out, "wb")); return n, r["status"]
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(PRIM, exist_ok=True)
    names = list(pd.read_csv(S + "/real_picks.csv").query("status == 'ok'").name)
    with Pool(int(os.environ.get("NPROC", 4)), maxtasksperchild=10) as p: res = p.map(job, names)
    print(len(names), "library MOFs:", pd.Series([s for _, s in res]).value_counts().to_dict())
