"""Featurise the generated (ToBaCCo) structures used at 130 K and in the topology checks, from the packaged CIF copies.
Writes topo_cache/<original ToBaCCo name>.pkl, which finetune_130*.py and pi_head.py read."""
import os, sys, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
CIF = S + "/raspa/share/raspa/structures/cif"
def job(a):
    name, raspa = a
    out = f"{P.CACHE}/{name}.pkl"
    if os.path.exists(out): return
    import pickle; r = P._featurise(f"{CIF}/{raspa}.cif"); r["name"] = name; pickle.dump(r, open(out, "wb"))
if __name__ == "__main__":
    os.makedirs(P.CACHE, exist_ok=True)
    items = list(pd.read_csv(S + "/t130_picks.csv")[["name", "raspa"]].itertuples(index=False, name=None))
    items += list(pd.read_csv(S + "/topo_gcmc_picks.csv")[["name", "raspa"]].itertuples(index=False, name=None))
    with Pool(int(os.environ.get("NPROC", 4)), maxtasksperchild=20) as p: p.map(job, items)
    print("featurised", len(items))
