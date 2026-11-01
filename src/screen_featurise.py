"""Featurise all accessible, family-unique CoRE 2019 MOFs with fast Zeo++ (validated: LCD/PLD within ~0.12 A of -ha)."""
import os, re, sys, subprocess, tempfile, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
from predict_topo import ZEO
KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")
D = S + "/coredl/MOFSimplify/CoRE2019"; OUT = S + "/screen_cache"
def zeo_fast(cif):
    from pymatgen.core import Structure
    s = Structure.from_file(cif)
    with tempfile.TemporaryDirectory() as t:
        o = {"density_pmg": s.density}
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-res", f"{t}/r", cif], capture_output=True, timeout=300)
        tok = open(f"{t}/r").read().split(); o["lcd_zeo"], o["pld_zeo"] = 0.981 * float(tok[1]) + 0.152, 0.987 * float(tok[2]) + 0.186
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-sa", "1.86", "1.86", "500", f"{t}/s", cif], capture_output=True, timeout=300)
        o["sa_m2g_zeo"] = float(dict(KV.findall(open(f"{t}/s").readline()))["ASA_m^2/g"])
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-vol", "0.0", "0.0", "5000", f"{t}/v", cif], capture_output=True, timeout=300)
        o["vf_zeo_0.0"] = float(dict(KV.findall(open(f"{t}/v").readline()))["AV_Volume_fraction"])
    return s, o
P.zeo = zeo_fast
def job(fn):
    cp = f"{OUT}/{fn[:-4]}.pkl"
    if os.path.exists(cp): return
    r = P._featurise(f"{D}/{fn}"); r["file"] = fn
    pickle.dump(r, open(cp, "wb"))
if __name__ == "__main__":
    d = pd.read_csv(S + "/core_meta.csv").merge(pd.read_csv(S + "/core_res.csv"), on="file")
    d = d[(d.pld_raw >= 2.6) & (d.lcd_raw >= 3.3)].copy(); d["name"] = d.file.str[:-4]
    d["family"] = [m.group(1) if (m := re.match(r"^([A-Z]{6})\d*_", n)) else n for n in d.name]
    d = d.sample(frac=1, random_state=0).drop_duplicates("family")
    d[["file", "name", "family", "n_atoms", "density", "metals", "cells", "lcd_raw", "pld_raw"]].to_csv(S + "/screen_list.csv", index=False)
    fs = list(d.file); print("to featurise:", len(fs), flush=True)
    with Pool(2, maxtasksperchild=50) as p:
        for k, _ in enumerate(p.imap_unordered(job, fs, chunksize=4)):
            if k % 500 == 0: print(k, flush=True)
    print("FEATURISE_DONE", flush=True)
