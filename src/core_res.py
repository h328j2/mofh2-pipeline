import os, sys, subprocess, tempfile, numpy as np, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from predict_topo import ZEO
D = S + "/coredl/MOFSimplify/CoRE2019"
def res(fn):
    with tempfile.TemporaryDirectory() as t:
        try:
            subprocess.run(["prlimit", "--as=1500000000", ZEO, "-res", f"{t}/s.res", f"{D}/{fn}"], capture_output=True, timeout=120)
            tok = open(f"{t}/s.res").read().split(); return fn, float(tok[1]), float(tok[2])
        except Exception: return fn, np.nan, np.nan
if __name__ == "__main__":
    d = pd.read_csv(S + "/core_meta.csv")
    d["ncells"] = d.cells.str.split().apply(lambda x: np.prod(list(map(int, x))))
    m = (d.n_atoms <= 600) & d.metals.notna() & d.density.between(0.2, 2.2) & d.H & (d.n_atoms * d.ncells <= 6000)
    fs = list(d[m].file)
    with Pool(2) as p: out = p.map(res, fs, chunksize=20)
    pd.DataFrame(out, columns=["file", "lcd_raw", "pld_raw"]).to_csv(S + "/core_res.csv", index=False); print("done", len(out))
