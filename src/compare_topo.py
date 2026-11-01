import glob, sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from extract import parse_data
P = pd.read_csv(S + "/topo_gcmc_picks.csv")
def g(name, p):
    f = glob.glob(f"{S}/runs_topo/{name}_{p}/Output/System_0/*.data")
    if not f: return (np.nan, np.nan)
    r = parse_data(f[0])
    return (r["gL"], r["gL_err"]) if r.get("finished") and r.get("gL") is not None else (np.nan, np.nan)
for p in (2, 100):
    v = [g(n, p) for n in P.raspa]
    P[f"gcmc_y{p}"] = [a for a, _ in v]; P[f"gcmc_y{p}_err"] = [b for _, b in v]
P["gcmc_dc"] = P.gcmc_y100 - P.gcmc_y2
P["err_dc"] = P.gnn_dc - P.gcmc_dc
P["err_y100"] = P.gnn_y100 - P.gcmc_y100
P["err_y2"] = P.gnn_y2 - P.gcmc_y2
P.to_csv(S + "/topo_gcmc_validation.csv", index=False)
pd.set_option("display.width", 220)
print(P[["raspa", "gnn_y2", "gcmc_y2", "gnn_y100", "gcmc_y100", "gnn_dc", "gcmc_dc", "err_dc", "void_fraction", "lcd"]].round(2).to_string(index=False))
d = P.dropna(subset=["gcmc_dc"])
if len(d) >= 3:
    for t in ("y2", "y100", "dc"):
        e = d[f"err_{t}"]
        print(f"{t}: n={len(d)} MAE={e.abs().mean():.2f} bias={e.mean():+.2f} Spearman={spearmanr(d[f'gnn_{t}'], d[f'gcmc_{t}'])[0]:.2f}")
