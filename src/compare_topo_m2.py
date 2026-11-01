import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from extract import parse_data
t = pd.read_csv(S + "/topo_gcmc_validation.csv")
def g(name):
    f = glob.glob(f"{S}/runs_m2/{name}_100/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") else np.nan
t["gcmc_y100_m2"] = [g(n) for n in t.raspa]
t["gcmc_dc_m2"] = t.gcmc_y100_m2 - t.gcmc_y2
t["err_y100_m2"] = t.gnn_y100 - t.gcmc_y100_m2
t["err_dc_m2"] = t.gnn_dc - t.gcmc_dc_m2
pd.set_option("display.width", 220)
print(t[["raspa", "density", "void_fraction", "gnn_y100", "gcmc_y100", "gcmc_y100_m2", "gnn_dc", "gcmc_dc", "gcmc_dc_m2", "err_dc", "err_dc_m2"]].round(2).to_string(index=False))
d = t.dropna(subset=["gcmc_y100_m2"])
print(f"n={len(d)}  DC: MAE old {d.err_dc.abs().mean():.2f} -> {d.err_dc_m2.abs().mean():.2f}, bias {d.err_dc.mean():+.2f} -> {d.err_dc_m2.mean():+.2f}, Spearman {spearmanr(d.gnn_dc, d.gcmc_dc_m2)[0]:.2f}")
print(f"      y100: MAE old {d.err_y100.abs().mean():.2f} -> {d.err_y100_m2.abs().mean():.2f}, bias {d.err_y100.mean():+.2f} -> {d.err_y100_m2.mean():+.2f}")
if len(d) > 4:
    print(f"      r(err_dc_m2, density) {pearsonr(d.density, d.err_dc_m2)[0]:+.2f}  r(err_dc_m2, void fraction) {pearsonr(d.void_fraction, d.err_dc_m2)[0]:+.2f}")
t.to_csv(S + "/topo_gcmc_validation_m2.csv", index=False)
