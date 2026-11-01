"""How well does the 77 K model stand in for the DOE swing (77 K/100 bar -> 160 K/5 bar)?"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from extract import parse_data
def g(root, name, p):
    f = glob.glob(f"{S}/{root}/{name}_{p}/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan
rows = []
v = pd.read_csv(S + "/gcmc_validation.csv")
for _, r in v.iterrows():
    if r.set == "hMOF" and r["name"] not in ("hMOF-3001465", "hMOF-5040736", "hMOF-8081"):
        continue
    raspa = r["name"] if r.set == "hMOF" else "CoRE_" + re.sub(r"_+", "_", r["name"].replace("[", "_").replace("]", "_")).strip("_")
    rows.append(dict(set=r.set, name=r["name"], raspa=raspa, ref_kind="hMOF label" if r.set == "hMOF" else "ML consensus",
                     pred_y2=r.ref_y2, pred_y100=r.ref_y100, gcmc_y2=r.gcmc_y2))
t = pd.read_csv(S + "/topo_gcmc_validation.csv")
for _, r in t.iterrows():
    rows.append(dict(set="ToBaCCo", name=r.raspa, raspa=r.raspa, ref_kind="GNN", pred_y2=r.gnn_y2, pred_y100=r.gnn_y100, gcmc_y2=r.gcmc_y2, density=r.density))
d = pd.DataFrame(rows)
d["gcmc_y100"] = [g("runs_m2", n, 100) for n in d.raspa]
d["gcmc_160_5"] = [g("runs_160", n, 5) for n in d.raspa]
d["pred_dc_iso"] = d.pred_y100 - d.pred_y2               # what the model predicts (77 K, 100 -> 2 bar)
d["gcmc_dc_iso"] = d.gcmc_y100 - d.gcmc_y2
d["gcmc_dc_doe"] = d.gcmc_y100 - d.gcmc_160_5            # DOE swing, 77 K/100 bar -> 160 K/5 bar
d["gain_doe"] = d.gcmc_dc_doe - d.gcmc_dc_iso
pd.set_option("display.width", 230)
print(d[["set", "name", "gcmc_y2", "gcmc_160_5", "pred_y2", "gcmc_dc_iso", "gcmc_dc_doe", "gain_doe", "pred_dc_iso"]].round(2).to_string(index=False))
m = d[d.set != "hMOF"].dropna(subset=["gcmc_dc_doe"])
if len(m) > 3:
    e1 = m.pred_dc_iso - m.gcmc_dc_doe
    e2 = m.pred_y2 - m.gcmc_160_5
    e3 = m.pred_dc_iso - m.gcmc_dc_iso
    print(f"\nn={len(m)} (CoRE + ToBaCCo)")
    print(f"Model isothermal DC vs GCMC isothermal DC : MAE {e3.abs().mean():.2f}  bias {e3.mean():+.2f}  Spearman {spearmanr(m.pred_dc_iso, m.gcmc_dc_iso)[0]:.2f}")
    print(f"Model isothermal DC vs GCMC DOE DC        : MAE {e1.abs().mean():.2f}  bias {e1.mean():+.2f}  Spearman {spearmanr(m.pred_dc_iso, m.gcmc_dc_doe)[0]:.2f}")
    print(f"Model 77K/2bar vs GCMC 160K/5bar          : MAE {e2.abs().mean():.2f}  bias {e2.mean():+.2f}  Spearman {spearmanr(m.pred_y2, m.gcmc_160_5)[0]:.2f}")
    print(f"DOE gain over isothermal: mean {m.gain_doe.mean():.2f} g/L, range {m.gain_doe.min():.2f} to {m.gain_doe.max():.2f}")
    # simple fixed-offset estimate: DOE DC ~ model DC + mean gain (fitted on CoRE, tested on ToBaCCo)
    c = m[m.set == "CoRE"]; o = m[m.set == "ToBaCCo"]
    if len(c) and len(o):
        off = (c.gcmc_dc_doe - c.pred_dc_iso).mean()
        e4 = o.pred_dc_iso + off - o.gcmc_dc_doe
        print(f"Offset fitted on CoRE ({off:+.2f}) applied to ToBaCCo: MAE {e4.abs().mean():.2f}  bias {e4.mean():+.2f}")
d.to_csv(S + "/doe_swing_test.csv", index=False)
