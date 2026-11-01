"""Compare 100-bar uptake under three H2-H2 conventions with hMOF labels and model predictions."""
import glob, os, re, sys
import numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from extract import parse_data
def g(root, name):
    f = glob.glob(f"{S}/{root}/{name}_100/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0])
    return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan
v = pd.read_csv(S + "/gcmc_validation.csv")
rows = []
for _, r in v.iterrows():
    if r.set == "hMOF" and r["name"] not in ("hMOF-3001465", "hMOF-5040736", "hMOF-8081"):
        continue
    raspa = r["name"] if r.set == "hMOF" else "CoRE_" + re.sub(r"_+", "_", r["name"].replace("[", "_").replace("]", "_")).strip("_")
    rows.append(dict(set=r.set, name=r["name"], ref_kind="label" if r.set == "hMOF" else "prediction",
                     ref_y100=r.ref_y100, ref_y2=r.ref_y2, gcmc_y2=r.gcmc_y2,
                     fh=r.gcmc_y100, fh_mass2=g("runs_m2", raspa), classical=g("runs_noFH", raspa)))
d = pd.DataFrame(rows)
d["gap_fh"] = d.ref_y100 - d.fh
d["gap_mass2"] = d.ref_y100 - d.fh_mass2
d["dc_ref"] = d.ref_y100 - d.ref_y2
d["dc_mass2"] = d.fh_mass2 - d.gcmc_y2
d["dc_gap_mass2"] = d.dc_ref - d.dc_mass2
pd.set_option("display.width", 220)
print(d[["set", "name", "ref_y100", "fh", "fh_mass2", "classical", "gap_fh", "gap_mass2", "dc_ref", "dc_mass2", "dc_gap_mass2"]].round(2).to_string(index=False))
for s in ("hMOF", "CoRE"):
    x = d[(d.set == s)].dropna(subset=["fh_mass2"])
    if len(x):
        print(f"{s}: n={len(x)} mean gap FH {x.gap_fh.mean():+.2f} -> mass-2.016 FH {x.gap_mass2.mean():+.2f} g/L at 100 bar; DC gap {x.dc_gap_mass2.mean():+.2f}")
d.to_csv(S + "/h2h2_convention_test.csv", index=False)
