"""Collect hMOF-matched GCMC results and compare with hMOF labels / ML predictions."""
import glob, os, re, sys
import pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
U = S + "/user_zip"
sys.path.insert(0, S)
from extract import parse_data

def gcmc(name, P):
    f = glob.glob(f"{S}/runs_fh/{name}_{P}/Output/System_0/*.data")
    if not f:
        return None
    r = parse_data(f[0])
    return r if r.get("finished") and r.get("gL") is not None else None

rows = []
h = pd.read_csv(U + "/hmof_targets.csv")
for _, t in h.iterrows():
    r2, r100 = gcmc(t["name"], 2), gcmc(t["name"], 100)
    rows.append(dict(set="hMOF", name=t["name"], role=t["reason"],
                     ref_y2=t["y2"], ref_y100=t["y100"], ref_kind="hMOF label",
                     vf=t.get("void_fraction"),
                     gcmc_y2=r2 and r2["gL"], gcmc_y2_err=r2 and r2["gL_err"],
                     gcmc_y100=r100 and r100["gL"], gcmc_y100_err=r100 and r100["gL_err"]))
c = pd.read_csv(U + "/core_top10.csv")
c = c[~c["name"].str.contains(r"\[FSR\]")]
for _, t in c.iterrows():
    safe = "CoRE_" + re.sub(r"_+", "_", t["name"].replace("[", "_").replace("]", "_")).strip("_")
    r2, r100 = gcmc(safe, 2), gcmc(safe, 100)
    rows.append(dict(set="CoRE", name=t["name"], role="top-10 in-domain prediction",
                     ref_y2=(t["xgb_y2"] + t["gnn_y2"]) / 2, ref_y100=(t["xgb_y100"] + t["gnn_y100"]) / 2,
                     ref_kind="ML consensus", vf=t["void_fraction"],
                     gcmc_y2=r2 and r2["gL"], gcmc_y2_err=r2 and r2["gL_err"],
                     gcmc_y100=r100 and r100["gL"], gcmc_y100_err=r100 and r100["gL_err"]))
df = pd.DataFrame(rows)
df["ref_dc"] = df["ref_y100"] - df["ref_y2"]
df["gcmc_dc"] = df["gcmc_y100"] - df["gcmc_y2"]
df["floor100"] = df["vf"] * 31.3
out = S + "/gcmc_validation.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 220)
print(df[["set", "name", "role", "ref_y2", "gcmc_y2", "ref_y100", "gcmc_y100", "gcmc_y100_err", "ref_dc", "gcmc_dc", "floor100"]].round(2).to_string(index=False))
