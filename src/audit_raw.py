"""Raw-file audit: re-read every reported simulation value directly from the RASPA output files and compare."""
import glob, re, os, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
def raw(d):
    f = glob.glob(d + "/Output/System_0/*.data")
    if not f: return np.nan, "missing"
    t = open(f[0]).read()
    if "Simulation finished" not in t: return np.nan, "unfinished"
    m = re.search(r"Average loading absolute \[cm\^3 \(STP\)/cm\^3 framework\]\s+([-\d.eE+]+)", t)
    return float(m.group(1)) * 2.016 / 22.414, "ok"
checks = []
def chk(label, reported, d):
    v, s = raw(d); checks.append(dict(item=label, reported=reported, raw_file=v, status=s, match=(s == "ok" and abs(v - reported) < 0.01)))
def opt(fn):
    p = f"{S}/{fn}"; return pd.read_csv(p) if os.path.exists(p) else None
if (t := opt("screen_top20_ft_gcmc.csv")) is not None:
    for _, r in t.iterrows(): chk(f"top20 {r.raspa} 77K/100", r.g100, f"{S}/runs_m2/{r.raspa}_100"); chk(f"top20 {r.raspa} 160K/5", r.g160, f"{S}/runs_160/{r.raspa}_5")
if (o := opt("screen_top10_gcmc.csv")) is not None:
    for _, r in o.iterrows(): chk(f"orig top10 {r.raspa} 77K/100", r.g100, f"{S}/runs_m2/{r.raspa}_100"); chk(f"orig top10 {r.raspa} 77K/2", r.g2, f"{S}/runs_m2/{r.raspa}_2")
if (b := opt("screen_bottom5_gcmc.csv")) is not None:
    for _, r in b.iterrows(): chk(f"bottom5 {r.raspa} 77K/100", r.g100, f"{S}/runs_m2/{r.raspa}_100")
if (e := opt("exp_gcmc_isotherms.csv")) is not None:
    for _, r in e.iterrows(): chk(f"exp {r.mof} {r.P}bar", r.abs_gL, f"{S}/runs_exp/{r.mof}_{r.P}")
if (p := opt("train_160_test_predictions.csv")) is not None:
    for _, r in p.iterrows(): chk(f"160K test {r['name']}", r.y, f"{S}/runs_160/{r['name']}_5")
c = pd.DataFrame(checks)
print("values checked:", len(c), "| matching raw output files:", int(c.match.sum()), "| missing/unfinished:", int((c.status != "ok").sum()))
print(c[~c.match].to_string(index=False) if len(c) and (~c.match).any() else "no mismatches")
c.to_csv(S + "/audit_raw.csv", index=False)
