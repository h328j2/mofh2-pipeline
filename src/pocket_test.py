"""Do inaccessible pockets explain the 77 K/100 bar overprediction on dense CoRE-2019 test MOFs?
Zeo++ with an H2-sized probe (r = 1.45 A): accessible (AV) vs non-accessible (NAV) volume fraction."""
import os, re, subprocess, tempfile, sys, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from predict_topo import ZEO
KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")
d = pd.read_csv(S + "/train_160_test_predictions.csv").dropna(subset=["y77"])
rows = []
for n in d.name:
    with tempfile.TemporaryDirectory() as t:
        subprocess.run([ZEO, "-ha", "-vol", "1.45", "1.45", "50000", f"{t}/v", f"{S}/raspa/share/raspa/structures/cif/{n}.cif"], capture_output=True, timeout=600)
        kv = dict(KV.findall(open(f"{t}/v").read().replace("\n", " ")))
        rows.append(dict(name=n, AV_frac=float(kv["AV_Volume_fraction"]), NAV_frac=float(kv["NAV_Volume_fraction"])))
z = d.merge(pd.DataFrame(rows), on="name"); z["err77"] = z.gnn77_y100 - z.y77
vf = [float(x) for x in z.get("vf", pd.Series([np.nan] * len(z)))]
z["nav_share"] = z.NAV_frac / (z.AV_frac + z.NAV_frac)
print(z[["name", "density", "AV_frac", "NAV_frac", "y77", "gnn77_y100", "err77"]].sort_values("err77", ascending=False).round(3).to_string(index=False))
for c in ["NAV_frac", "nav_share", "AV_frac", "density", "qst0"]:
    print(f"err77 vs {c:10s}: Pearson {pearsonr(z[c], z.err77)[0]:+.2f}  Spearman {spearmanr(z[c], z.err77)[0]:+.2f}")
print("MOFs with any NAV:", int((z.NAV_frac > 0.001).sum()), "of", len(z))
z.to_csv(S + "/pocket_test.csv", index=False)
