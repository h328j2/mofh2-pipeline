"""Bulk H2 density (g/L) from the empty-box GCMC runs (30 A box): the H2-H2 convention test."""
import glob, re, os, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); rows = []
for d in sorted(glob.glob(S + "/runs_box/*")):
    f = glob.glob(d + "/Output/System_0/*.data")
    if not f: continue
    m = re.search(r"Average loading absolute \[molecules/unit cell\]\s+([\d.]+)", open(f[0]).read())
    if m: rows.append(dict(run=os.path.basename(d), molecules=float(m.group(1)), density_gL=float(m.group(1)) * 2.016 / 6.02214e23 / 27000e-24 * 1000))
df = pd.DataFrame(rows); print(df.round(2).to_string(index=False)); df.to_csv(S + "/bulk_box_densities.csv", index=False)
print("Reference: real H2 at 77 K, 100 bar = 31.3 g/L (CoolProp); report: FH 1.008 u 27.8, FH 2.016 u 30.5, classical 35.2")
