"""Stage A: pick ~1000 accessible, de-duplicated CoRE 2019 MOFs, stratified by pore size x density (for Widom)."""
import os, re, sys, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import core_meta as M
from core_res import res
import core_res as R
d = pd.read_csv(S + "/core_meta.csv").merge(pd.read_csv(S + "/core_res.csv"), on="file")
d = d[(d.pld_raw >= 2.6) & (d.lcd_raw >= 3.3)].copy()
d["name"] = d.file.str[:-4]
d["family"] = [m.group(1) if (m := re.match(r"^([A-Z]{6})\d*_", n)) else n for n in d.name]
d = d.sample(frac=1, random_state=0).drop_duplicates("family")
print("accessible, unique families:", len(d))
# drop near-duplicates of the 11 external and the 58 library MOFs (density within 1.5%, LCD within 0.3 A, same metals)
CIF = S + "/raspa/share/raspa/structures/cif"
held = ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1", "CoRE_2007_Zn_pcu_3_ASR_3",
        "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2", "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
lib = pd.read_csv(S + "/real_picks.csv"); held += list(lib[lib.status == "ok"].name)
M.D = CIF; R.D = CIF
ref = []
for n in held:
    m = M.parse(n + ".cif"); _, lcd, _ = res(n + ".cif"); ref.append((n, m.get("density"), lcd, m.get("metals")))
drop = set()
for n, den, lcd, met in ref:
    if den is None or np.isnan(lcd): continue
    hit = d[(abs(d.density / den - 1) < 0.015) & (abs(d.lcd_raw - lcd) < 0.3) & (d.metals == met)]
    drop |= set(hit.name)
d = d[~d.name.isin(drop)]; print("dropped near-duplicates of held-out/library:", len(drop))
d["lcd_bin"] = pd.cut(d.lcd_raw, [0, 5, 6.5, 8, 10, 13, 100]); d["den_bin"] = pd.cut(d.density, [0, 0.8, 1.1, 1.4, 1.7, 3])
g = d.groupby(["lcd_bin", "den_bin"], observed=True)
print(g.size().unstack())
cells = list(g.groups); per = 1000 // len(cells); pick = []
sizes = g.size().sort_values()
left, k = 1000, len(sizes)
for key, n in sizes.items():       # equal allocation, redistributing from small cells
    take = min(n, left // k); pick.append(g.get_group(key).head(take)); left -= take; k -= 1
A = pd.concat(pick); print("stage A picks:", len(A)); print(A.groupby(["lcd_bin", "den_bin"], observed=True).size().unstack())
A.drop(columns=["lcd_bin", "den_bin"]).to_csv(S + "/core_stageA.csv", index=False)
