"""Stage B: from the 1000 Widom-characterised CoRE MOFs choose 500, over-sampling strong binders (high Qst0 / Henry);
hold out a stratified test set; write GCMC queues (160 K/5 bar for all 500, 77 K/100 bar for a 30-MOF test subset)."""
import os, sys, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from widom import widom
A = pd.read_csv(S + "/core_stageA.csv")
W = [widom(f"{S}/runs_w160/{r}") or {} for r in A.raspa]
A["kH"] = [w.get("kH") for w in W]; A["qst0"] = [w.get("qst0") for w in W]
A = A.dropna(subset=["kH"]); A = A[(A.kH > 0) & (A.qst0 <= 12.5)]   # >12.5 kJ/mol: deep pockets / likely structure artefacts
A["logkH"] = np.log10(A.kH)
print("widom ok:", len(A)); print(A[["qst0", "logkH", "density", "lcd_raw"]].describe().round(2))
rng = np.random.default_rng(0)
top = A.nlargest(150, "qst0")                       # strongest binders: always included
rest = A.drop(top.index)
rest["q_bin"] = pd.qcut(rest.logkH, 5, labels=False); rest["d_bin"] = pd.qcut(rest.density, 5, labels=False)
pick = pd.concat([g.sample(min(len(g), 20), random_state=0) for _, g in rest.groupby(["q_bin", "d_bin"])])
pick = pick.sample(min(len(pick), 350), random_state=0)
B = pd.concat([top, pick.drop(columns=["q_bin", "d_bin"], errors="ignore")])
# test split: 75 stratified by Qst0 quintile; first 30 of them also get 77 K/100 bar
B["qq"] = pd.qcut(B.qst0, 5, labels=False)
test = pd.concat([g.sample(15, random_state=1) for _, g in B.groupby("qq")])
B["split"] = np.where(B.index.isin(test.index), "test", "train")
t30 = pd.concat([g.head(6) for _, g in test.groupby("qq")])
B["g77"] = B.index.isin(t30.index)
B.drop(columns=["qq"], errors="ignore").to_csv(S + "/core_stageB.csv", index=False)
print(len(B), "picked;", B.split.value_counts().to_dict(), "; 77 K test:", int(B.g77.sum()))
with open(S + "/q_stage2.txt", "w") as f:
    for r, u in zip(B.raspa, B.cells): f.write(f"g160|{r}|{u}|{S}/runs_160/{r}_5\n")
with open(S + "/q_stage3.txt", "w") as f:
    for r, u in zip(B[B.g77].raspa, B[B.g77].cells): f.write(f"g77|{r}|{u}|{S}/runs_m2/{r}_100\n")
