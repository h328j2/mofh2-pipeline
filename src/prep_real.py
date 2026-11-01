"""Pick experimental MOFs from the RASPA library, featurise (graph + Zeo++), write 130 K GCMC queue."""
import os, sys, pickle, math, re
import numpy as np, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
P.CACHE = S + "/real_cache"
CIF = S + "/raspa/share/raspa/structures/cif"
skip = re.compile(r"^(COF|CoRE_|T130_|TOPO|hMOF|ZIF-1-blocked|IRMOF-1-Ar)|-primitive$|_ortho$|VASP|-DFT$|Dubbeldam|Dybtsev|-crystal$|-dry$")
dup = {"MIL-100", "MIL-101", "MOF-210", "NU-100SP", "NU-108-Cu", "PCN-68"}   # use the -primitive cell instead
names = sorted(n[:-4] for n in os.listdir(CIF) if n.endswith(".cif"))
cand = [n for n in names if (not skip.search(n) and n not in dup) or (n.endswith("-primitive") and n[:-10] in dup)]
cand += ["MIL-140A-DFT", "MIL-140B-DFT", "MIL-140C-DFT", "MIL-140D-DFT"]
def cells(cif):
    from pymatgen.core import Structure
    L = Structure.from_file(cif).lattice.matrix
    V = abs(np.linalg.det(L)); out = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        w = V / np.linalg.norm(np.cross(L[j], L[k])); out.append(math.ceil(25.6 / w))
    return " ".join(map(str, out))
if __name__ == "__main__":
    with Pool(2, maxtasksperchild=5) as pool:
        paths = list(pool.imap(P.featurise, [f"{CIF}/{n}.cif" for n in cand]))
    rows = []
    for n, p in zip(cand, paths):
        f = pickle.load(open(p, "rb"))
        rows.append(dict(name=n, status=f["status"], n_atoms=f.get("n_atoms"), density=f.get("density"), lcd=f.get("lcd"),
                         void_fraction=f.get("void_fraction"), cells=cells(f"{CIF}/{n}.cif") if f["status"] == "ok" else ""))
    d = pd.DataFrame(rows); print(d.to_string(index=False)); d.to_csv(S + "/real_picks.csv", index=False)
    ok = d[d.status == "ok"]
    with open(S + "/q_real.txt", "w") as fh:
        for p in (100, 5):
            for _, r in ok.iterrows(): fh.write(f"{r['name']}|{p}|{r.cells}\n")
    print(len(cand), "candidates,", len(ok), "featurised ok")
