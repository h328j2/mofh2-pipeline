"""Fast text-level metadata for all CoRE 2019 CIFs (all P1): atoms, density, metals, cell widths."""
import os, re, sys, math, numpy as np, pandas as pd
from multiprocessing import Pool
D, OUT = (sys.argv[1:3] + [None, None])[:2] if __name__ == "__main__" else (None, None)
MASS = {}
from pymatgen.core.periodic_table import Element
METALS = None
def parse(fn):
    try:
        t = open(os.path.join(D, fn)).read().splitlines()
        c = {}
        for l in t:
            for k in ("a", "b", "c", "alpha", "beta", "gamma"):
                if l.startswith(f"_cell_length_{k} ") or l.startswith(f"_cell_angle_{k} "):
                    c[k] = float(l.split()[1].split("(")[0])
        # atom rows: lines after _atom_site_* headers with >=5 tokens
        hdr, rows, inloop = [], [], False
        for l in t:
            s = l.strip()
            if s.startswith("_atom_site_"): hdr.append(s); inloop = True; continue
            if inloop and s and not s.startswith("_") and not s.startswith("loop_"):
                rows.append(s.split())
            elif inloop and rows and (not s or s.startswith("loop_") or s.startswith("_")): break
        ti = hdr.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in hdr else 0
        els = [re.sub(r"[^A-Za-z]", "", r[ti]) for r in rows if len(r) >= 4]
        a, b, cc = c["a"], c["b"], c["c"]; al, be, ga = [math.radians(c[k]) for k in ("alpha", "beta", "gamma")]
        V = a*b*cc*math.sqrt(1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2 + 2*math.cos(al)*math.cos(be)*math.cos(ga))
        mass = sum(Element(e).atomic_mass for e in els)
        # perpendicular widths
        ax = np.array([a, 0, 0]); bx = np.array([b*math.cos(ga), b*math.sin(ga), 0])
        cx = cc*math.cos(be); cy = cc*(math.cos(al) - math.cos(be)*math.cos(ga))/math.sin(ga); cz = math.sqrt(max(cc**2 - cx**2 - cy**2, 1e-9))
        L = np.array([ax, bx, [cx, cy, cz]])
        w = [V / np.linalg.norm(np.cross(L[(i+1) % 3], L[(i+2) % 3])) for i in range(3)]
        met = sorted({e for e in els if Element(e).is_metal or e in ("Si", "Ge", "B", "As", "Sb", "Te")} - {"B", "Si"})
        return dict(file=fn, n_atoms=len(els), volume=V, density=mass / V * 1.66054, metals="-".join(met),
                    elements="-".join(sorted(set(els))), cells=" ".join(str(math.ceil(25.6 / x)) for x in w), H=("H" in els))
    except Exception as e:
        return dict(file=fn, error=str(e)[:60])
if __name__ == "__main__":
    fs = sorted(f for f in os.listdir(D) if f.endswith(".cif"))
    with Pool(2) as p: rows = p.map(parse, fs, chunksize=50)
    pd.DataFrame(rows).to_csv(OUT, index=False); print(len(rows))
