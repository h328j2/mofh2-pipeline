"""Convert simulated 77 K isotherms (absolute g/L) to excess and wt% for comparison with experiment; MOF-5 DOE swing check."""
import glob, re, os, sys, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from extract import parse_data
from pymatgen.core import Structure
import CoolProp.CoolProp as C
def gl(d):
    f = glob.glob(f"{d}/Output/System_0/*.data")
    if not f: return float("nan")
    r = parse_data(f[0]); return r["gL"] if r.get("finished") else float("nan")
def he(m):
    f = glob.glob(f"{S}/runs/he_{m}/Output/System_0/*.data")
    return float(re.search(r"\[helium\] Average Widom Rosenbluth-weight:\s+([\d.]+)", open(f[0]).read()).group(1))
rows = []
for m in ["IRMOF-1", "UIO-66", "Zn-DOBDC", "Cu-BTC"]:
    rho = Structure.from_file(f"{S}/raspa/share/raspa/structures/cif/{m}.cif").density; phi = he(m)
    for P in (1, 5, 10, 20, 40, 70, 100):
        a = gl(f"{S}/runs_exp/{m}_{P}"); rb = C.PropsSI("D", "T", 77, "P", P * 1e5, "Hydrogen"); ex = a - phi * rb
        rows.append(dict(mof=m, rho=round(float(rho), 3), he_void=phi, P=P, abs_gL=a, excess_gL=ex,
                         excess_wt=100 * (ex / rho) / (1000 + ex / rho), total_wt=100 * (a / rho) / (1000 + a / rho)))
d = pd.DataFrame(rows); pd.set_option("display.width", 200); print(d.round(2).to_string(index=False)); d.to_csv(S + "/exp_gcmc_isotherms.csv", index=False)
print("\nExperiment (literature): MOF-5 excess max 7.1 wt% (~40 bar), total 66 g/L at 100 bar [Kaye 2007]; UiO-66 35 g/L total, 2.1 wt% excess at 100 bar [Bambalaza 2018];")
print("Zn-MOF-74 2.3 wt% excess saturation at 26 bar [Wong-Foy 2006]; HKUST-1 1.8-2.4 wt% at 1 bar [2023 studies]")
m5 = d[d.mof == "IRMOF-1"].set_index("P"); g160 = gl(f"{S}/runs_160/IRMOF-1_5")
print(f"\nMOF-5 DOE swing (77 K/100 bar -> 160 K/5 bar): GCMC {m5.abs_gL[100] - g160:.1f} g/L vs measured 51.9 g/L [Ahmed et al. 2017]")
print(f"MOF-5 pressure swing (77 K, 100 -> 5 bar): GCMC {m5.abs_gL[100] - m5.abs_gL[5]:.1f} g/L vs measured 31.1, their GCMC 32.5 g/L")
