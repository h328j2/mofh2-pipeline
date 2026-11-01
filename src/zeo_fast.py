"""Faster Zeo++ descriptors (fewer MC samples, no CIF rewrite) - validated against the accurate pipeline."""
import os, re, subprocess, tempfile, sys, time
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from predict_topo import ZEO
KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")
def zeo_fast(cif, ha="-ha", sa_n=500, vol_n=5000):
    with tempfile.TemporaryDirectory() as t:
        out = {}
        subprocess.run(["prlimit", "--as=1500000000", ZEO, ha, "-res", f"{t}/s.res", cif], capture_output=True, timeout=300)
        tok = open(f"{t}/s.res").read().split(); out["lcd_zeo"], out["pld_zeo"] = float(tok[1]), float(tok[2])
        subprocess.run(["prlimit", "--as=1500000000", ZEO, ha, "-sa", "1.86", "1.86", str(sa_n), f"{t}/s.sa", cif], capture_output=True, timeout=300)
        out["sa_m2g_zeo"] = float(dict(KV.findall(open(f"{t}/s.sa").readline()))["ASA_m^2/g"])
        subprocess.run(["prlimit", "--as=1500000000", ZEO, ha, "-vol", "0.0", "0.0", str(vol_n), f"{t}/s.vol", cif], capture_output=True, timeout=300)
        out["vf_zeo_0.0"] = float(dict(KV.findall(open(f"{t}/s.vol").readline()))["AV_Volume_fraction"])
        return out
if __name__ == "__main__":
    import pickle, pandas as pd, numpy as np, glob
    fs = sorted(glob.glob(S + "/core_cache/*.pkl"))[:24]
    rows = []
    for ha in ["-ha", "-nomax"]:
        for p in fs:
            f = pickle.load(open(p, "rb")); n = os.path.basename(p)[:-4]
            t0 = time.time()
            try: z = zeo_fast(f"{S}/raspa/share/raspa/structures/cif/{n}.cif", ha=ha if ha == "-ha" else "-r")
            except Exception as e: continue
            rows.append(dict(mode=ha, name=n, t=time.time() - t0, **{k + "_fast": v for k, v in z.items()}, **{k: f["raw_" + k] for k in z}))
    d = pd.DataFrame(rows)
    for m, g in d.groupby("mode"):
        print(m, "sec/structure %.2f" % g.t.mean())
        for k in ["lcd_zeo", "pld_zeo", "sa_m2g_zeo", "vf_zeo_0.0"]:
            print(f"   {k:12s} MAE {np.mean(np.abs(g[k+'_fast']-g[k])):.3f}  r {np.corrcoef(g[k+'_fast'], g[k])[0,1]:.4f}  mean {g[k].mean():.3f}")
