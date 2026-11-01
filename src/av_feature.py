"""H2-probe (r = 1.45 A) accessible volume fraction, fast Zeo++ (no -ha), for MOFs used in 77 K fine-tuning / screening."""
import os, re, sys, subprocess, tempfile, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from predict_topo import ZEO
KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")
def av(item):
    raspa, cif = item
    try:
        with tempfile.TemporaryDirectory() as t:
            subprocess.run(["prlimit", "--as=1500000000", ZEO, "-vol", "1.45", "1.45", "5000", f"{t}/v", cif], capture_output=True, timeout=300)
            kv = dict(KV.findall(open(f"{t}/v").read().replace("\n", " ")))
            return raspa, float(kv["AV_Volume_fraction"]), float(kv["NAV_Volume_fraction"])
    except Exception: return raspa, float("nan"), float("nan")
if __name__ == "__main__":
    which = sys.argv[1]
    CIF = S + "/raspa/share/raspa/structures/cif"
    if which == "ft":
        names = list(pd.read_csv(S + "/ft77_train.csv").raspa) + list(pd.read_csv(S + "/core_stageB.csv").query("g77").raspa) + \
                list(pd.read_csv(S + "/screen_core2019_refined.csv").head(10).raspa) + \
                ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1", "CoRE_2007_Zn_pcu_3_ASR_3",
                 "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2", "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
        items = [(n, f"{CIF}/{n}.cif") for n in names]; out = S + "/av_ft.csv"
    else:
        sc = pd.read_csv(S + "/screen_core2019_refined.csv"); sl = pd.read_csv(S + "/screen_list.csv").set_index("name")
        items = [(r, f"{S}/coredl/MOFSimplify/CoRE2019/{sl.loc[n, 'file']}") for n, r in zip(sc.name, sc.raspa)]; out = S + "/av_screen.csv"
    with Pool(2) as p: rows = p.map(av, items, chunksize=8)
    pd.DataFrame(rows, columns=["raspa", "av_h2", "nav_h2"]).to_csv(out, index=False); print(len(rows), "done")
