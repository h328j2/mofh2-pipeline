"""Build the unified queue file for each notebook stage from the packaged selection lists.
Output: queues/<stage>.txt  (format: NAME|T|P|CYCLES|FF|CELLS|OUTDIR)"""
import os, re, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); D = S
os.makedirs(S + "/queues", exist_ok=True)
def legacy(fn):                      # "NAME|P|cells" files
    out = []
    for l in open(f"{D}/{fn}"):
        l = l.strip()
        if l: n, p, u = l.split("|"); out.append((n, p, u))
    return out
def gen(fn):                         # rq_gen format "KIND|NAME|CELLS|OUTDIR"
    out = []
    for l in open(f"{D}/{fn}"):
        l = l.strip()
        if l: k, n, u, o = l.split("|"); out.append((k, n, u, re.search(r"(runs[^/]*/[^/]+)/?$", o).group(1)))
    return out
KIND = {"w160": ("160", "W", 1000), "g160": ("160", "5", 1000), "g77": ("77", "100", 500), "g77t": ("77", "100", 400),
        "g77s": ("77", "100", 400), "g77_2s": ("77", "2", 500), "g77_2": ("77", "2", 1000)}
Q = {}
# 02: first setup (UFF_FH) hMOF + CoRE-2024, matched (UFF_FHm2), classical (UFF_noFHH2)
Q["02_convention_core2024"] = [f"{n}|77|{p}|2000|UFF_FH|{u}|runs_fh/{n}_{p}" for n, p, u in legacy("queue_fh.txt")] + \
    [f"{n}|77|{p}|1000|UFF_FH|{u}|runs_fh/{n}_{p}" for n, p, u in legacy("queue_core.txt")] + \
    [f"{n}|77|{p}|1000|UFF_FHm2|{u}|runs_m2/{n}_{p}" for n, p, u in legacy("q_m2.txt")] + \
    [f"{n}|77|{p}|1000|UFF_noFHH2|{u}|runs_noFH/{n}_{p}" for n, p, u in legacy("q_noFH.txt")]
# 03: ToBaCCo checks
Q["03_topologies"] = [f"{n}|77|{p}|1000|UFF_FH|{u}|runs_topo/{n}_{p}" for n, p, u in legacy("queue_topo.txt")] + \
    [f"{n}|77|{p}|1000|UFF_FHm2|{u}|runs_m2/{n}_{p}" for n, p, u in legacy("q_m2topo.txt")]
# 04: 160 K swing test, 130 K labels (generated + external), 130 K real library MOFs
Q["04_beyond77_130K"] = [f"{n}|160|{p}|1000|UFF_FHm2|{u}|runs_160/{n}_{p}" for n, p, u in legacy("q_160.txt")] + \
    [f"{n}|130|{p}|400|UFF_FHm2|{u}|runs_130/{n}_{p}" for n, p, u in sorted(set(legacy("q_130.txt") + legacy("q_130b.txt")))] + \
    [f"{n}|130|{p}|400|UFF_FHm2|{u}|runs_130/{n}_{p}" for n, p, u in legacy("q_real.txt") + legacy("q_real2.txt")]
def unify(fn):
    out = []
    for k, n, u, o in gen(fn):
        T, P, C = KIND[k]; out.append(f"{n}|{T}|{P}|{C}|UFF_FHm2|{u}|{o}")
    return out
# 05: CoRE-2019 160 K dataset (Widom on 1,069, 160 K GCMC on library + 500, 77 K/100 bar on 30 test MOFs)
Q["05_core2019_160K"] = unify("q_w160.txt") + unify("q_g160_known.txt") + unify("q_stage2.txt") + unify("q_stage3.txt")
# 06: screening: top-10 GCMC + top-300 Widom, 150-MOF 77 K fine-tuning labels, then top 20 and bottom 5
Q["06a_screen_top10_widom300"] = unify("q_screenG.txt") + unify("q_screenW.txt")
Q["06b_ft77_labels"] = unify("q_ft77.txt")
Q["06c_top20_bottom5"] = unify("q_top20.txt") + unify("q_bottom5.txt")
# 07: experiment isotherms
Q["07_experiment"] = [f"{n}|77|{p}|500|UFF_FHm2|{u}|runs_exp/{n}_{p}" for n, p, u in legacy("q_exp.txt")]
for k, v in Q.items():
    open(f"{S}/queues/{k}.txt", "w").write("\n".join(v) + "\n"); print(f"{k:28s} {len(v):5d} runs")
