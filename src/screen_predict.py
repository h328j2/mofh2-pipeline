"""Screen CoRE 2019: GNN 77 K uptake (2/100 bar, DC) + 160 K/5 bar discharge term -> DOE capacity.
stage 1: 160 K term from the structure-only fine-tuned GAT (test MAE 0.63 g/L).
stage 2 (argv 'refine'): for MOFs with a Widom run, 160 K term from fine-tuned GAT + Widom inputs (test MAE 0.15 g/L)."""
import glob, os, sys, copy, pickle, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
from mofnet import MOFNet
from extract import parse_data
from widom import widom
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
REFINE = len(sys.argv) > 1 and sys.argv[1] == "refine"
def gl(d):
    f = glob.glob(f"{d}/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan
# ---- labelled real MOFs (training for the 160 K term) ----
lab = []
for _, r in pd.read_csv(S + "/core_stageB.csv").iterrows():
    p = f"{S}/core_cache/{r.raspa}.pkl"
    if os.path.exists(p): lab.append(dict(name=r.raspa, feat=pickle.load(open(p, "rb"))))
lib = pd.read_csv(S + "/real_picks.csv")
for n in lib[lib.status == "ok"].name: lab.append(dict(name=n, feat=pickle.load(open(f"{S}/real_cache/{n}.pkl", "rb"))))
ext = ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1", "CoRE_2007_Zn_pcu_3_ASR_3",
       "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2", "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
for n in ext: lab.append(dict(name=n, feat=P._featurise(f"{S}/raspa/share/raspa/structures/cif/{n}.cif")))
lab = pd.DataFrame(lab); lab = lab[[f.get("status") == "ok" for f in lab.feat]]
lab["y"] = [gl(f"{S}/runs_160/{n}_5") for n in lab.name]
w = [widom(f"{S}/runs_w160/{n}") or {} for n in lab.name]
lab["logkH"] = [np.log10(x["kH"]) if x else np.nan for x in w]; lab["qst0"] = [x.get("qst0", np.nan) for x in w]
lab = lab.dropna(subset=["y", "logkH"]).reset_index(drop=True); lab["set"] = "labelled"
LABFEAT = list(lab.feat); lab = lab.drop(columns=["feat"])
# ---- screening set ----
sl = pd.read_csv(S + "/screen_list.csv")
scr = []
for _, r in sl.iterrows():
    p = f"{S}/screen_cache/{r['name']}.pkl"
    if not os.path.exists(p): continue
    f = pickle.load(open(p, "rb"))
    if f.get("status") == "ok" and len(f["edges"]):
        scr.append(dict(name=r["name"], raspa="C19_" + "".join(c if c.isalnum() else "_" for c in r["name"]), path=p, cells=r.cells,
                        elements=f["elements"], **{c: float(f[c]) for c in ["density", "void_fraction", "lcd", "pld", "surface_area_m2g", "surface_area_m2cm3"]}))
    del f
scr = pd.DataFrame(scr); scr["set"] = "screen"
if REFINE:
    w = [widom(f"{S}/runs_w160/{n}") or {} for n in scr.raspa]
    scr["logkH"] = [np.log10(x["kH"]) if x else np.nan for x in w]; scr["qst0"] = [x.get("qst0", np.nan) for x in w]
else:
    scr["logkH"] = np.nan; scr["qst0"] = np.nan
df = pd.concat([lab, scr], ignore_index=True)
print("labelled", len(lab), "| screening", len(scr), "| with Widom", int(scr.logkH.notna().sum()), flush=True)
ckpts = sorted(glob.glob(P.REPO + "/models/gnn_full_random_s*.pth") + glob.glob(P.REPO + "/results/gnn_full_random_s*.pth"))
base = []
for c in ckpts:
    n = MOFNet(); n.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); n.eval(); base.append(n)
def to_graph(f):
    g = Data(x=torch.tensor(f["x"], dtype=torch.float), edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(f["dists"], dtype=torch.float)); g.gfeat = torch.tensor([[float(f[c]) for c in P.GLOBALS]], dtype=torch.float); return g
def feats():
    for f in LABFEAT: yield f
    for pth in scr.path: yield pickle.load(open(pth, "rb"))
POOL = [[] for _ in base]; ENS = [[] for _ in base]
def run_chunk(gs):
    with torch.no_grad():
        for k, net in enumerate(base):
            for b in DataLoader(gs, batch_size=32):
                x = (b.x - net.x_mu) / net.x_sd; ea = b.edge_attr[:, 0:1] / 4.5
                x = F.elu(net.conv1(x, b.edge_index, edge_attr=ea)); x = F.elu(net.conv2(x, b.edge_index, edge_attr=ea)); x = net.conv3(x, b.edge_index, edge_attr=ea)
                gf = (b.gfeat.view(-1, net.n_globals) - net.g_mu) / net.g_sd; pooled = global_mean_pool(x, b.batch)
                POOL[k].append(torch.cat([pooled, gf], 1)); ENS[k].append((net.head(torch.cat([pooled, net.global_mlp(gf)], 1)) * net.y_sd + net.y_mu).numpy())
chunk = []
for i, f in enumerate(feats()):
    chunk.append(to_graph(f))
    if len(chunk) == 256: run_chunk(chunk); chunk = []; print("embedded", i + 1, flush=True)
if chunk: run_chunk(chunk)
POOL = [torch.cat(x) for x in POOL]; ENS = [np.concatenate(x) for x in ENS]
ENS = np.clip(np.stack(ENS), 0, None)
for j, t in enumerate(["y2", "y100", "dc"]):
    df[f"gnn77_{t}"] = ENS[:, :, j].mean(0); df[f"gnn77_{t}_std"] = ENS[:, :, j].std(0)
Wf = df[["logkH", "qst0"]].values; Y = df.y.values
def finetune(tr, te, use_w, epochs=300):
    Wn = torch.tensor(np.nan_to_num((Wf - Wf[tr].mean(0)) / Wf[tr].std(0)), dtype=torch.float)
    ymu, ysd = Y[tr].mean(), Y[tr].std(); yt = torch.tensor(np.nan_to_num((Y - ymu) / ysd), dtype=torch.float)[:, None]
    preds = []
    for seed, net0 in enumerate(base):
        torch.manual_seed(seed)
        gm = copy.deepcopy(net0.global_mlp); hd = copy.deepcopy(net0.head); hd[5] = nn.Linear(64, 1)
        if use_w:
            old = hd[0]; new = nn.Linear(old.in_features + 2, old.out_features)
            with torch.no_grad(): new.weight.zero_(); new.weight[:, :old.in_features] = old.weight; new.bias.copy_(old.bias)
            hd[0] = new
        opt = torch.optim.Adam(list(gm.parameters()) + list(hd.parameters()), lr=1e-3, weight_decay=1e-4)
        Xp = POOL[seed]
        f = lambda idx: hd(torch.cat([Xp[idx, :256], gm(Xp[idx, 256:])] + ([Wn[idx]] if use_w else []), 1))
        trt = torch.tensor(tr)
        for ep in range(epochs):
            gm.train(); hd.train(); perm = trt[torch.randperm(len(tr))]
            for k in range(0, len(tr), 32):
                idx = perm[k:k+32]; opt.zero_grad(); ((f(idx) - yt[idx]) ** 2).mean().backward(); opt.step()
        gm.eval(); hd.eval()
        with torch.no_grad():
            out = []
            for k in range(0, len(te), 2048): out.append((f(torch.tensor(te[k:k+2048])) * ysd + ymu).numpy()[:, 0])
            preds.append(np.concatenate(out))
    return np.clip(np.mean(preds, 0), 0, None), np.std(preds, 0)
tr = np.where(df.set == "labelled")[0]; te = np.where(df.set == "screen")[0]
p, s = finetune(tr, te, False); df.loc[te, "p160"] = p; df.loc[te, "p160_std"] = s; df.loc[te, "p160_source"] = "structure-only GAT"
if REFINE:
    tw = te[~np.isnan(Wf[te, 0])]
    p, s = finetune(tr, tw, True); df.loc[tw, "p160"] = p; df.loc[tw, "p160_std"] = s; df.loc[tw, "p160_source"] = "GAT + Widom"
out = df.loc[te].copy()
out["metals"] = [",".join(sorted(set(e) - {"H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B", "Si", "Se"})) for e in out.elements]
SEEN = {"H", "C", "N", "O", "F", "Cl", "Br", "Zn", "Cu", "Zr", "V"}
gmu, gsd = base[0].g_mu.numpy(), base[0].g_sd.numpy()
out["max_abs_z"] = np.abs((out[P.GLOBALS].values.astype(float) - gmu) / gsd).max(1)
out["unseen_el"] = [",".join(sorted(set(e) - SEEN)) for e in out.elements]
out["in_domain"] = (out.max_abs_z < 3) & (out.unseen_el == "")
out["doe_dc"] = out.gnn77_y100 - out.p160
out = out.drop(columns=["path", "set", "y", "elements"], errors="ignore").sort_values("doe_dc", ascending=False)
out.to_csv(S + ("/screen_core2019_refined.csv" if REFINE else "/screen_core2019.csv"), index=False)
pd.set_option("display.width", 220)
cols = ["name", "metals", "density", "void_fraction", "lcd", "gnn77_y100", "gnn77_dc", "p160", "doe_dc", "gnn77_dc_std", "in_domain"]
print(out[cols].head(25).round(2).to_string(index=False))
print(out[["gnn77_y100", "gnn77_dc", "p160", "doe_dc"]].describe().round(2).to_string())
print("in domain:", int(out.in_domain.sum()), "of", len(out))
