"""Physics-informed head on the frozen 77 K GAT: the network predicts Langmuir parameters (Qst, dS, n_sat) per MOF and
uptake is computed analytically, n(P,T) = n_sat K P / (1 + K P), K = exp(Qst/RT + dS/R) / (1 bar).
Training signal: 77 K GNN predictions (2, 100 bar; pseudo-labels), 130 K GCMC (5, 100 bar), Widom Henry coefficient at 160 K,
and optionally k labels of 160 K / 5 bar. Test: 160 K / 5 bar on held-out CoRE-2019 (75) + external (11), and DOE capacity."""
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
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
R = 8.314462618e-3
def gl(d):
    f = glob.glob(f"{d}/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan
recs = []
B = pd.read_csv(S + "/core_stageB.csv")
for _, r in B.iterrows():
    cp = f"{S}/core_cache/{r.raspa}.pkl"
    if not os.path.exists(cp): continue
    f = pickle.load(open(cp, "rb"))
    if f.get("status") == "ok": recs.append(dict(set="CoRE19-" + r.split, name=r.raspa, feat=f))
lib = pd.read_csv(S + "/real_picks.csv")
for n in lib[lib.status == "ok"].name: recs.append(dict(set="library", name=n, feat=pickle.load(open(f"{S}/real_cache/{n}.pkl", "rb"))))
ext = ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1", "CoRE_2007_Zn_pcu_3_ASR_3",
       "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2", "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
for n in ext: recs.append(dict(set="external", name=n, feat=P._featurise(f"{S}/raspa/share/raspa/structures/cif/{n}.cif")))
for _, r in pd.read_csv(S + "/t130_picks.csv").iterrows():
    cp = f"{S}/topo_cache/{r['name']}.pkl"
    if os.path.exists(cp):
        f = pickle.load(open(cp, "rb"))
        if f.get("status") == "ok": recs.append(dict(set="ToBaCCo", name=r.raspa, feat=f))
df = pd.DataFrame(recs)
df = df[[f.get("status") == "ok" and len(f["edges"]) > 0 for f in df.feat]].reset_index(drop=True)
df["y160"] = [gl(f"{S}/runs_160/{n}_5") for n in df.name]
df["y77_100"] = [gl(f"{S}/runs_m2/{n}_100") for n in df.name]
df["y130_5"] = [gl(f"{S}/runs_130/{n}_5") for n in df.name]
df["y130_100"] = [gl(f"{S}/runs_130/{n}_100") for n in df.name]
w = [widom(f"{S}/runs_w160/{n}") or {} for n in df.name]
df["kH"] = [x.get("kH", np.nan) for x in w]; df["qst0"] = [x.get("qst0", np.nan) for x in w]
df["density"] = [float(f["density"]) for f in df.feat]; df["vf"] = [float(f["void_fraction"]) for f in df.feat]
df["henry_gL_bar"] = df.kH * 1e5 * df.density * 2.016
print(df.set.value_counts().to_dict(), flush=True)
graphs = []
for f in df.feat:
    g = Data(x=torch.tensor(f["x"], dtype=torch.float), edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(f["dists"], dtype=torch.float)); g.gfeat = torch.tensor([[float(f[c]) for c in P.GLOBALS]], dtype=torch.float); graphs.append(g)
ckpts = sorted(glob.glob(P.REPO + "/models/gnn_full_random_s*.pth") + glob.glob(P.REPO + "/results/gnn_full_random_s*.pth"))
base = []
for c in ckpts:
    n = MOFNet(); n.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); n.eval(); base.append(n)
POOL, EMB, P77 = [], [], []
with torch.no_grad():
    for net in base:
        ps, es, pr = [], [], []
        for b in DataLoader(graphs, batch_size=32):
            x = (b.x - net.x_mu) / net.x_sd; ea = b.edge_attr[:, 0:1] / 4.5
            x = F.elu(net.conv1(x, b.edge_index, edge_attr=ea)); x = F.elu(net.conv2(x, b.edge_index, edge_attr=ea)); x = net.conv3(x, b.edge_index, edge_attr=ea)
            gf = (b.gfeat.view(-1, net.n_globals) - net.g_mu) / net.g_sd; pooled = global_mean_pool(x, b.batch)
            h = torch.cat([pooled, net.global_mlp(gf)], 1); h2 = net.head[4](net.head[3](net.head[1](net.head[0](h))))
            ps.append(torch.cat([pooled, gf], 1)); es.append(h2); pr.append(net.head[5](h2) * net.y_sd + net.y_mu)
        POOL.append(torch.cat(ps)); EMB.append(torch.cat(es).numpy()); P77.append(torch.cat(pr).numpy())
EMB = np.concatenate(EMB, 1); P77 = np.clip(np.mean(P77, 0), 0, None)
df["gnn77_y2"], df["gnn77_y100"] = P77[:, 0], P77[:, 1]
VF = torch.tensor(df.vf.values, dtype=torch.float).clamp(0.05, 1)

BULK = os.environ.get("PI_BULK") == "1"
RHO = {(77.0, 2.0): 0.62, (77.0, 100.0): 30.5, (130.0, 5.0): 0.93, (130.0, 100.0): 16.41, (160.0, 5.0): 0.76}   # matched-FF bulk H2, g/L (RASPA box)
def rho_bulk(T, Pbar):
    return torch.tensor([RHO.get((float(t), float(p)), 0.0) for t, p in zip(T, Pbar)], dtype=torch.float)
def langmuir(z, T, Pbar, idx):
    Q = 2.0 + F.softplus(z[:, 0])                          # kJ/mol, > 2
    dS = -(40.0 + F.softplus(z[:, 1]) * 20.0) * 1e-3       # kJ/mol/K, < -0.040
    nsat = 71.0 * VF[idx] * torch.sigmoid(z[:, 2] + 1.0)   # g/L, bounded by liquid H2 density x void fraction
    K = torch.exp(Q / (R * T) + dS / R)                    # 1/bar
    n = nsat * K * Pbar / (1 + K * Pbar); slope = nsat * K
    if BULK:                                               # total = adsorbed excess + bulk-like gas in the pore volume
        n = n + VF[idx] * rho_bulk(T, Pbar); slope = slope + VF[idx] * 2.016 / (0.0831446 * T)
    return n, slope, Q
def obs_table(tr, k160_idx):
    """(mof index, T, P, value, weight) for every training observation."""
    o = []
    for i in tr:
        r = df.loc[i]
        o += [(i, 77.0, 2.0, r.gnn77_y2, 0.5), (i, 77.0, 100.0, r.gnn77_y100, 0.5)]
        if not np.isnan(r.y77_100): o.append((i, 77.0, 100.0, r.y77_100, 1.0))
        if not np.isnan(r.y130_5): o.append((i, 130.0, 5.0, r.y130_5, 1.0))
        if not np.isnan(r.y130_100): o.append((i, 130.0, 100.0, r.y130_100, 1.0))
    for i in k160_idx: o.append((i, 160.0, 5.0, df.y160[i], 1.0))
    return np.array(o, dtype=float)
def train_pi(tr, k160_idx, te, use_henry=True, iters=int(os.environ.get("PI_ITERS", 1500))):
    O = torch.tensor(obs_table(tr, k160_idx), dtype=torch.float); oi = O[:, 0].long()
    hk = [i for i in tr if not np.isnan(df.henry_gL_bar[i])] if use_henry else []
    hk_t = torch.tensor(hk, dtype=torch.long); hv = torch.tensor(np.log(df.henry_gL_bar.values[hk]), dtype=torch.float)
    preds, params = [], []
    for seed, net0 in enumerate(base):
        torch.manual_seed(seed)
        gm = copy.deepcopy(net0.global_mlp); hd = copy.deepcopy(net0.head); hd[5] = nn.Linear(64, 3)
        opt = torch.optim.Adam(list(gm.parameters()) + list(hd.parameters()), lr=1e-3, weight_decay=1e-4)
        Xp = POOL[seed]; zf = lambda idx: hd(torch.cat([Xp[idx, :256], gm(Xp[idx, 256:])], 1))
        allidx = torch.tensor(sorted(set(oi.tolist()) | set(hk)), dtype=torch.long); pos = {int(j): k for k, j in enumerate(allidx)}
        oi_l = torch.tensor([pos[int(j)] for j in oi]); hk_l = torch.tensor([pos[int(j)] for j in hk], dtype=torch.long)
        for it in range(iters):
            gm.train(); hd.train(); opt.zero_grad()
            z = zf(allidx)
            n, kH, _ = langmuir(z[oi_l], O[:, 1], O[:, 2], allidx[oi_l])
            loss = (O[:, 4] * F.smooth_l1_loss(n, O[:, 3], reduction="none", beta=1.0)).mean()
            if len(hk):
                _, kH160, _ = langmuir(z[hk_l], torch.full((len(hk),), 160.0), torch.zeros(len(hk)), allidx[hk_l])
                loss = loss + 0.5 * F.smooth_l1_loss(torch.log(kH160), hv)
            loss.backward(); opt.step()
        gm.eval(); hd.eval()
        with torch.no_grad():
            tt = torch.tensor(te, dtype=torch.long); z = zf(tt)
            p160, _, Q = langmuir(z, torch.full((len(te),), 160.0), torch.full((len(te),), 5.0), tt)
            p77, _, _ = langmuir(z, torch.full((len(te),), 77.0), torch.full((len(te),), 100.0), tt)
            preds.append(np.c_[p160.numpy(), p77.numpy(), Q.numpy()])
    return np.mean(preds, 0)
# ---- data-driven comparisons with the same 160 K label subsets ----
G = np.array([[float(f[c]) for c in P.GLOBALS] for f in df.feat])
def dd(method, tr, te):
    Y = df.y160.values
    if method == "henry (Widom, no ML)": return (df.henry_gL_bar.values * 5.0)[te]
    if method == "GBM descriptors":
        return GradientBoostingRegressor(random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8).fit(G[tr], Y[tr]).predict(G[te])
    if method == "linear probe":
        X = np.c_[EMB, P77]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], Y[tr]).predict(X[te])
core_tr = np.where((df.set.isin(["CoRE19-train", "library"])) & df.y160.notna())[0]
base_tr = np.where(df.set.isin(["CoRE19-train", "library", "ToBaCCo"]))[0]     # MOFs whose 77 K / 130 K / Widom info may be used
tests = {"CoRE-2019 test": np.where((df.set == "CoRE19-test") & df.y160.notna())[0], "external": np.where((df.set == "external") & df.y160.notna())[0]}
allte = np.r_[tuple(tests.values())]
rng = np.random.default_rng(0); order = rng.permutation(core_tr)
rows, keep = [], {}
KS = [int(x) for x in os.environ["PI_KS"].split(",")] if os.environ.get("PI_KS") else ([0, 25] if os.environ.get("PI_ITERS") else [0, 25, 50, 100, 200, len(core_tr)])
for k in [min(x, len(core_tr)) for x in KS]:
    sub = order[:k]
    cands = {"PI head (Langmuir)": train_pi(base_tr, sub, allte)[:, 0]}
    if k == 0: cands["henry (Widom, no ML)"] = dd("henry (Widom, no ML)", None, allte)
    if k >= 25:
        for m in ["GBM descriptors", "linear probe"]: cands[m] = dd(m, sub, allte)
    for m, p in cands.items():
        p = np.clip(p, 0, None); keep[(m, k)] = p
        for tn, te in tests.items():
            sel = np.isin(allte, te); y = df.y160.values[te]; pp = p[sel]
            rows.append(dict(k_160K_labels=k, method=m, test=tn, n=len(te), MAE=mean_absolute_error(y, pp), bias=float(np.mean(pp - y)), spearman=spearmanr(y, pp)[0]))
    print("k =", k, "done", flush=True)
res = pd.DataFrame(rows); pd.set_option("display.width", 200); print(res.round(3).to_string(index=False)); res.to_csv(S + ("/pi_head_bulk_results.csv" if BULK else "/pi_head_results.csv"), index=False)
# DOE capacity on test MOFs with 77 K / 100 bar GCMC: PI head gives both terms itself
full = train_pi(base_tr, order, allte)
d = df.loc[allte].copy(); d["pi160"], d["pi77"], d["piQ"] = full[:, 0], full[:, 1], full[:, 2]
d = d[d.y77_100.notna()]; d["doe"] = d.y77_100 - d.y160
for k, v in {"rule: GNN 77 K - 1.54": d.gnn77_y100 - 1.54, "GNN 77 K - PI 160 K": d.gnn77_y100 - d.pi160, "PI 77 K - PI 160 K (one model)": d.pi77 - d.pi160}.items():
    print(f"DOE  {k:32s} n={len(d)}  MAE {np.mean(np.abs(v - d.doe)):.2f}  bias {np.mean(v - d.doe):+.2f}  spearman {spearmanr(v, d.doe)[0]:.2f}")
print("PI 77 K/100 bar vs GCMC on these: MAE %.2f" % np.mean(np.abs(d.pi77 - d.y77_100)), " | Qst(PI) vs Widom qst0 r = %.2f" % np.corrcoef(d.piQ, d.qst0)[0, 1])
d.drop(columns=["feat"]).to_csv(S + ("/pi_head_bulk_doe.csv" if BULK else "/pi_head_doe.csv"), index=False)
