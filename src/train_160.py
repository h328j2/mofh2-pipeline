"""Predict 160 K / 5 bar uptake (DOE discharge) on real MOFs, with and without Widom binding features.
Train: CoRE-2019 'train' split + 58 library MOFs. Test: CoRE-2019 'test' split (75) and 11 external CoRE/hMOF.
Then DOE capacity = GNN 77 K/100 bar prediction - predicted 160 K/5 bar, checked on 30 test MOFs with 77 K GCMC."""
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
torch.manual_seed(0); np.random.seed(0)
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
    if f.get("status") != "ok": continue
    recs.append(dict(set="CoRE19-" + r.split, name=r.raspa, feat=f, y=gl(f"{S}/runs_160/{r.raspa}_5"), y77=gl(f"{S}/runs_m2/{r.raspa}_100")))
lib = pd.read_csv(S + "/real_picks.csv")
for n in lib[lib.status == "ok"].name:
    recs.append(dict(set="library", name=n, feat=pickle.load(open(f"{S}/real_cache/{n}.pkl", "rb")), y=gl(f"{S}/runs_160/{n}_5"), y77=np.nan))
ext = ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1", "CoRE_2007_Zn_pcu_3_ASR_3",
       "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2", "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
for n in ext:
    f = P._featurise(f"{S}/raspa/share/raspa/structures/cif/{n}.cif")
    recs.append(dict(set="external", name=n, feat=f, y=gl(f"{S}/runs_160/{n}_5"), y77=gl(f"{S}/runs_m2/{n}_100")))
df = pd.DataFrame(recs)
w = [widom(f"{S}/runs_w160/{n}") or {} for n in df.name]
df["kH"] = [x.get("kH", np.nan) for x in w]; df["qst0"] = [x.get("qst0", np.nan) for x in w]
df = df.dropna(subset=["y", "kH"]).reset_index(drop=True)
df["logkH"] = np.log10(df.kH)
df["density"] = [float(f["density"]) for f in df.feat]
df["henry_gL"] = df.kH * 5e5 * df.density * 2.016          # Henry's law at 5 bar, mol/kg -> g/L
print(df.set.value_counts().to_dict())
graphs = []
for f in df.feat:
    g = Data(x=torch.tensor(f["x"], dtype=torch.float), edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(f["dists"], dtype=torch.float)); g.gfeat = torch.tensor([[float(f[c]) for c in P.GLOBALS]], dtype=torch.float); graphs.append(g)
G = np.array([[float(f[c]) for c in P.GLOBALS] for f in df.feat]); Wf = df[["logkH", "qst0"]].values; Y = df.y.values
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
df["gnn77_y100"] = P77[:, 1]
def finetune(tr, te, use_w, epochs=300):
    Wn = torch.tensor((Wf - Wf[tr].mean(0)) / Wf[tr].std(0), dtype=torch.float)
    ymu, ysd = Y[tr].mean(), Y[tr].std(); yt = torch.tensor((Y - ymu) / ysd, dtype=torch.float)[:, None]
    preds = []
    for seed, net0 in enumerate(base):
        torch.manual_seed(seed)
        gm = copy.deepcopy(net0.global_mlp); hd = copy.deepcopy(net0.head); hd[5] = nn.Linear(64, 1)
        if use_w:                                   # widen first head layer by 2 inputs, new columns start at zero
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
                idx = perm[k:k+32]; opt.zero_grad(); loss = ((f(idx) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
        gm.eval(); hd.eval()
        with torch.no_grad(): preds.append((f(torch.tensor(te)) * ysd + ymu).numpy()[:, 0])
    return np.mean(preds, 0)
def fit(method, tr, te):
    if method == "henry": return df.henry_gL.values[te]
    if method == "rule": return np.full(len(te), np.nan)
    if method.startswith("GBM"):
        X = np.c_[G, Wf] if method.endswith("+W") else G
        return GradientBoostingRegressor(random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8).fit(X[tr], Y[tr]).predict(X[te])
    if method.startswith("probe"):
        X = np.c_[EMB, P77, Wf] if method.endswith("+W") else np.c_[EMB, P77]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], Y[tr]).predict(X[te])
    if method.startswith("finetune"): return finetune(tr, te, method.endswith("+W"))
methods = ["henry", "GBM", "GBM+W", "probe", "probe+W", "finetune", "finetune+W"]
tr = np.where(df.set.isin(["CoRE19-train", "library"]))[0]
tests = {"CoRE-2019 test (75)": np.where(df.set == "CoRE19-test")[0], "external CoRE/hMOF (11)": np.where(df.set == "external")[0]}
allte = np.r_[tuple(tests.values())]
rows, PRED = [], {}
for m in methods:
    p = np.clip(fit(m, tr, allte), 0, None); PRED[m] = dict(zip(allte, p))
    for tn, te in tests.items():
        pp = np.array([PRED[m][i] for i in te])
        rows.append(dict(method=m, test=tn, n=len(te), MAE=mean_absolute_error(Y[te], pp), bias=float(np.mean(pp - Y[te])),
                         R2=r2_score(Y[te], pp), spearman=spearmanr(Y[te], pp)[0]))
    print(m, "done", flush=True)
res = pd.DataFrame(rows); pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
res.to_csv(S + "/train_160_results.csv", index=False)
# DOE capacity on test MOFs that have 77 K / 100 bar GCMC
d = df.loc[allte].copy()
for m in methods: d["p160_" + m] = [PRED[m][i] for i in d.index]
d = d.dropna(subset=["y77"]); d["doe_gcmc"] = d.y77 - d.y
rows = []
cands = {"77 K prediction - 1.54 g/L (rule)": d.gnn77_y100 - 1.54}
for m in methods: cands[f"77 K prediction - {m}"] = d.gnn77_y100 - d["p160_" + m]
cands["77 K GCMC - best 160 K model (isolates the 160 K term)"] = d.y77 - d["p160_finetune+W"]
for k, v in cands.items():
    rows.append(dict(estimate=k, n=len(d), MAE=float(np.mean(np.abs(v - d.doe_gcmc))), bias=float(np.mean(v - d.doe_gcmc)), spearman=spearmanr(v, d.doe_gcmc)[0]))
doe = pd.DataFrame(rows); print(doe.round(3).to_string(index=False)); doe.to_csv(S + "/train_160_doe.csv", index=False)
print("77 K GNN error on these:", round(float(np.mean(np.abs(d.gnn77_y100 - d.y77))), 2), "bias", round(float(np.mean(d.gnn77_y100 - d.y77)), 2))
d.drop(columns=["feat"]).to_csv(S + "/train_160_test_predictions.csv", index=False)
print("label range 160 K/5 bar: %.2f-%.2f, DOE GCMC %.1f-%.1f" % (Y.min(), Y.max(), d.doe_gcmc.min(), d.doe_gcmc.max()))
