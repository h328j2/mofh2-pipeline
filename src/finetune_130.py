"""Fine-tune the 77 K GNN ensemble to predict 130 K uptake (5 and 100 bar).

Methods compared, all trained on the same ToBaCCo labels:
  A  descriptors only   : gradient boosting on the 6 pore descriptors (from scratch)
  B  scaling            : linear map from the GNN's own 77 K predictions
  C  linear probe       : ridge on frozen GNN embeddings (+ its 77 K predictions)
  D  fine-tuned GNN     : copy of each seed, new 2-output layer, last layers retrained; ensemble mean
Evaluation: 5-fold CV on ToBaCCo, and an external test on CoRE + hMOF (trained on all ToBaCCo).
"""
import glob, os, re, sys, copy, pickle, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import predict_topo as P
from mofnet import MOFNet
from extract import parse_data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
torch.manual_seed(0); np.random.seed(0)

def gcmc(name, p):
    f = glob.glob(f"{S}/runs_130/{name}_{p}/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan

# ---------- data ----------
picks = pd.read_csv(S + "/t130_picks.csv")
recs = []
for _, r in picks.iterrows():
    cp = f"{S}/topo_cache/{r['name']}.pkl"
    if not os.path.exists(cp): continue
    f = pickle.load(open(cp, "rb"))
    if f.get("status") != "ok": continue
    recs.append(dict(set="ToBaCCo", name=r["name"], raspa=r.raspa, feat=f, y5=gcmc(r.raspa, 5), y100=gcmc(r.raspa, 100)))
ext = ["hMOF-3001465", "hMOF-5040736", "hMOF-8081", "CoRE_2010_Zn_pyr_3_ASR_1", "CoRE_2018_Zr_scu_3_ASR_1",
       "CoRE_2007_Zn_pcu_3_ASR_3", "CoRE_2015_Zr_ftw_3_ASR_3", "CoRE_2011_Zn_pcu_3_ASR_8", "CoRE_2018_Zr_scu_3_ASR_2",
       "CoRE_2024_Zn_pcu_3_ASR_1", "CoRE_2015_Zr_ftw_3_ASR_2"]
for n in ext:
    f = P._featurise(f"{S}/raspa/share/raspa/structures/cif/{n}.cif")
    if f["status"] != "ok": print("skip", n, f["status"]); continue
    recs.append(dict(set="external", name=n, raspa=n, feat=f, y5=gcmc(n, 5), y100=gcmc(n, 100)))
df = pd.DataFrame(recs).dropna(subset=["y5", "y100"]).reset_index(drop=True)
print(f"labelled: ToBaCCo {int((df.set=='ToBaCCo').sum())}, external {int((df.set=='external').sum())}")

def to_graph(f):
    g = Data(x=torch.tensor(f["x"], dtype=torch.float), edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(f["dists"], dtype=torch.float))
    g.gfeat = torch.tensor([[f[c] for c in P.GLOBALS]], dtype=torch.float)
    return g
graphs = [to_graph(f) for f in df.feat]
G = np.array([[f[c] for c in P.GLOBALS] for f in df.feat])
Y = df[["y5", "y100"]].values

ckpts = sorted(glob.glob(P.REPO + "/models/gnn_full_random_s*.pth") + glob.glob(P.REPO + "/results/gnn_full_random_s*.pth"))
base = []
for c in ckpts:
    n = MOFNet(); n.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); n.eval(); base.append(n)

def embed(net, b):
    x = (b.x - net.x_mu) / net.x_sd; ea = b.edge_attr[:, 0:1] / 4.5
    import torch.nn.functional as F
    x = F.elu(net.conv1(x, b.edge_index, edge_attr=ea)); x = F.elu(net.conv2(x, b.edge_index, edge_attr=ea))
    x = net.conv3(x, b.edge_index, edge_attr=ea)
    gf = (b.gfeat.view(-1, net.n_globals) - net.g_mu) / net.g_sd
    h = torch.cat([global_mean_pool(x, b.batch), net.global_mlp(gf)], 1)
    h1 = net.head[1](net.head[0](h)); h2 = net.head[4](net.head[3](h1))   # 64-d penultimate
    return h2, net.head[5](h2) * net.y_sd + net.y_mu

with torch.no_grad():
    E, P77 = [], []
    for net in base:
        es, ps = [], []
        for b in DataLoader(graphs, batch_size=32):
            e, p = embed(net, b); es.append(e); ps.append(p)
        E.append(torch.cat(es).numpy()); P77.append(torch.cat(ps).numpy())
E = np.concatenate(E, 1)                    # 4 seeds x 64
P77 = np.clip(np.mean(P77, 0), 0, None)     # ensemble 77 K predictions (y2, y100, dc)

def fit_predict(method, tr, te):
    out = np.zeros((len(te), 2))
    for j in range(2):
        if method == "A":
            m = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05).fit(G[tr], Y[tr, j]); out[:, j] = m.predict(G[te])
        elif method == "B":
            X = np.c_[P77[:, :2]]; m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], Y[tr, j]); out[:, j] = m.predict(X[te])
        elif method == "C":
            X = np.c_[E, P77]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], Y[tr, j]); out[:, j] = m.predict(X[te])
    if method == "D":
        out = finetune(tr, te)
    return np.clip(out, 0, None)

# frozen graph part computed once per seed: pooled conv embedding (256-d) + normalised globals (6)
POOL = []
with torch.no_grad():
    for net in base:
        ps = []
        for b in DataLoader(graphs, batch_size=32):
            import torch.nn.functional as F
            x = (b.x - net.x_mu) / net.x_sd; ea = b.edge_attr[:, 0:1] / 4.5
            x = F.elu(net.conv1(x, b.edge_index, edge_attr=ea)); x = F.elu(net.conv2(x, b.edge_index, edge_attr=ea))
            x = net.conv3(x, b.edge_index, edge_attr=ea)
            gf = (b.gfeat.view(-1, net.n_globals) - net.g_mu) / net.g_sd
            ps.append(torch.cat([global_mean_pool(x, b.batch), gf], 1))
        POOL.append(torch.cat(ps))

def finetune(tr, te, epochs=400):
    """Same as fine-tuning the GNN with its graph layers frozen: retrain global MLP + head (new 2-output layer)."""
    preds = []
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
    yt = torch.tensor((Y[tr] - ymu) / ysd, dtype=torch.float)
    for seed, net0 in enumerate(base):
        torch.manual_seed(seed)
        gm = copy.deepcopy(net0.global_mlp); hd = copy.deepcopy(net0.head); hd[5] = nn.Linear(64, 2)
        params = list(gm.parameters()) + list(hd.parameters())
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        Xp = POOL[seed]
        f = lambda X: hd(torch.cat([X[:, :256], gm(X[:, 256:])], 1))
        Xtr = Xp[tr]
        for ep in range(epochs):
            gm.train(); hd.train()
            perm = torch.randperm(len(tr))
            for k in range(0, len(tr), 16):
                idx = perm[k:k+16]; opt.zero_grad()
                loss = ((f(Xtr[idx]) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
        gm.eval(); hd.eval()
        with torch.no_grad():
            preds.append((f(Xp[te]) * torch.tensor(ysd, dtype=torch.float) + torch.tensor(ymu, dtype=torch.float)).numpy())
    return np.mean(preds, 0)

tob = np.where(df.set == "ToBaCCo")[0]; extx = np.where(df.set == "external")[0]
names = {"A": "descriptors only (GBM)", "B": "scale 77 K predictions", "C": "linear probe on frozen GNN", "D": "fine-tuned GNN"}
rows = []
cv = {m: np.zeros((len(tob), 2)) for m in names}
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tob)):
    for m in names:
        cv[m][b] = fit_predict(m, tob[a], tob[b])
for m in names:
    for j, t in enumerate(["130 K / 5 bar", "130 K / 100 bar"]):
        rows.append(dict(method=names[m], target=t, test="5-fold CV, ToBaCCo", n=len(tob),
                         MAE=mean_absolute_error(Y[tob, j], cv[m][:, j]), R2=r2_score(Y[tob, j], cv[m][:, j])))
    pe = fit_predict(m, tob, extx)
    for j, t in enumerate(["130 K / 5 bar", "130 K / 100 bar"]):
        rows.append(dict(method=names[m], target=t, test="external: CoRE + hMOF", n=len(extx),
                         MAE=mean_absolute_error(Y[extx, j], pe[:, j]), R2=r2_score(Y[extx, j], pe[:, j])))
    if m == "D":
        ext_pred = pe
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv(S + "/finetune_130_results.csv", index=False)
out = df.loc[extx, ["name", "y5", "y100"]].copy(); out["ft_y5"], out["ft_y100"] = ext_pred[:, 0], ext_pred[:, 1]
print(out.round(2).to_string(index=False))
cvd = df.loc[tob, ["name", "y5", "y100"]].copy(); cvd["ft_y5"], cvd["ft_y100"] = cv["D"][:, 0], cv["D"][:, 1]
cvd["density"] = [df.feat[i]["density"] for i in tob]
cvd.to_csv(S + "/finetune_130_cv_predictions.csv", index=False); out.to_csv(S + "/finetune_130_external.csv", index=False)
print("label ranges: 5 bar %.2f-%.2f, 100 bar %.2f-%.2f" % (Y[tob,0].min(), Y[tob,0].max(), Y[tob,1].min(), Y[tob,1].max()))
