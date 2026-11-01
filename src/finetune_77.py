"""Fine-tune the 77 K GAT (2 bar, 100 bar, DC) on real CoRE-2019 GCMC labels; test on held-out real MOFs; re-screen CoRE 2019."""
import glob, os, sys, copy, pickle, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
from mofnet import MOFNet
from extract import parse_data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
def gl(d):
    f = glob.glob(f"{d}/Output/System_0/*.data")
    if not f: return np.nan
    r = parse_data(f[0]); return r["gL"] if r.get("finished") and r.get("gL") is not None else np.nan
scr = pd.read_csv(S + "/screen_core2019_refined.csv")
r2n = dict(zip(scr.raspa, scr.name))
def featpath(raspa):
    p = f"{S}/core_cache/{raspa}.pkl"
    if os.path.exists(p): return p
    return f"{S}/screen_cache/{r2n[raspa]}.pkl"
rows = []
for _, r in pd.read_csv(S + "/ft77_train.csv").iterrows(): rows.append(dict(set="train", raspa=r.raspa, path=featpath(r.raspa)))
B = pd.read_csv(S + "/core_stageB.csv")
for x in B[B.g77].raspa: rows.append(dict(set="test: CoRE-2019 random (30)", raspa=x, path=featpath(x)))
for x in scr.head(10).raspa: rows.append(dict(set="test: screened top 10", raspa=x, path=featpath(x)))
df = pd.DataFrame(rows)
df["y2"] = [gl(f"{S}/runs_m2/{x}_2") for x in df.raspa]; df["y100"] = [gl(f"{S}/runs_m2/{x}_100") for x in df.raspa]
# external CoRE-2024 / hMOF: 100 bar matched, 2 bar from the first setup (convention-insensitive at 2 bar)
v = pd.read_csv(S + "/gcmc_validation.csv"); import re
ext = []
for _, r in v.iterrows():
    if r.set == "hMOF" and r["name"] not in ("hMOF-3001465", "hMOF-5040736", "hMOF-8081"): continue
    raspa = r["name"] if r.set == "hMOF" else "CoRE_" + re.sub(r"_+", "_", r["name"].replace("[", "_").replace("]", "_")).strip("_")
    cp = f"{S}/ext_cache_{raspa}.pkl"
    if not os.path.exists(cp): pickle.dump(P._featurise(f"{S}/raspa/share/raspa/structures/cif/{raspa}.cif"), open(cp, "wb"))
    ext.append(dict(set="test: external CoRE-2024/hMOF (11)", raspa=raspa, path=cp, y2=r.gcmc_y2, y100=gl(f"{S}/runs_m2/{raspa}_100")))
df = pd.concat([df, pd.DataFrame(ext)], ignore_index=True).dropna(subset=["y2", "y100"]).reset_index(drop=True)
df["dc"] = df.y100 - df.y2
df["vf"] = [float(pickle.load(open(p, "rb"))["void_fraction"]) for p in df.path]
av = pd.read_csv(S + "/av_ft.csv").drop_duplicates("raspa").set_index("raspa")
df["av_h2"] = [av.av_h2.get(x, np.nan) for x in df.raspa]; df["av_h2"] = df.av_h2.fillna(df.av_h2.median())
ACC = np.c_[df.av_h2.values, df.av_h2.values / df.vf.clip(0.05).values]
print(df.set.value_counts().to_dict(), flush=True)
ckpts = sorted(glob.glob(P.REPO + "/models/gnn_full_random_s*.pth") + glob.glob(P.REPO + "/results/gnn_full_random_s*.pth"))
base = []
for c in ckpts:
    n = MOFNet(); n.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); n.eval(); base.append(n)
def to_graph(f):
    g = Data(x=torch.tensor(f["x"], dtype=torch.float), edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(f["dists"], dtype=torch.float)); g.gfeat = torch.tensor([[float(f[c]) for c in P.GLOBALS]], dtype=torch.float); return g
def embed(paths):
    POOL = [[] for _ in base]; ENS = [[] for _ in base]; EM = [[] for _ in base]; chunk = []
    def run(gs):
        with torch.no_grad():
            for k, net in enumerate(base):
                for b in DataLoader(gs, batch_size=32):
                    x = (b.x - net.x_mu) / net.x_sd; ea = b.edge_attr[:, 0:1] / 4.5
                    x = F.elu(net.conv1(x, b.edge_index, edge_attr=ea)); x = F.elu(net.conv2(x, b.edge_index, edge_attr=ea)); x = net.conv3(x, b.edge_index, edge_attr=ea)
                    gf = (b.gfeat.view(-1, net.n_globals) - net.g_mu) / net.g_sd; pooled = global_mean_pool(x, b.batch)
                    h = torch.cat([pooled, net.global_mlp(gf)], 1); h2 = net.head[4](net.head[3](net.head[1](net.head[0](h))))
                    POOL[k].append(torch.cat([pooled, gf], 1)); EM[k].append(h2.numpy()); ENS[k].append((net.head[5](h2) * net.y_sd + net.y_mu).numpy())
    for p in paths:
        chunk.append(to_graph(pickle.load(open(p, "rb"))))
        if len(chunk) == 256: run(chunk); chunk = []
    if chunk: run(chunk)
    return [torch.cat(x) for x in POOL], np.clip(np.stack([np.concatenate(x) for x in ENS]), 0, None), np.concatenate([np.concatenate(x) for x in EM], 1)
POOL, ENS, EMB = embed(df.path)
BASEP = ENS.mean(0)                                         # original model: y2, y100, dc
Y = df[["y2", "y100", "dc"]].values
def finetune(tr, POOLX, epochs=300, lr=5e-4):
    """Retrain global MLP + head (graph layers frozen), starting from the hMOF weights, on real-MOF labels (y2, y100, dc)."""
    models = []
    for seed, net0 in enumerate(base):
        torch.manual_seed(seed)
        gm = copy.deepcopy(net0.global_mlp); hd = copy.deepcopy(net0.head)
        ymu, ysd = net0.y_mu, net0.y_sd
        opt = torch.optim.Adam(list(gm.parameters()) + list(hd.parameters()), lr=lr, weight_decay=1e-4)
        Xp = POOLX[seed]; yt = (torch.tensor(Y, dtype=torch.float) - ymu) / ysd; trt = torch.tensor(tr)
        f = lambda X: hd(torch.cat([X[:, :256], gm(X[:, 256:])], 1))
        for ep in range(epochs):
            gm.train(); hd.train(); perm = trt[torch.randperm(len(tr))]
            for k in range(0, len(tr), 16):
                idx = perm[k:k+16]; opt.zero_grad(); ((f(Xp[idx]) - yt[idx]) ** 2).mean().backward(); opt.step()
        gm.eval(); hd.eval(); models.append((gm, hd, ymu, ysd))
    return models
def predict(models, POOLX, idx=None):
    out = []
    with torch.no_grad():
        for seed, (gm, hd, ymu, ysd) in enumerate(models):
            X = POOLX[seed] if idx is None else POOLX[seed][idx]
            ps = []
            for k in range(0, len(X), 4096): ps.append((hd(torch.cat([X[k:k+4096, :256], gm(X[k:k+4096, 256:])], 1)) * ysd + ymu).numpy())
            out.append(np.concatenate(ps))
    return np.clip(np.mean(out, 0), 0, None)
def linear(tr, te):
    out = np.zeros((len(te), 3))
    for j in range(2):
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(BASEP[tr, :2], Y[tr, j]); out[:, j] = m.predict(BASEP[te, :2])
    out[:, 2] = out[:, 1] - out[:, 0]; return out
def resid_fit(tr, extra=None):
    X = np.c_[EMB, BASEP[:, :2]] if extra is None else np.c_[EMB, BASEP[:, :2], extra]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ms = [RidgeCV(alphas=np.logspace(0, 5, 11)).fit((X[tr] - mu) / sd, Y[tr, j] - BASEP[tr, j]) for j in range(2)]
    return ms, mu, sd
def resid_pred(fit, Xemb, Xbase, extra=None):
    ms, mu, sd = fit; X = ((np.c_[Xemb, Xbase[:, :2]] if extra is None else np.c_[Xemb, Xbase[:, :2], extra]) - mu) / sd
    out = np.zeros((len(X), 3)); out[:, 0] = Xbase[:, 0] + ms[0].predict(X); out[:, 1] = Xbase[:, 1] + ms[1].predict(X); out[:, 2] = out[:, 1] - out[:, 0]
    return np.clip(out, 0, None)
tr = np.where(df.set == "train")[0]
rows = []
def report(name, test, idx, P3):
    for j, t in enumerate(["77 K/2 bar", "77 K/100 bar", "DC (100-2)"]):
        e = P3[:, j] - Y[idx, j]
        rows.append(dict(method=name, test=test, n=len(idx), target=t, MAE=np.abs(e).mean(), bias=e.mean(), spearman=spearmanr(P3[:, j], Y[idx, j])[0]))
# 5-fold CV on the training MOFs
cvb, cvl, cvf, cvr, cva = [np.zeros((len(tr), 3)) for _ in range(5)]
for a, b in ([] if os.environ.get("FT_SKIPCV") else KFold(5, shuffle=True, random_state=0).split(tr)):
    cvl[b] = linear(tr[a], tr[b]); cvf[b] = predict(finetune(tr[a], POOL), POOL, tr[b]); cvr[b] = resid_pred(resid_fit(tr[a]), EMB[tr[b]], BASEP[tr[b]])
    cva[b] = resid_pred(resid_fit(tr[a], ACC), EMB[tr[b]], BASEP[tr[b]], ACC[tr[b]])
report("residual correction (ridge)", "5-fold CV, 150 training MOFs", tr, cvr)
report("residual + H2-accessible volume", "5-fold CV, 150 training MOFs", tr, cva)
cv_mae = {"fine-tuned head": np.abs(cvf[:, 1] - Y[tr, 1]).mean(), "residual correction (ridge)": np.abs(cvr[:, 1] - Y[tr, 1]).mean(), "residual + H2-accessible volume": np.abs(cva[:, 1] - Y[tr, 1]).mean()}
BEST = "fine-tuned head" if os.environ.get("FT_SKIPCV") else min(cv_mae, key=cv_mae.get); print("CV 100-bar MAE:", {k: round(v, 3) for k, v in cv_mae.items()}, "-> re-screen with", BEST, flush=True)
report("original hMOF model", "5-fold CV, 150 training MOFs", tr, BASEP[tr]); report("linear recalibration", "5-fold CV, 150 training MOFs", tr, cvl); report("fine-tuned head", "5-fold CV, 150 training MOFs", tr, cvf)
M = finetune(tr, POOL); RF = resid_fit(tr); RA = resid_fit(tr, ACC)
for t in [s for s in df.set.unique() if s != "train"]:
    idx = np.where(df.set == t)[0]
    report("original hMOF model", t, idx, BASEP[idx]); report("linear recalibration", t, idx, linear(tr, idx)); report("fine-tuned head", t, idx, predict(M, POOL, idx)); report("residual correction (ridge)", t, idx, resid_pred(RF, EMB[idx], BASEP[idx])); report("residual + H2-accessible volume", t, idx, resid_pred(RA, EMB[idx], BASEP[idx], ACC[idx]))
res = pd.DataFrame(rows); pd.set_option("display.width", 220); print(res.round(3).to_string(index=False)); res.to_csv(S + "/finetune_77_results.csv", index=False)
top = np.where(df.set == "test: screened top 10")[0]; pt = {"fine-tuned head": lambda: predict(M, POOL, top), "residual correction (ridge)": lambda: resid_pred(RF, EMB[top], BASEP[top]),
      "residual + H2-accessible volume": lambda: resid_pred(RA, EMB[top], BASEP[top], ACC[top])}[BEST]()
tt = df.loc[top, ["raspa", "y2", "y100"]].copy(); tt["orig_y100"] = BASEP[top, 1]; tt["ft_y100"] = pt[:, 1]; print(tt.round(2).to_string(index=False))
# ---- re-screen CoRE 2019 with the fine-tuned 77 K model ----
if os.environ.get("FT_EXTRA"):
    names = os.environ["FT_EXTRA"].split(",")
    XP, XE, XEMB = embed([f"{S}/real_cache/{n}.pkl" for n in names])
    xf = predict(M, XP)
    pd.DataFrame(dict(name=names, orig_y2=XE.mean(0)[:, 0], orig_y100=XE.mean(0)[:, 1], ft_y2=xf[:, 0], ft_y100=xf[:, 1])).to_csv(S + "/exp_ml_predictions.csv", index=False)
    print(pd.read_csv(S + "/exp_ml_predictions.csv").round(2).to_string(index=False))
if os.environ.get("FT_NOSCREEN"): sys.exit(0)
print("re-screening...", flush=True)
SP, SE, SEMB = embed([f"{S}/screen_cache/{n}.pkl" for n in scr.name])
if BEST == "residual + H2-accessible volume":
    if not os.path.exists(S + "/av_screen.csv"): import subprocess; subprocess.run([sys.executable, S + "/av_feature.py", "screen"], check=True)
    a2 = pd.read_csv(S + "/av_screen.csv").set_index("raspa"); sav = np.array([a2.av_h2.get(x, np.nan) for x in scr.raspa]); sav = np.where(np.isnan(sav), np.nanmedian(sav), sav)
    SACC = np.c_[sav, sav / scr.void_fraction.clip(0.05).values]
ft = {"fine-tuned head": lambda: predict(M, SP), "residual correction (ridge)": lambda: resid_pred(RF, SEMB, SE.mean(0)),
      "residual + H2-accessible volume": lambda: resid_pred(RA, SEMB, SE.mean(0), SACC)}[BEST]()
scr["ft77_method"] = BEST
scr["ft77_y2"], scr["ft77_y100"], scr["ft77_dc"] = ft[:, 0], ft[:, 1], ft[:, 2]
scr["doe_dc_ft"] = scr.ft77_y100 - scr.p160
scr = scr.sort_values("doe_dc_ft", ascending=False)
scr.to_csv(S + "/screen_core2019_ft77.csv", index=False)
print(scr[["name", "metals", "density", "void_fraction", "gnn77_y100", "ft77_y100", "p160", "p160_source", "doe_dc", "doe_dc_ft", "in_domain"]].head(25).round(2).to_string(index=False))
print(scr[["ft77_y100", "doe_dc_ft"]].describe().round(2).to_string()); print(">45 g/L:", int((scr.doe_dc_ft > 45).sum()))
