"""Fine-tune the 77 K hMOF ensemble on PoroNet/Tobacco labels.

The 80/20 split is stratified by structural difficulty inferred from descriptor
z-scores, without using adsorption labels to define the hard region.
"""
import copy
import glob
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

S = Path(__file__).resolve().parent
ROOT = S.parent
sys.path.insert(0, str(S))
from mofnet import MOFNet

GLOBALS = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "lcd", "pld", "density"]
TARGETS = ["uptake_77K_100bar_gL", "deliverable_77K100_to_160K5_gL"]


def graph(f):
    return Data(
        x=torch.tensor(f["x"], dtype=torch.float),
        edge_index=torch.tensor(f["edges"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(f["dists"], dtype=torch.float),
        gfeat=torch.tensor([[float(f[c]) for c in GLOBALS]], dtype=torch.float),
    )


def metrics(y, pred):
    rows = []
    for j, target in enumerate(TARGETS):
        err = pred[:, j] - y[:, j]
        rows.append(dict(target=target, n=len(y), MAE_gL=np.mean(abs(err)),
                         RMSE_gL=np.sqrt(np.mean(err ** 2)), bias_gL=np.mean(err),
                         spearman=spearmanr(y[:, j], pred[:, j]).statistic))
    return pd.DataFrame(rows)


def embeddings(model, graphs):
    model.eval()
    pooled, baseline = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=32, shuffle=False):
            x = (batch.x - model.x_mu) / model.x_sd
            ea = batch.edge_attr[:, :1] / 4.5
            x = F.elu(model.conv1(x, batch.edge_index, edge_attr=ea))
            x = F.elu(model.conv2(x, batch.edge_index, edge_attr=ea))
            x = model.conv3(x, batch.edge_index, edge_attr=ea)
            gf = (batch.gfeat.view(-1, model.n_globals) - model.g_mu) / model.g_sd
            h = torch.cat([global_mean_pool(x, batch.batch), gf], dim=1)
            pooled.append(h)
            baseline.append(model.head(torch.cat([h[:, :256], model.global_mlp(h[:, 256:])], dim=1))
                            * model.y_sd + model.y_mu)
    return torch.cat(pooled), torch.cat(baseline)


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    table = pd.read_csv(ROOT / "data/poronet_h2_benchmark.csv")
    features, names = [], []
    for name in table.mof:
        f = pickle.load((ROOT / "data/poronet_cache" / f"{name}.pkl").open("rb"))
        if f.get("status") == "ok" and f.get("edges"):
            names.append(name); features.append(f)
    table = table.set_index("mof").loc[names].reset_index()
    y = table[TARGETS].to_numpy(dtype=np.float32)
    g = np.array([[float(f[c]) for c in GLOBALS] for f in features])
    ref = torch.load(ROOT / "model_repo/models/gnn_full_random_s42.pth", map_location="cpu", weights_only=True)
    z = (g - ref["g_mu"].numpy()) / ref["g_sd"].numpy()
    hard = ((z[:, 5] < -1) | (z[:, 0] > 1) | (z[:, 3] > 2) | (z[:, 4] > 2) | (z[:, 1] > 2))
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=.20, random_state=42,
                                           stratify=hard.astype(int))
    graphs = [graph(f) for f in features]
    checkpoints = sorted(glob.glob(str(ROOT / "model_repo/models/gnn_full_random_s*.pth")) +
                         glob.glob(str(ROOT / "model_repo/results/gnn_full_random_s*.pth")))
    baseline_members, adapted_members, saved = [], [], []
    for checkpoint in checkpoints:
        base = MOFNet()
        base.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        H, base_pred = embeddings(base, graphs)
        adapted = copy.deepcopy(base)
        mu = torch.cat([base.y_mu[:1], torch.tensor(y[train_idx].mean(0), dtype=torch.float)])
        sd = torch.cat([base.y_sd[:1], torch.tensor(y[train_idx].std(0) + 1e-6, dtype=torch.float)])
        adapted.y_mu.copy_(mu); adapted.y_sd.copy_(sd)
        opt = torch.optim.Adam(list(adapted.global_mlp.parameters()) + list(adapted.head.parameters()),
                               lr=1e-3, weight_decay=1e-4)
        yt = (torch.tensor(y[train_idx]) - mu[1:]) / sd[1:]
        htr = H[train_idx]
        weights = torch.tensor(np.where(hard[train_idx], 2.0, 1.0), dtype=torch.float)
        for _ in range(500):
            adapted.train(); opt.zero_grad()
            pred = adapted.head(torch.cat([htr[:, :256], adapted.global_mlp(htr[:, 256:])], dim=1))[:, 1:]
            loss = (((pred - yt) ** 2) * weights[:, None]).mean()
            loss.backward(); opt.step()
        adapted.eval()
        with torch.no_grad():
            pred = adapted.head(torch.cat([H[:, :256], adapted.global_mlp(H[:, 256:])], dim=1))[:, 1:] * sd[1:] + mu[1:]
        baseline_members.append(base_pred.numpy())
        adapted_members.append(pred.numpy())
        saved.append({"source": checkpoint, "state_dict": adapted.state_dict()})
    baseline = np.mean(baseline_members, axis=0)
    adapted = np.mean(adapted_members, axis=0)
    out = ROOT / "data"
    baseline_targets = baseline[:, [1, 2]]
    metrics(y[test_idx], baseline_targets[test_idx]).assign(model="baseline").to_csv(out / "poronet_77_finetune_baseline_test.csv", index=False)
    metrics(y[test_idx], adapted[test_idx]).assign(model="finetuned").to_csv(out / "poronet_77_finetune_test.csv", index=False)
    result = table.iloc[test_idx].copy()
    for j, target in enumerate(TARGETS):
        result[f"baseline_{target}"] = baseline_targets[test_idx, j]
        result[f"finetuned_{target}"] = adapted[test_idx, j]
    result["hard_region"] = hard[test_idx]
    result.to_csv(out / "poronet_77_finetune_predictions.csv", index=False)
    screen = pd.read_csv(out / "poronet_screen_predictions.csv").set_index("mof").loc[table.mof].reset_index()
    screen["finetuned_77K_100bar_gL"] = adapted[:, 0]
    screen["finetuned_deliverable_gL"] = adapted[:, 1]
    screen["finetuned_doe_gL"] = screen["finetuned_77K_100bar_gL"] - screen["pred_160K_5bar_gL"]
    screen["hard_region"] = hard
    screen.sort_values("finetuned_doe_gL", ascending=False).to_csv(
        out / "poronet_screen_predictions_77_finetuned.csv", index=False
    )
    torch.save({"format": "mofh2-77k-poronet-finetuned-v1", "target": TARGETS,
                "train_names": table.iloc[train_idx].mof.tolist(), "test_names": table.iloc[test_idx].mof.tolist(),
                "hard_region_definition": "density z<-1 or void_fraction z>1 or LCD/PLD/SA z>2",
                "seeds": saved}, ROOT / "model_repo/models/finetune_77_poron​​et_ensemble.pth".replace("\u200b", ""))
    print("train/test:", len(train_idx), len(test_idx), "hard:", int(hard.sum()))
    print(metrics(y[test_idx], baseline_targets[test_idx]).round(4).to_string(index=False))
    print(metrics(y[test_idx], adapted[test_idx]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
