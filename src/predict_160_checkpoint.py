"""Predict 160 K / 5 bar uptake with the saved fine-tuned + Widom ensemble.

Usage:
  python predict_160_checkpoint.py FEATURE.pkl --logkH -5.2 --qst0 8.5
  python predict_160_checkpoint.py a.pkl b.pkl --widom-csv widom.csv

The feature pickle must be produced by the repository featurisation pipeline.
The optional Widom CSV must contain name, logkH, and qst0 columns.
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from mofnet import MOFNet


def graph_from_feature(feature):
    if feature.get("status") != "ok":
        raise ValueError(f"{feature.get('name', 'feature')} has status {feature.get('status')}")
    if not feature.get("edges"):
        raise ValueError(f"{feature.get('name', 'feature')} has no graph edges")
    return Data(
        x=torch.tensor(feature["x"], dtype=torch.float),
        edge_index=torch.tensor(feature["edges"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(feature["dists"], dtype=torch.float),
        gfeat=torch.tensor([[float(feature[c]) for c in GLOBALS]], dtype=torch.float),
    )


def load_widom(args, names):
    if args.widom_csv:
        table = pd.read_csv(args.widom_csv).set_index("name")
        missing = [n for n in names if n not in table.index]
        if missing:
            raise ValueError(f"Widom CSV is missing: {', '.join(missing)}")
        return np.asarray(table.loc[names, ["logkH", "qst0"]], dtype=np.float32)
    if args.logkH is None or args.qst0 is None:
        raise ValueError("Provide --logkH and --qst0, or --widom-csv")
    return np.tile([args.logkH, args.qst0], (len(names), 1)).astype(np.float32)


def predict(features, widom_values, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "mofh2-160k-finetune-v1":
        raise ValueError("Unsupported checkpoint format")
    graphs = [graph_from_feature(f) for f in features]
    outputs = []
    wmu = torch.tensor(checkpoint["widom_mean"], dtype=torch.float)
    wsd = torch.tensor(checkpoint["widom_std"], dtype=torch.float)
    wnorm = (torch.tensor(widom_values, dtype=torch.float) - wmu) / wsd
    for member in checkpoint["seeds"]:
        base = MOFNet()
        base.load_state_dict(member["base_state_dict"])
        base.eval()
        global_mlp = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
        )
        global_mlp.load_state_dict(member["global_mlp_state_dict"])
        global_mlp.eval()
        head = torch.nn.Sequential(
            torch.nn.Linear(322, 128), torch.nn.SiLU(), torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1),
        )
        head.load_state_dict(member["head_state_dict"])
        head.eval()
        pooled = []
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=32):
                x = (batch.x - base.x_mu) / base.x_sd
                edge_attr = batch.edge_attr[:, 0:1] / 4.5
                x = F.elu(base.conv1(x, batch.edge_index, edge_attr=edge_attr))
                x = F.elu(base.conv2(x, batch.edge_index, edge_attr=edge_attr))
                x = base.conv3(x, batch.edge_index, edge_attr=edge_attr)
                gf = (batch.gfeat.view(-1, base.n_globals) - base.g_mu) / base.g_sd
                pooled.append(torch.cat([global_mean_pool(x, batch.batch), gf], dim=1))
            pooled = torch.cat(pooled)
            pred = head(torch.cat([pooled[:, :256], global_mlp(pooled[:, 256:]), wnorm], dim=1))
            outputs.append((pred[:, 0] * checkpoint["target_std"] + checkpoint["target_mean"]).numpy())
    values = np.stack(outputs)
    return np.clip(values.mean(0), 0, None), values.std(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", nargs="+", help="Feature pickle files")
    parser.add_argument("--checkpoint", default=os.path.join(
        S, "..", "model_repo", "models", "finetune_160_finetuneW_ensemble.pth"))
    parser.add_argument("--widom-csv")
    parser.add_argument("--logkH", type=float)
    parser.add_argument("--qst0", type=float)
    parser.add_argument("--output", default="predictions_160k.csv")
    args = parser.parse_args()
    features = [pickle.load(open(path, "rb")) for path in args.features]
    names = [f.get("name") or os.path.splitext(os.path.basename(path))[0]
             for f, path in zip(features, args.features)]
    widom_values = load_widom(args, names)
    mean, std = predict(features, widom_values, os.path.abspath(args.checkpoint))
    result = pd.DataFrame({
        "name": names,
        "uptake_160K_5bar_gL": mean,
        "ensemble_std_gL": std,
        "logkH": widom_values[:, 0],
        "qst0": widom_values[:, 1],
    })
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"saved {args.output}")


GLOBALS = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "lcd", "pld", "density"]


if __name__ == "__main__":
    main()
