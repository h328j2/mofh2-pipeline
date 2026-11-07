"""Memory-bounded PoroNet inference for enlarged graph limits."""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from mofnet import MOFNet

ROOT = Path(__file__).resolve().parent.parent
GLOBALS = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "lcd", "pld", "density"]


def graph(feature: dict) -> Data:
    return Data(
        x=torch.tensor(feature["x"], dtype=torch.float),
        edge_index=torch.tensor(feature["edges"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(feature["dists"], dtype=torch.float),
        gfeat=torch.tensor([[float(feature[c]) for c in GLOBALS]], dtype=torch.float),
    )


def batches(features: list[dict], batch_size: int):
    return DataLoader([graph(f) for f in features], batch_size=batch_size, shuffle=False)


def base_predictions(features: list[dict], checkpoints: list[str], batch_size: int):
    members = []
    for path in checkpoints:
        model = MOFNet()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch in batches(features, batch_size):
                outputs.append((model(batch) * model.y_sd + model.y_mu).numpy())
        members.append(np.concatenate(outputs))
    return np.clip(np.stack(members), 0, None)


def adapted_77_predictions(features: list[dict], checkpoint: Path, batch_size: int):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    members = []
    for member in state["seeds"]:
        model = MOFNet()
        model.load_state_dict(member["state_dict"])
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch in batches(features, batch_size):
                outputs.append((model(batch) * model.y_sd + model.y_mu)[:, 1:].numpy())
        members.append(np.concatenate(outputs))
    values = np.stack(members)
    return np.clip(values.mean(0), 0, None), values[:, :, 0].std(0)


def structure_160_predictions(features: list[dict], checkpoint: Path, batch_size: int):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    members = []
    for member in state["seeds"]:
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
            torch.nn.Linear(320, 128), torch.nn.SiLU(), torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1),
        )
        head.load_state_dict(member["head_state_dict"])
        head.eval()
        outputs = []
        with torch.no_grad():
            for batch in batches(features, batch_size):
                x = (batch.x - base.x_mu) / base.x_sd
                edge_attr = batch.edge_attr[:, :1] / 4.5
                x = F.elu(base.conv1(x, batch.edge_index, edge_attr=edge_attr))
                x = F.elu(base.conv2(x, batch.edge_index, edge_attr=edge_attr))
                x = base.conv3(x, batch.edge_index, edge_attr=edge_attr)
                gf = (batch.gfeat.view(-1, base.n_globals) - base.g_mu) / base.g_sd
                pooled = global_mean_pool(x, batch.batch)
                value = head(torch.cat([pooled, global_mlp(gf)], dim=1))
                outputs.append((value * state["target_std"] + state["target_mean"]).numpy()[:, 0])
        members.append(np.concatenate(outputs))
    values = np.stack(members)
    return np.clip(values.mean(0), 0, None), values.std(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/poronet_cache_large")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/poronet_h2_benchmark.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/poronet_excluded_predicted_vs_actual.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    old_cache = ROOT / "data/poronet_cache"
    excluded = []
    for name in labels.mof:
        old = pickle.load((old_cache / f"{name}.pkl").open("rb"))
        if old.get("status") != "ok" or not old.get("edges"):
            excluded.append(name)

    checkpoints = sorted(
        glob.glob(str(ROOT / "model_repo/models/gnn_full_random_s*.pth"))
        + glob.glob(str(ROOT / "model_repo/results/gnn_full_random_s*.pth"))
    )
    rows = []
    for start in range(0, len(excluded), args.chunk_size):
        names = excluded[start:start + args.chunk_size]
        features = [pickle.load((args.cache / f"{name}.pkl").open("rb")) for name in names]
        base = base_predictions(features, checkpoints, args.batch_size)
        adapted, adapted_std = adapted_77_predictions(
            features, ROOT / "model_repo/models/finetune_77_poronet_ensemble.pth", args.batch_size
        )
        warm, warm_std = structure_160_predictions(
            features, ROOT / "model_repo/models/finetune_160_structure_ensemble.pth", args.batch_size
        )
        label = labels.set_index("mof").loc[names].reset_index()
        z = np.abs(
            (
                np.array([[float(f[c]) for c in GLOBALS] for f in features])
                - torch.load(checkpoints[0], map_location="cpu", weights_only=True)["g_mu"].numpy()
            )
            / torch.load(checkpoints[0], map_location="cpu", weights_only=True)["g_sd"].numpy()
        ).max(1)
        label["n_atoms"] = [f["n_atoms"] for f in features]
        label["density"] = [f["density"] for f in features]
        label["void_fraction"] = [f["void_fraction"] for f in features]
        label["lcd"] = [f["lcd"] for f in features]
        label["pld"] = [f["pld"] for f in features]
        label["baseline_77K_100bar_gL"] = base[:, :, 1].mean(0)
        label["baseline_77K_100bar_std_gL"] = base[:, :, 1].std(0)
        label["adapted_77K_100bar_gL"] = adapted[:, 0]
        label["adapted_77K_100bar_std_gL"] = adapted_std
        label["pred_160K_5bar_gL"] = warm
        label["pred_160K_5bar_std_gL"] = warm_std
        label["pred_DOE_baseline_gL"] = label["baseline_77K_100bar_gL"] - warm
        label["pred_DOE_adapted_gL"] = label["adapted_77K_100bar_gL"] - warm
        label["abs_error_160K_gL"] = abs(label["pred_160K_5bar_gL"] - label["uptake_160K_5bar_gL"])
        label["abs_error_DOE_baseline_gL"] = abs(
            label["pred_DOE_baseline_gL"] - label["deliverable_77K100_to_160K5_gL"]
        )
        label["abs_error_DOE_adapted_gL"] = abs(
            label["pred_DOE_adapted_gL"] - label["deliverable_77K100_to_160K5_gL"]
        )
        label["max_abs_descriptor_z"] = z
        label["in_descriptor_domain"] = z < 3
        rows.append(label)
        print(f"processed {min(start + len(names), len(excluded))}/{len(excluded)}", flush=True)

    out = pd.concat(rows, ignore_index=True).sort_values("pred_DOE_adapted_gL", ascending=False)
    out.to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(out)} rows)")
    for col in ["abs_error_160K_gL", "abs_error_DOE_baseline_gL", "abs_error_DOE_adapted_gL"]:
        print(col, "MAE", out[col].mean())


if __name__ == "__main__":
    main()
