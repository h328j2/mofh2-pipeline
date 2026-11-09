"""Screen the PoroNet literature structures with the saved transfer models."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
from mofnet import MOFNet  # noqa: E402

GLOBALS = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "lcd", "pld", "density"]
MAX_ATOMS = int(os.environ.get("MOF_MAX_ATOMS", "600"))


def graph(feature: dict) -> Data:
    return Data(
        x=torch.tensor(feature["x"], dtype=torch.float),
        edge_index=torch.tensor(feature["edges"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(feature["dists"], dtype=torch.float),
        gfeat=torch.tensor([[float(feature[c]) for c in GLOBALS]], dtype=torch.float),
    )


def base_predictions(features: list[dict]) -> np.ndarray:
    checkpoints = sorted(
        glob.glob(str(S.parent / "model_repo/models/gnn_full_random_s*.pth"))
        + glob.glob(str(S.parent / "model_repo/results/gnn_full_random_s*.pth"))
    )
    graphs = [graph(f) for f in features]
    outputs = []
    for path in checkpoints:
        model = MOFNet()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        batches = []
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=32):
                batches.append((model(batch) * model.y_sd + model.y_mu).numpy())
        outputs.append(np.concatenate(batches))
    return np.stack(outputs)


def structure_transfer(features: list[dict], checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "mofh2-160k-finetune-structure-v1":
        raise ValueError(f"Unsupported checkpoint format: {state.get('format')}")
    graphs = [graph(f) for f in features]
    means, members = [], []
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
            for batch in DataLoader(graphs, batch_size=32):
                x = (batch.x - base.x_mu) / base.x_sd
                edge_attr = batch.edge_attr[:, 0:1] / 4.5
                x = F.elu(base.conv1(x, batch.edge_index, edge_attr=edge_attr))
                x = F.elu(base.conv2(x, batch.edge_index, edge_attr=edge_attr))
                x = base.conv3(x, batch.edge_index, edge_attr=edge_attr)
                gf = (batch.gfeat.view(-1, base.n_globals) - base.g_mu) / base.g_sd
                pooled = global_mean_pool(x, batch.batch)
                outputs.append(
                    (head(torch.cat([pooled, global_mlp(gf)], dim=1))
                     * state["target_std"] + state["target_mean"]).numpy()[:, 0]
                )
        members.append(np.concatenate(outputs))
    values = np.stack(members)
    return np.clip(values.mean(0), 0, None), values.std(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=S.parent / "data/poronet_cache")
    parser.add_argument("--labels", type=Path, default=S.parent / "data/poronet_h2_benchmark.csv")
    parser.add_argument("--output", type=Path, default=S.parent / "data/poronet_screen_predictions.csv")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=S.parent / "model_repo/models/finetune_160_structure_ensemble.pth",
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    records = []
    for name in labels.mof:
        path = args.cache / f"{name}.pkl"
        if not path.exists():
            continue
        feature = pickle.load(path.open("rb"))
        if feature.get("status") == "ok" and feature.get("edges"):
            records.append(feature)
    if len(records) != len(labels):
        print(
            f"screening {len(records)} usable structures; "
            f"{len(labels) - len(records)} exceed the configured {MAX_ATOMS}-atom limit",
            flush=True,
        )
    names = [f["name"] for f in records]
    labels = labels.set_index("mof").loc[names].reset_index()
    base = base_predictions(records)
    p160, p160_std = structure_transfer(records, args.checkpoint)
    labels["pred_77K_100bar_gL"] = base[:, :, 1].mean(0)
    labels["pred_77K_100bar_std_gL"] = base[:, :, 1].std(0)
    labels["pred_160K_5bar_gL"] = p160
    labels["pred_160K_5bar_std_gL"] = p160_std
    labels["pred_deliverable_gL"] = (
        labels["pred_77K_100bar_gL"] - labels["pred_160K_5bar_gL"]
    )
    labels["abs_error_160K_gL"] = abs(
        labels["pred_160K_5bar_gL"] - labels["uptake_160K_5bar_gL"]
    )
    labels["abs_error_deliverable_gL"] = abs(
        labels["pred_deliverable_gL"] - labels["deliverable_77K100_to_160K5_gL"]
    )
    labels.sort_values("pred_deliverable_gL", ascending=False).to_csv(args.output, index=False)
    print(f"saved {args.output} ({len(labels)} structures)")
    print(labels[["abs_error_160K_gL", "abs_error_deliverable_gL"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
