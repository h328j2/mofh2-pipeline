"""Featurise ToBaCCo MOFs exactly like the hMOF notebook and predict with the 4-seed GNN ensemble."""
import os, sys, glob, re, json, subprocess, tempfile, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MODEL_REPO", S + "/model_repo")
ZEO = S + "/ext/zeopp-lsmo/zeo++/network"
CIF_DIR = S + "/tob_test/output_cifs"
OUT = S + "/topo_predictions.csv"
NEIGHBOR_CUTOFF = 4.5
# The hMOF checkpoints accept variable-size graphs; this is a resource guard,
# not an architectural limit. Override it for larger structures when memory
# and Zeo++ runtime are available, for example MOF_MAX_ATOMS=1200.
MAX_ATOMS = int(os.environ.get("MOF_MAX_ATOMS", "600"))
CACHE = S + "/topo_cache"
os.makedirs(CACHE, exist_ok=True)
GLOBALS = ["void_fraction", "surface_area_m2g", "surface_area_m2cm3", "lcd", "pld", "density"]
KV = re.compile(r"(\S+):\s+([-+\d.eE]+)")

def element_features(sym, cache={}):
    from pymatgen.core import Element
    if sym not in cache:
        el = Element(sym)
        try:
            ion = float(el.ionization_energies[0]) if el.ionization_energies else 0.0
        except Exception:
            ion = 0.0
        cache[sym] = [float(el.Z), float(el.X if el.X is not None else 0.0),
                      float(el.atomic_radius if el.atomic_radius is not None else 1.5),
                      float(el.atomic_mass), ion, float(el.group), float(el.row)]
    return cache[sym]

def zeo(cif):
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    s = Structure.from_file(cif)
    with tempfile.TemporaryDirectory() as tmp:
        c = os.path.join(tmp, "s.cif"); CifWriter(s).write_file(c)
        out = {"density_pmg": float(s.density)}
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-ha", "-res", f"{tmp}/s.res", c], capture_output=True, timeout=900)
        tok = open(f"{tmp}/s.res").read().split()
        out["lcd_zeo"], out["pld_zeo"] = float(tok[1]), float(tok[2])
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-ha", "-sa", "1.86", "1.86", "2000", f"{tmp}/s.sa", c], capture_output=True, timeout=900)
        kv = dict(KV.findall(open(f"{tmp}/s.sa").readline()))
        out["sa_m2g_zeo"] = float(kv["ASA_m^2/g"])
        subprocess.run(["prlimit", "--as=1500000000", ZEO, "-ha", "-vol", "0.0", "0.0", "50000", f"{tmp}/s.vol", c], capture_output=True, timeout=900)
        kv = dict(KV.findall(open(f"{tmp}/s.vol").readline()))
        out["vf_zeo_0.0"] = float(kv["AV_Volume_fraction"])
        return s, out

CAL = pd.read_csv(S + "/user_zip/zeo_calibration.csv").set_index("descriptor")
def calibrate(z):
    d = {}
    for t in ["lcd", "pld", "surface_area_m2g", "void_fraction"]:
        r = CAL.loc[t]; d[t] = r.slope * z[r.zeo_column] + r.intercept
    d["density"] = z["density_pmg"]
    d["surface_area_m2cm3"] = d["surface_area_m2g"] * d["density"]   # same definition as hMOF JSON
    return d

def featurise(cif):
    import pickle
    cp = os.path.join(CACHE, os.path.basename(cif)[:-4] + ".pkl")
    if os.path.exists(cp):
        return cp
    r = _featurise(cif)
    pickle.dump(r, open(cp, "wb"))
    return cp

def _featurise(cif):
    try:
        from pymatgen.core import Structure
        s = Structure.from_file(cif)
        if len(s) > MAX_ATOMS:
            return {"name": os.path.basename(cif)[:-4], "status": "too_large", "n_atoms": len(s)}
        x = [list(element_features(site.specie.symbol)) for site in s]
        edges, dists, deg = [], [], [0] * len(s)
        for i, nl in enumerate(s.get_all_neighbors(r=NEIGHBOR_CUTOFF)):
            for n in nl:
                edges.append([i, n.index]); dists.append([float(n.nn_distance)]); deg[i] += 1
        for i in range(len(x)):
            x[i].append(float(deg[i]))
        _, z = zeo(cif)
        return {"name": os.path.basename(cif)[:-4], "status": "ok", "n_atoms": len(s), "x": x, "edges": edges,
                "dists": dists, **calibrate(z), **{f"raw_{k}": v for k, v in z.items()},
                "elements": sorted({site.specie.symbol for site in s})}
    except Exception as e:
        return {"name": os.path.basename(cif)[:-4], "status": "error:" + str(e)[:60]}

if __name__ == "__main__":
    cifs = sorted(glob.glob(CIF_DIR + "/*.cif"))
    if len(sys.argv) > 1:
        cifs = cifs[: int(sys.argv[1])]
    import pickle
    with Pool(2, maxtasksperchild=20) as pool:
        paths = list(pool.imap_unordered(featurise, cifs, chunksize=1))
    res = [pickle.load(open(p, "rb")) for p in paths]
    ok = [r for r in res if r["status"] == "ok" and r["edges"]]
    print(f"featurised {len(ok)} / {len(cifs)}", pd.Series([r["status"] for r in res]).value_counts().to_dict())

    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool

    class MOFNet(nn.Module):
        def __init__(self, in_features=8, n_globals=6, hidden=64, heads=4, n_targets=3):
            super().__init__()
            self.n_globals = n_globals
            for name, size in [("x_mu", in_features), ("x_sd", in_features), ("g_mu", n_globals),
                               ("g_sd", n_globals), ("y_mu", n_targets), ("y_sd", n_targets)]:
                self.register_buffer(name, torch.zeros(size) if "mu" in name else torch.ones(size))
            self.conv1 = GATConv(in_features, hidden, heads=heads, edge_dim=1)
            self.conv2 = GATConv(hidden * heads, hidden, heads=heads, edge_dim=1)
            self.conv3 = GATConv(hidden * heads, hidden, heads=heads, edge_dim=1)
            self.global_mlp = nn.Sequential(nn.Linear(n_globals, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(hidden * heads + 64, 128), nn.SiLU(), nn.Dropout(0.05),
                                      nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, n_targets))
        def forward(self, data):
            x = (data.x - self.x_mu) / self.x_sd
            ea = data.edge_attr[:, 0:1] / NEIGHBOR_CUTOFF
            x = F.elu(self.conv1(x, data.edge_index, edge_attr=ea))
            x = F.elu(self.conv2(x, data.edge_index, edge_attr=ea))
            x = self.conv3(x, data.edge_index, edge_attr=ea)
            gf = (data.gfeat.view(-1, self.n_globals) - self.g_mu) / self.g_sd
            return self.head(torch.cat([global_mean_pool(x, data.batch), self.global_mlp(gf)], dim=1))

    graphs = []
    for r in ok:
        g = Data(x=torch.tensor(r["x"], dtype=torch.float),
                 edge_index=torch.tensor(r["edges"], dtype=torch.long).t().contiguous(),
                 edge_attr=torch.tensor(r["dists"], dtype=torch.float))
        g.gfeat = torch.tensor([[r[c] for c in GLOBALS]], dtype=torch.float)
        graphs.append(g)
    ckpts = sorted(glob.glob(REPO + "/models/gnn_full_random_s*.pth") + glob.glob(REPO + "/results/gnn_full_random_s*.pth"))
    print("checkpoints:", [os.path.basename(c) for c in ckpts])
    ens = []
    gmu = gsd = None
    for c in ckpts:
        net = MOFNet(); net.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); net.eval()
        gmu, gsd = net.g_mu.numpy(), net.g_sd.numpy()
        outs = []
        with torch.no_grad():
            for b in DataLoader(graphs, batch_size=32):
                outs.append(net(b) * net.y_sd + net.y_mu)
        ens.append(torch.cat(outs).numpy())
    ens = np.clip(np.stack(ens), 0, None)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "edges", "dists")} for r in ok])
    for j, t in enumerate(["y2", "y100", "dc"]):
        df[f"gnn_{t}"] = ens[:, :, j].mean(0); df[f"gnn_{t}_std"] = ens[:, :, j].std(0)
    # domain: elements seen in hMOF and descriptors within +-3 SD of the hMOF training distribution
    SEEN = {"H", "C", "N", "O", "F", "Cl", "Br", "Zn", "Cu", "Zr", "V"}
    zsc = (df[GLOBALS].values - gmu) / gsd
    df["max_abs_z"] = np.abs(zsc).max(1)
    df["unseen_el"] = df.elements.map(lambda e: ",".join(sorted(set(e) - SEEN)))
    df["in_domain"] = (df.max_abs_z < 3) & (df.unseen_el == "")
    df["net"] = df.name.str.split("_").str[0]
    df["node"] = df.name.str.extract(r"_v\d+-(\w+?)_\d+-")[0]
    df["linker"] = df.name.str.extract(r"-([^-]+)$")[0]
    seen_nets = set(pd.read_csv(REPO + "/results/error_by_topology.csv").topology.str.lower())
    df["seen_net"] = df.net.str.lower().isin(seen_nets)
    df.drop(columns=["elements"]).to_csv(OUT, index=False)
    print(df[["gnn_dc", "gnn_y100", "void_fraction", "lcd", "pld"]].describe().round(2).to_string())
    print("in domain:", int(df.in_domain.sum()), "/", len(df), "| nets:", df.net.nunique(),
          "| new nets:", df.loc[~df.seen_net, "net"].nunique())
