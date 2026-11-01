import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
NEIGHBOR_CUTOFF=4.5
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
