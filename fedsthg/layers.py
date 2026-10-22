"""GIN and HGIN with the paper's exact aggregation and MLP ordering."""
import torch
from torch import nn
from .graphs import hypergraph_message


class IsomorphismLayer(nn.Module):
    def __init__(self, hidden=64, dropout=.3, hypergraph=False):
        super().__init__()
        self.hypergraph = hypergraph
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, structure):
        message = hypergraph_message(x, structure) if self.hypergraph else structure @ x
        return self.norm(self.mlp((1 + self.eps) * x + message))


class ParallelPaths(nn.Module):
    """Both paths consume the SAME x0; no inter-path propagation (Eq.22)."""
    def __init__(self, hidden=64, layers=2, dropout=.3, variant='full'):
        super().__init__()
        self.gin = nn.ModuleList([IsomorphismLayer(hidden, dropout) for _ in range(layers)]) \
            if variant != 'hypergraph_only' else nn.ModuleList()
        self.hgin = nn.ModuleList([IsomorphismLayer(hidden, dropout, True) for _ in range(layers)]) \
            if variant != 'graph_only' else nn.ModuleList()

    def forward(self, x0, adjacency, incidence):
        outputs = []
        if self.gin:
            x = x0
            for layer in self.gin:
                x = layer(x, adjacency)
            outputs.append(x)
        if self.hgin:
            x = x0
            for layer in self.hgin:
                x = layer(x, incidence)
            outputs.append(x)
        return torch.cat(outputs, dim=-1)
