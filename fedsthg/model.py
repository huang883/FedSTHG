"""Four-path encoder, adaptive fusion and local prediction, Eqs.11-34."""
import torch
from torch import nn
import torch.nn.functional as F
from .config import Config
from .graphs import (spatial_structures, temporal_structures, upper_triangle,
                     cosine_similarity, knn_graph)
from .layers import ParallelPaths


def input_projection(n_in, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.LayerNorm(hidden))


class Fusion(nn.Module):
    def __init__(self, width=128, dropout=.3, mode='full'):
        super().__init__()
        self.mode = mode
        self.query = nn.Parameter(torch.empty(width)) if mode != 'no_temporal' else None
        if self.query is not None:
            nn.init.normal_(self.query, std=.02)
        self.gate = nn.Linear(2 * width, 2) if mode not in {
            'no_temporal', 'no_spatial', 'equal_fusion'} else None
        self.output = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                    nn.LayerNorm(width), nn.Dropout(dropout))

    def forward(self, spatial, temporal):
        zs = spatial.mean(-2) if spatial is not None else None
        alpha = (temporal @ self.query).softmax(-1) if temporal is not None else None
        zt = (alpha.unsqueeze(-1) * temporal).sum(-2) if temporal is not None else None
        if zs is None or zt is None:
            joint = zt if zs is None else zs
            gate = joint.new_tensor([0., 1.] if zs is None else [1., 0.]).expand(len(joint), -1)
            consistency = joint.new_zeros(len(joint))
        else:
            gate = self.gate(torch.cat([zs, zt], -1)).softmax(-1) if self.gate is not None \
                else zs.new_full((len(zs), 2), .5)
            joint = gate[:, :1] * zs + gate[:, 1:] * zt
            consistency = 1 - F.cosine_similarity(zs, zt, dim=-1, eps=1e-8)
        return self.output(joint), {'spatial': zs, 'temporal': zt,
            'attention': alpha, 'gate': gate, 'consistency': consistency}


class Encoder(nn.Module):
    """This entire module is theta; no client-local head/refinement inside it."""
    def __init__(self, cfg: Config):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.hidden
        self.width = d if cfg.variant in {'graph_only', 'hypergraph_only'} else 2*d
        if cfg.variant != 'no_spatial':
            self.spatial_input = input_projection(cfg.n_rois, d)
            self.spatial_paths = ParallelPaths(d, cfg.layers, cfg.dropout, cfg.variant)
        if cfg.variant != 'no_temporal':
            # B03: temporal projection's internal structure is not fully specified.
            self.temporal_input = input_projection(cfg.n_rois*(cfg.n_rois-1)//2, d)
            self.position = nn.Parameter(torch.empty(cfg.windows, d))
            nn.init.normal_(self.position, std=.02)
            self.temporal_paths = ParallelPaths(d, cfg.layers, cfg.dropout, cfg.variant)
        self.fusion = Fusion(self.width, cfg.dropout, cfg.variant)

    def forward(self, static, dynamic):
        c = self.cfg
        if static.ndim != 3 or static.shape[1:] != (c.n_rois, c.n_rois):
            raise ValueError('sFC must be [batch, N, N] with configured N')
        if dynamic.shape != (len(static), c.windows, c.n_rois, c.n_rois):
            raise ValueError('dFC must be [batch, W, N, N] with configured W')
        zs = zt = None
        if c.variant != 'no_spatial':
            x = self.spatial_input(static)  # signed FC, no rectification here
            a, h = spatial_structures(static, c.k_sg, c.k_sh)
            zs = self.spatial_paths(x, a, h)
        if c.variant != 'no_temporal':
            x = self.temporal_input(upper_triangle(dynamic)) + self.position
            # Rebuilt on EVERY forward from projected, positioned windows.
            a, h = temporal_structures(x, c.k_tg, c.k_th)
            zt = self.temporal_paths(x, a, h)
        return self.fusion(zs, zt)


class SubjectRefinement(nn.Module):
    def __init__(self, width=128, k=5, dropout=.3):
        super().__init__()
        self.k = k
        self.message = nn.Linear(width, width, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, z):
        if len(z) == 1:
            return z  # Entire layer bypassed, INCLUDING LN, Eq.30 discussion.
        a = knn_graph(cosine_similarity(z), min(self.k, len(z)-1))
        degree = a.sum(-1).clamp_min(1)
        normalized = a / (degree[:, None] * degree[None, :]).sqrt()
        return self.norm(z + F.relu(normalized @ self.message(self.drop(z))))


class LocalPredictor(nn.Module):
    """phi_k. Adapter/head dimensions are B05, not author-confirmed facts."""
    def __init__(self, width=128, hidden=64, k=5, dropout=.3, global_only=False):
        super().__init__()
        self.refine = nn.Identity() if global_only else SubjectRefinement(width, k, dropout)
        self.adapter = nn.Identity() if global_only else nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.LayerNorm(width), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.head(self.adapter(self.refine(z)))  # logits, CE handles log-softmax


def make_local(cfg, width):
    return LocalPredictor(width, cfg.hidden, cfg.k_subject, cfg.dropout,
                          global_only=cfg.variant == 'no_personalization')


def task_loss(logits, labels, auxiliary, coefficient):
    ce = F.cross_entropy(logits, labels)
    st = auxiliary['consistency'].mean()
    return ce + coefficient * st, ce, st
