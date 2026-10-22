"""Binary, unweighted graph/hypergraph construction, Eqs.8-14 and 29."""
import torch
import torch.nn.functional as F


def cosine_similarity(x):
    unit = F.normalize(x, p=2, dim=-1, eps=1e-8)
    return unit @ unit.transpose(-1, -2)


@torch.no_grad()
def neighbors(similarity, k):
    """Hard kNN: nondifferentiable membership; ties broken by node index."""
    n = similarity.shape[-1]
    if similarity.shape[-2] != n or not 0 <= k < n:
        raise ValueError("Require square similarity and 0 <= k < node count")
    if not torch.isfinite(similarity).all():
        raise ValueError("Nonfinite similarity")
    sim = similarity.clone()
    sim.diagonal(dim1=-2, dim2=-1).fill_(-torch.inf)
    return sim.argsort(dim=-1, descending=True, stable=True)[..., :k]


def knn_graph(similarity, k):
    idx = neighbors(similarity, k)
    directed = torch.zeros_like(similarity).scatter_(-1, idx, 1)
    return torch.maximum(directed, directed.transpose(-1, -2))


def knn_hypergraph(similarity, k):
    idx = neighbors(similarity, k)
    # First construct anchor x member, then transpose to member x hyperedge.
    membership = torch.zeros_like(similarity).scatter_(-1, idx, 1)
    membership.diagonal(dim1=-2, dim2=-1).fill_(1)
    return membership.transpose(-1, -2).contiguous()


def spatial_structures(static_fc, k_graph, k_hyper):
    sim = static_fc.clamp_min(0)
    return knn_graph(sim, k_graph), knn_hypergraph(sim, k_hyper)


def temporal_structures(initial_features, k_graph, k_hyper):
    sim = cosine_similarity(initial_features)
    return knn_graph(sim, k_graph), knn_hypergraph(sim, k_hyper)


def upper_triangle(dynamic_fc):
    n = dynamic_fc.shape[-1]
    if dynamic_fc.shape[-2] != n:
        raise ValueError("FC must be square")
    idx = torch.triu_indices(n, n, 1, device=dynamic_fc.device)
    return dynamic_fc[..., idx[0], idx[1]]


def hypergraph_message(x, incidence):
    """Eq.20: mean within each edge; SUM back to each member (no Dv)."""
    sizes = incidence.sum(dim=-2).clamp_min(1).unsqueeze(-1)
    edge_mean = (incidence.transpose(-1, -2) @ x) / sizes
    return incidence @ edge_mean
