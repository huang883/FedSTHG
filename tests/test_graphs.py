import pytest
import torch
from fedsthg.graphs import (neighbors, knn_graph, knn_hypergraph,
    spatial_structures, temporal_structures, upper_triangle, hypergraph_message)


def test_positive_priority_and_tie_breaking():
    c = torch.tensor([[0., -.9, .2, -.1], [-.9, 0., .3, -.5],
                      [.2, .3, 0., .1], [-.1, -.5, .1, 0.]])
    idx = neighbors(c.clamp_min(0), 2)
    assert idx[0].tolist() == [2, 1]  # zero tie resolves by index, NOT signed/absolute FC
    a, h = spatial_structures(c, 1, 2)
    assert torch.equal(a, a.T) and not a.diag().any()
    assert (h.sum(0) == 3).all() and (h.diag() == 1).all()
    assert c[0, 1] == -.9
    assert h[:, 0].tolist() == [1, 1, 1, 0]


def test_temporal_uses_window_features_and_offdiagonal_vector():
    fc = torch.arange(18.).reshape(2, 3, 3)
    assert upper_triangle(fc).tolist() == [[1, 2, 5], [10, 11, 14]]
    x = torch.tensor([[1., 0], [0., 1], [.9, .1], [.1, .9]])
    a, h = temporal_structures(x, 1, 2)
    assert a[0, 2] == 1 and a[0, 1] == 0  # non-adjacent window
    assert a.shape == (4, 4) and (h.sum(0) == 3).all()


def test_hypergraph_matches_literal_two_stage_and_gradient():
    h = torch.tensor([[1., 0], [1., 1], [0., 1]])
    x = torch.tensor([[1., 2], [3., 4], [5., 6]], requires_grad=True)
    expected = torch.tensor([[2., 3], [6., 8], [4., 5]])
    out = hypergraph_message(x, h)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out, h @ torch.diag(1 / h.sum(0)) @ h.T @ x)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[1., 1], [2., 2], [1., 1]]))


def test_batched_graphs_and_invalid_k():
    s = torch.eye(4).repeat(3, 1, 1)
    assert knn_graph(s, 1).shape == (3, 4, 4)
    assert (knn_hypergraph(s, 2).sum(-2) == 3).all()
    assert not knn_graph(s, 0).any()
    with pytest.raises(ValueError):
        knn_graph(s, 4)
