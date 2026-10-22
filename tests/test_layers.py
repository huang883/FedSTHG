import torch
from torch import nn
from fedsthg.layers import IsomorphismLayer, ParallelPaths


def test_gin_formula_and_learnable_epsilon():
    layer = IsomorphismLayer(3, 0)
    layer.mlp = nn.Identity(); layer.norm = nn.Identity()
    layer.eps.data.fill_(.5)
    x = torch.arange(9.).reshape(3, 3).requires_grad_()
    a = torch.tensor([[0., 1, 0], [1., 0, 1], [0., 1, 0]])
    expected = torch.stack([1.5*x[v] + sum((x[u] for u in range(3) if a[v,u]), torch.zeros(3))
                            for v in range(3)])
    torch.testing.assert_close(layer(x, a), expected)
    layer(x, a).sum().backward()
    torch.testing.assert_close(layer.eps.grad, x.sum())


def test_hgin_formula_no_vertex_normalization():
    layer = IsomorphismLayer(2, 0, True)
    layer.mlp = nn.Identity(); layer.norm = nn.Identity()
    x = torch.tensor([[1., 2], [3., 4], [5., 6]])
    h = torch.tensor([[1., 0], [1., 1], [0., 1]])
    torch.testing.assert_close(layer(x, h), x + torch.tensor([[2., 3], [6., 8], [4., 5]]))


def test_parallel_input_independence_and_all_gradients():
    torch.manual_seed(1)
    paths = ParallelPaths(8, 2, 0)
    seen = []
    paths.gin[0].register_forward_pre_hook(lambda m, args: seen.append(args[0]))
    paths.hgin[0].register_forward_pre_hook(lambda m, args: seen.append(args[0]))
    x = torch.randn(2, 4, 8, requires_grad=True)
    a = (torch.ones(4,4)-torch.eye(4)).expand(2,-1,-1)
    h = torch.eye(4).expand(2,-1,-1)
    out = paths(x, a, h)
    assert seen[0] is x and seen[1] is x
    assert out.shape == (2,4,16)
    ids_g = {id(p) for p in paths.gin.parameters()}
    assert not ids_g & {id(p) for p in paths.hgin.parameters()}
    (out * torch.randn_like(out)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in paths.parameters())
    assert x.grad.abs().sum() > 0
