from dataclasses import replace
import pytest
import torch
from torch import nn
from fedsthg.config import Config
from fedsthg.model import Encoder, Fusion, SubjectRefinement, make_local, task_loss


def small_config(**kwargs):
    return replace(Config(n_rois=8, aligned_length=24, window=8, step=4,
        hidden=8, k_sg=2, k_sh=3, k_tg=1, k_th=2, dropout=0), **kwargs)


def test_attention_and_gate_reference():
    f = Fusion(4, 0)
    f.query.data.zero_(); f.gate.weight.data.zero_(); f.gate.bias.data.zero_()
    s = torch.randn(3, 8, 4); t = torch.randn(3, 5, 4)
    z, aux = f(s, t)
    torch.testing.assert_close(aux['attention'], torch.full((3,5), .2))
    torch.testing.assert_close(aux['temporal'], t.mean(1))
    torch.testing.assert_close(aux['gate'], torch.full((3,2), .5))
    torch.testing.assert_close(z, f.output((s.mean(1)+t.mean(1))/2))
    _, same = f(s[:, :1], s[:, :1])
    torch.testing.assert_close(same['consistency'], torch.zeros(3), atol=1e-6, rtol=0)


def test_subject_refinement_formula_and_single_bypass():
    layer = SubjectRefinement(2, 1, 0)
    layer.message.weight.data.copy_(torch.eye(2)); layer.norm = nn.Identity()
    z = torch.tensor([[1.,2.], [3.,4.]])
    torch.testing.assert_close(layer(z), z + z.flip(0))
    layer.norm = nn.LayerNorm(2)
    one = z[:1]
    assert layer(one) is one


@pytest.mark.parametrize('variant', ['full','no_spatial','no_temporal','graph_only',
                                    'hypergraph_only','equal_fusion','no_personalization'])
def test_end_to_end_shapes_gradients_and_variants(variant):
    torch.manual_seed(17)
    c = small_config(variant=variant)
    encoder = Encoder(c); local = make_local(c, encoder.width)
    s = torch.randn(4, c.n_rois,c.n_rois)
    t = torch.randn(4,c.windows,c.n_rois,c.n_rois)
    z, aux = encoder(s,t); logits = local(z)
    assert logits.shape == (4,2) and torch.isfinite(logits).all()
    torch.testing.assert_close(aux['gate'].sum(-1), torch.ones(4))
    loss, _, _ = task_loss(logits, torch.tensor([0,1,1,0]), aux, c.consistency)
    loss.backward()
    for name,p in list(encoder.named_parameters())+list(local.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if variant != 'no_temporal':
        assert encoder.position.grad.abs().sum() > 0


def test_encoder_uses_positioned_latent_for_graph(monkeypatch):
    import fedsthg.model as module
    c=small_config(); m=Encoder(c); seen=[]
    original=module.temporal_structures
    def spy(x,kg,kh):
        seen.append(x.detach().clone()); return original(x,kg,kh)
    monkeypatch.setattr(module,'temporal_structures',spy)
    d=torch.randn(2,c.windows,8,8); s=torch.randn(2,8,8)
    m(s,d)
    from fedsthg.graphs import upper_triangle
    torch.testing.assert_close(seen[0],m.temporal_input(upper_triangle(d))+m.position)
    assert seen[0].shape == (2,c.windows,c.hidden)


def test_eval_save_load_and_batch1(tmp_path):
    c=small_config(); e=Encoder(c).eval(); l=make_local(c,e.width).eval()
    s=torch.randn(1,8,8); d=torch.randn(1,c.windows,8,8)
    expected=l(e(s,d)[0])
    p=tmp_path/'model.pt'; torch.save({'encoder':e.state_dict(),'local':l.state_dict()},p)
    e2=Encoder(c).eval(); l2=make_local(c,e2.width).eval()
    state=torch.load(p,weights_only=True);e2.load_state_dict(state['encoder']);l2.load_state_dict(state['local'])
    torch.testing.assert_close(l2(e2(s,d)[0]),expected,rtol=0,atol=0)
