import copy
import pytest
import torch
from fedsthg.config import Config
from fedsthg.federated import (FederatedTrainer,aggregate,clone_state,proximal_penalty,seed_everything)


def cfg():
    return Config(n_rois=6,aligned_length=16,window=8,step=2,hidden=8,
                  k_sg=2,k_sh=3,k_tg=1,k_th=2,batch_size=3,dropout=.2)


def data():
    return {k:{'static':torch.randn(n,6,6),'dynamic':torch.randn(n,5,6,6),
               'labels':torch.arange(n)%2} for k,n in [('a',5),('b',7)]}


def same(a,b):
    return all(torch.equal(a[k],b[k]) for k in a)


def test_weighted_encoder_aggregation_and_schema():
    out=aggregate([{'w':torch.tensor([1.,3.])},{'w':torch.tensor([5.,7.])}],[1,3],['w'])
    torch.testing.assert_close(out['w'],torch.tensor([4.,6.]))
    with pytest.raises(ValueError,match='schema'):
        aggregate([{'w':torch.ones(2),'head':torch.ones(2)}],[1],['w'])


def test_phase_paths_and_frozen_reference():
    seed_everything(3); c=cfg(); t=FederatedTrainer(c,['a','b']); ds=data()
    personal={k:clone_state(m) for k,m in t.personal.items()}
    global_before=clone_state(t.global_encoder); local_before=clone_state(t.local['a'])
    t.phase_one(ds)
    assert all(same(personal[k],clone_state(t.personal[k])) for k in t.names)
    assert not same(global_before,clone_state(t.global_encoder))
    assert not same(local_before,clone_state(t.local['a']))
    new_global=clone_state(t.global_encoder)
    t.phase_two(ds)
    assert same(new_global,clone_state(t.global_encoder))
    assert all(p.grad is None for p in t.global_encoder.parameters())
    assert not same(personal['a'],clone_state(t.personal['a']))
    assert not same(clone_state(t.personal['a']),clone_state(t.personal['b']))


def test_proximal_exact_gradient_and_reference_detached():
    m=torch.nn.Linear(2,1,bias=False)
    ref={'weight':torch.zeros_like(m.weight,requires_grad=True)}
    value=proximal_penalty(m,ref);value.backward()
    torch.testing.assert_close(m.weight.grad,2*m.weight)
    assert ref['weight'].grad is None


def test_resume_exact_with_dropout(tmp_path):
    seed_everything(8);ds=data();t=FederatedTrainer(cfg(),['a','b'])
    t.run_round(ds);p=tmp_path/'checkpoint.pt';t.save(p,{'split':'test'})
    t.run_round(ds)
    resumed,prov=FederatedTrainer.load(p)
    assert prov=={'split':'test'}
    resumed.run_round(ds)
    assert resumed.round==2
    assert same(clone_state(t.global_encoder),clone_state(resumed.global_encoder))
    for k in t.names:
        assert same(clone_state(t.personal[k]),clone_state(resumed.personal[k]))
        assert same(clone_state(t.local[k]),clone_state(resumed.local[k]))
    assert (t.predict('a',ds['a'])==resumed.predict('a',ds['a'])).all()


def test_global_only_is_explicit_ablation():
    c=cfg();c.variant='no_personalization'
    t=FederatedTrainer(c,['a','b']);ds=data();log=t.run_round(ds)
    assert log['phase_II']=={} and t.global_local is not None
    assert len(t.predict('a',ds['a']))==5
