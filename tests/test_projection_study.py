import copy
from dataclasses import replace
import pytest
import torch
from fedsthg.config import Config
from fedsthg.model import Encoder, make_local, task_loss
from fedsthg.federated import FederatedTrainer, seed_everything, clone_state
from fedsthg.diagnostics import diagnose, train_validation


def small():
    return Config(n_rois=6, aligned_length=16, window=8, step=2, hidden=8,
                  k_sg=2, k_sh=3, k_tg=1, k_th=2, batch_size=3, dropout=.3,
                  temporal_mlp_width=16)


def dataset():
    return {'site': {'static': torch.randn(5,6,6), 'dynamic': torch.randn(5,5,6,6),
                     'labels': torch.arange(5)%2, 'ids': [str(i) for i in range(5)]}}


def test_paper_size_candidate_and_gradient():
    c=Config(temporal_mlp_width=256);e=Encoder(c);local=make_local(c,e.width)
    assert sum(p.numel() for p in e.parameters())==1885834
    assert sum(p.numel() for p in local.parameters())+1885834==1927628
    assert len(list(e.temporal_input.parameters()))+len(list(e.temporal_paths.parameters()))+1==35
    c=small();e=Encoder(c);local=make_local(c,e.width);ds=dataset()['site']
    z,aux=e(ds['static'],ds['dynamic']);loss,_,_=task_loss(local(z),ds['labels'],aux,c.consistency)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
    assert e.temporal_input[0].weight.grad.abs().sum()>0
    assert e.temporal_input[2].weight.grad.abs().sum()>0


def test_diagnostics_do_not_change_training_or_rng():
    torch.set_num_threads(2);seed_everything(7);ds=dataset()
    original=FederatedTrainer(small(),ds);other=copy.deepcopy(original)
    state=torch.get_rng_state().clone();flags=[m.training for m in other.personal['site'].modules()]
    report=diagnose(other,ds)
    assert torch.equal(state,torch.get_rng_state())
    assert flags==[m.training for m in other.personal['site'].modules()]
    assert report['pooled']['n']==5 and report['pooled']['ce']>0
    original.run_round(ds);torch.set_rng_state(state);other.run_round(ds)
    for first,second in [(original.global_encoder,other.global_encoder),
                         (original.personal['site'],other.personal['site']),
                         (original.local['site'],other.local['site'])]:
        assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())


def test_diagnostic_loss_from_logits_and_split_guard():
    seed_everything(5);ds=dataset();model=FederatedTrainer(small(),ds)
    report=diagnose(model,ds);e=model.personal['site'].eval();local=model.local['site'].eval()
    ce=0.
    with torch.no_grad():
        for start in [0,3]:
            x=ds['site'];z,_=e(x['static'][start:start+3],x['dynamic'][start:start+3])
            ce+=torch.nn.functional.cross_entropy(local(z),x['labels'][start:start+3],reduction='sum').item()
    assert report['pooled']['ce']==pytest.approx(ce/5)
    with pytest.raises(ValueError,match='overlap'):train_validation(model,ds,ds)


def test_projection_checkpoint_reload(tmp_path):
    seed_everything(17);ds=dataset();model=FederatedTrainer(small(),ds)
    model.run_round(ds);model.save(tmp_path/'model.pt')
    restored,_=FederatedTrainer.load(tmp_path/'model.pt')
    assert restored.cfg.temporal_mlp_width==16
    assert (model.predict('site',ds['site'])==restored.predict('site',ds['site'])).all()
