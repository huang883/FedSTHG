"""Small actual optimization and GPU parity checks, not metric reproduction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from dataclasses import replace
import numpy as np
import pytest
import torch
from fedsthg.config import Config
from fedsthg.data import synthetic_manifest,build_cache,make_splits
from fedsthg.experiment import load_tensors,select,evaluate
from fedsthg.federated import FederatedTrainer,seed_everything
from fedsthg.model import Encoder,make_local,task_loss


def test_real_training_pipeline_on_labelled_synthetic(tmp_path):
    torch.set_num_threads(2);seed_everything(11)
    c=Config(n_rois=8,aligned_length=32,window=8,step=4,hidden=8,
             k_sg=2,k_sh=3,k_tg=1,k_th=2,batch_size=8,dropout=0)
    manifest=synthetic_manifest(tmp_path/'data',per_site=30,n_rois=8,seed=11)
    _,rows=build_cache(manifest,c,tmp_path/'cache')
    data=load_tensors(rows,tmp_path/'cache');splits=make_splits(rows)
    train={s:select(data[s],splits[0]['clients'][s]['train']) for s in data}
    test={s:select(data[s],splits[0]['clients'][s]['test']) for s in data}
    model=FederatedTrainer(c,data)
    logs=[model.run_round(train) for _ in range(8)]
    # A functional optimization check: task CE should learn this deliberately simple signal.
    first=np.mean([v['ce'] for v in logs[0]['phase_II'].values()])
    last=np.mean([v['ce'] for v in logs[-1]['phase_II'].values()])
    assert last<first
    metrics,records=evaluate(model,test)
    assert metrics['pooled']['n']==len(records)==12
    assert all(0<=r['probability']<=1 for r in records)
    assert len(model.predict('virtual_0',test['virtual_0'],batch_size=1))==6


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_full_dimension_gpu_forward_backward_and_cpu_parity():
    seed_everything(1);torch.set_num_threads(2)
    c=replace(Config(),dropout=0)
    cpu=Encoder(c).eval();gpu=Encoder(c).cuda().eval();gpu.load_state_dict(cpu.state_dict())
    s=torch.randn(2,116,116);d=torch.randn(2,c.windows,116,116)
    with torch.no_grad():
        a,_=cpu(s,d);b,_=gpu(s.cuda(),d.cuda())
    # Floating point kernels differ. Structure is hard kNN; random inputs avoid exact ties.
    torch.testing.assert_close(a,b.cpu(),atol=2e-4,rtol=2e-4)
    gpu.train();local=make_local(c,gpu.width).cuda()
    s=torch.randn(16,116,116,device='cuda');d=torch.randn(16,c.windows,116,116,device='cuda')
    z,aux=gpu(s,d);loss,_,_=task_loss(local(z),torch.arange(16,device='cuda')%2,aux,c.consistency)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters())
