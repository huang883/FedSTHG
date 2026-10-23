"""Replay final train/validation diagnostic metrics from saved checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fedsthg.data import load_manifest, audit_splits
from fedsthg.experiment import load_tensors, select, source_hashes, dump
from fedsthg.federated import FederatedTrainer
from fedsthg.diagnostics import train_validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--out',default='results/projection_study');p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(4);root=Path(args.out)
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    _,rows=load_manifest(args.manifest);data=load_tensors(rows,args.cache)
    cfg1=read(root/'baseline/config.json');cfg2=read(root/'temporal256/config.json')
    assert {k for k in cfg1 if cfg1[k]!=cfg2[k]}=={'temporal_mlp_width'}
    reports=[]
    for name in ['baseline','temporal256']:
        folder=root/name;splits=read(folder/'splits.json');audit_splits(rows,splits)
        assert read(folder/'source_hashes.json')==source_hashes()
        for f in range(5):
            model,_=FederatedTrainer.load(folder/f'fold_{f}/checkpoint.pt',args.device,restore_rng=False)
            train={s:select(data[s],splits[f]['clients'][s]['train']) for s in data}
            val={s:select(data[s],splits[f]['clients'][s]['val']) for s in data}
            actual=train_validation(model,train,val)
            saved=read(folder/f'fold_{f}/training_log.json')[-1]['diagnostics']
            largest=0.
            for split in ['train','validation']:
                for metric in ['ACC','AUC','SEN','SPE','F1','ce']:
                    error=abs(actual[split]['pooled'][metric]-saved[split]['pooled'][metric])
                    largest=max(largest,error);assert error<1e-6,(name,f,split,metric,error)
                for site in data:
                    assert np.allclose(actual[split]['clients'][site]['gate_mean'],
                                       saved[split]['clients'][site]['gate_mean'],atol=1e-6)
            reports.append({'variant':name,'fold':f,'max_pooled_metric_error':largest})
            del model,train,val
    dump(root/'curve_audit.json',{'all_checks_passed':True,'only_config_difference':'temporal_mlp_width','folds':reports})
    print('Verified 10 final train/validation curve points against saved checkpoints.')


if __name__=='__main__':main()
