"""Independently recompute stored metrics and optionally replay checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,csv,json
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from fedsthg.config import Config
from fedsthg.data import load_manifest,audit_splits
from fedsthg.experiment import load_tensors,select,evaluate,dump
from fedsthg.federated import FederatedTrainer
from fedsthg.metrics import binary_metrics,fold_summary,METRICS


def audit(run,manifest,cache=None,device='cpu'):
    run=Path(run);cfg=Config.load(run/'config.json')
    _,rows=load_manifest(manifest)
    lookup={r['subject_id']:r for r in rows}
    splits=json.loads((run/'splits.json').read_text(encoding='utf-8'))
    audit_splits(rows,splits)
    data=load_tensors(rows,cache) if cache else None
    manifest_hash=sha256(Path(manifest).read_bytes()).hexdigest()
    reports=[];metrics=[]
    for folder in sorted(run.glob('fold_*')):
        if not (folder/'metrics.json').exists():continue
        f=int(folder.name.split('_')[1])
        records=list(csv.DictReader((folder/'predictions.csv').open(encoding='utf-8')))
        expected={sid for client in splits[f]['clients'].values() for sid in client['test']}
        ids=[r['subject_id'] for r in records]
        assert len(ids)==len(set(ids)) and set(ids)==expected
        assert all(int(r['label'])==lookup[r['subject_id']]['label'] and
                   r['site']==lookup[r['subject_id']]['site'] for r in records)
        m=binary_metrics([int(r['label']) for r in records],[float(r['probability']) for r in records])
        saved=json.loads((folder/'metrics.json').read_text())
        for k in METRICS:assert np.isclose(m[k],saved['pooled'][k]),(f,k)
        model,provenance=FederatedTrainer.load(folder/'checkpoint.pt',device,restore_rng=False)
        assert model.round==cfg.rounds and provenance['split']==splits[f]
        assert provenance['manifest_sha256']==manifest_hash
        replay_error=None
        if data:
            test={s:select(data[s],splits[f]['clients'][s]['test']) for s in data}
            _,replayed=evaluate(model,test)
            assert [r['subject_id'] for r in replayed]==ids
            replay_error=float(np.max(np.abs(np.array([r['probability'] for r in replayed])-
                                            np.array([float(r['probability']) for r in records]))))
            assert replay_error<1e-5,replay_error
        reports.append({'fold':f,'n':len(records),'metric_recalculation':True,
                        'checkpoint_round':model.round,'replay_max_abs_error':replay_error})
        metrics.append(m);del model
    summary=json.loads((run/'summary.json').read_text())
    computed=fold_summary(metrics)
    assert len(reports)==len(summary['evaluated_folds'])
    for k in METRICS:
        assert np.isclose(computed[k]['mean'],summary['pooled'][k]['mean'])
        assert np.isclose(computed[k]['sd'],summary['pooled'][k]['sd'])
    result={'run':str(run),'folds':reports,'all_checks_passed':True,
            'complete_five_fold_run':len(reports)==cfg.folds}
    dump(run/'audit.json',result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--cache',help='Recompute predictions from saved checkpoints if supplied')
    p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);audit(a.run,a.manifest,a.cache,a.device)


if __name__=='__main__':main()
