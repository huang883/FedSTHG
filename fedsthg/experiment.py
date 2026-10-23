"""Train/evaluate actual client-wise folds; log all predictions and provenance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .config import Config
from .data import build_cache, make_splits, load_manifest, synthetic_manifest
from .federated import FederatedTrainer, seed_everything
from .metrics import binary_metrics, fold_summary, METRICS
from .diagnostics import train_validation


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def environment():
    import scipy,sklearn
    return {'python':platform.python_version(),'os':platform.platform(),
            'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
            'deterministic':True,'tf32':False,'cpu_threads':torch.get_num_threads()}


def source_hashes():
    return {p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def load_tensors(rows, cache):
    result={}
    for site in sorted({r['site'] for r in rows}):
        subset=[r for r in rows if r['site']==site]
        static=[];dynamic=[]
        for r in subset:
            with np.load(Path(cache)/(r['subject_id']+'.npz'),allow_pickle=False) as z:
                static.append(z['static']);dynamic.append(z['dynamic'])
        result[site]={'static':torch.from_numpy(np.stack(static)),
                      'dynamic':torch.from_numpy(np.stack(dynamic)),
                      'labels':torch.tensor([r['label'] for r in subset],dtype=torch.long),
                      'ids':[r['subject_id'] for r in subset]}
    return result


def select(data, ids):
    lookup={sid:i for i,sid in enumerate(data['ids'])}
    idx=torch.tensor([lookup[sid] for sid in ids])
    return {**{k:data[k][idx] for k in ['static','dynamic','labels']},'ids':ids}


def evaluate(trainer,datasets):
    metrics={};records=[]
    for site,data in datasets.items():
        p=trainer.predict(site,data)
        y=data['labels'].numpy()
        metrics[site]=binary_metrics(y,p)
        records.extend({'site':site,'subject_id':sid,'label':int(label),'probability':float(prob),
                        'prediction':int(prob>=.5)} for sid,label,prob in zip(data['ids'],y,p))
    pooled=binary_metrics([r['label'] for r in records],[r['probability'] for r in records])
    macro={k:float(np.mean([v[k] for v in metrics.values() if v[k] is not None]))
           if any(v[k] is not None for v in metrics.values()) else None for k in METRICS}
    return {'pooled':pooled,'macro_client':macro,'clients':metrics},records


def run(cfg,manifest,cache,out,device='cpu',folds=None,resume=False):
    cfg.validate();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(cfg.seed)
    doc,rows=build_cache(manifest,cfg,cache)
    splits=make_splits(rows,cfg.folds,cfg.seed,cfg.validation_fraction)
    cfg.save(out/'config.json');dump(out/'splits.json',splits)
    dump(out/'environment.json',environment());dump(out/'source_hashes.json',source_hashes())
    dump(out/'cohort.json',{'dataset':doc['dataset'],'synthetic':doc['synthetic'],
         'manifest_sha256':sha256(Path(manifest).read_bytes()).hexdigest(),'subjects':len(rows),
         'counts':{s:sum(r['site']==s for r in rows) for s in sorted({r['site'] for r in rows})},
         'limitations':['B01-B12/C01 in docs/research.md apply','Not an exact author cohort/pipeline reproduction']})
    data=load_tensors(rows,cache);results=[]
    for f in (range(cfg.folds) if folds is None else folds):
        if f<0 or f>=cfg.folds:raise ValueError('Invalid fold')
        folder=out/f'fold_{f}';folder.mkdir(exist_ok=True)
        provenance={'manifest_sha256':sha256(Path(manifest).read_bytes()).hexdigest(),
                    'split':splits[f],'source_hashes':source_hashes()}
        train={s:select(data[s],splits[f]['clients'][s]['train']) for s in data}
        val={s:select(data[s],splits[f]['clients'][s]['val']) for s in data}
        test={s:select(data[s],splits[f]['clients'][s]['test']) for s in data}
        checkpoint=folder/'checkpoint.pt'
        if checkpoint.exists():
            if not resume:raise FileExistsError(f'{checkpoint}: use a new run directory or --resume')
            trainer,saved=FederatedTrainer.load(checkpoint,device)
            if saved!=provenance or asdict(trainer.cfg)!=asdict(cfg):
                raise ValueError('Resume provenance/config mismatch')
        else:
            seed_everything(cfg.seed+f);trainer=FederatedTrainer(cfg,data,device)
        start=time.perf_counter()
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        if cfg.diagnostic_interval and trainer.round == 0:
            dump(folder/'initial_diagnostics.json', train_validation(trainer,train,val))
        for r in range(trainer.round,cfg.rounds):
            log=trainer.run_round(train)
            if (r+1)%5==0 or r==cfg.rounds-1:
                log['validation']=evaluate(trainer,val)[0]
                print(f'fold={f} round={r+1}/{cfg.rounds} val_ACC={log["validation"]["pooled"]["ACC"]:.4f}',flush=True)
            if cfg.diagnostic_interval and ((r+1)%cfg.diagnostic_interval==0 or r==cfg.rounds-1):
                log['diagnostics']=train_validation(trainer,train,val)
            # Atomic round-boundary save. Optimizers are reset by design at next epoch.
            temporary=folder/'checkpoint.tmp';trainer.save(temporary,provenance);temporary.replace(checkpoint)
            dump(folder/'training_log.json',trainer.history)
        # Test evaluated only after the prescribed final round; never selected by test.
        metrics,records=evaluate(trainer,test)
        metrics['fold']=f;metrics['rounds']=trainer.round
        metrics['elapsed_this_invocation_s']=time.perf_counter()-start
        metrics['peak_gpu_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None
        metrics['encoder_parameters']=sum(p.numel() for p in trainer.global_encoder.parameters())
        metrics['inference_parameters']=metrics['encoder_parameters']+sum(p.numel() for p in trainer.local[trainer.names[0]].parameters())
        dump(folder/'metrics.json',metrics)
        with (folder/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        results.append(metrics)
        print(f'FINAL fold={f} ACC={metrics["pooled"]["ACC"]:.4f} AUC={metrics["pooled"]["AUC"]}',flush=True)
        del trainer,train,val,test
    summary={'dataset':doc['dataset'],'synthetic':doc['synthetic'],'variant':cfg.variant,
             'evaluated_folds':[m['fold'] for m in results],'expected_folds':cfg.folds,
             'rounds':cfg.rounds,'seed':cfg.seed,
             'pooled':fold_summary([m['pooled'] for m in results]),
             'macro_client':fold_summary([m['macro_client'] for m in results]),
             'strict_paper_reproduction':False}
    dump(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--manifest')
    parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--folds',nargs='+',type=int)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--synthetic-dir')
    args=parser.parse_args();cfg=Config.load(args.config)
    manifest=args.manifest
    if args.synthetic_dir:
        if manifest:parser.error('Choose real manifest OR synthetic-dir')
        manifest=synthetic_manifest(args.synthetic_dir,n_rois=cfg.n_rois,seed=cfg.seed)
    if not manifest:parser.error('--manifest or --synthetic-dir required')
    run(cfg,manifest,args.cache,args.out,args.device,args.folds,args.resume)


if __name__=='__main__':main()
