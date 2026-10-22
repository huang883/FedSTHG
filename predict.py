"""Infer a participating site's saved personalized model on ROI files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fedsthg.federated import FederatedTrainer
from fedsthg.preprocessing import read_roi,connectivity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--site',required=True)
    p.add_argument('--roi',nargs='+',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.batch_size<1:p.error('batch size must be positive')
    torch.set_num_threads(4)
    trainer,provenance=FederatedTrainer.load(a.checkpoint,a.device,restore_rng=False)
    c=trainer.cfg;static=[];dynamic=[]
    for path in a.roi:
        s,d=connectivity(read_roi(path,c.n_rois),c.aligned_length,c.window,c.step)
        static.append(s);dynamic.append(d)
    data={'static':torch.from_numpy(np.stack(static)),'dynamic':torch.from_numpy(np.stack(dynamic)),
          'labels':torch.zeros(len(a.roi),dtype=torch.long)}  # length only; never supplied to model
    prob=trainer.predict(a.site,data,a.batch_size)
    result={'site':a.site,'rounds':trainer.round,'batch_size':a.batch_size,
            'single_subject_bypasses_refinement':a.batch_size==1,
            'research_use_not_clinical_diagnosis':True,
            'predictions':[{'roi':path,'patient_probability':float(value),'prediction':int(value>=.5)}
                           for path,value in zip(a.roi,prob)]}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
