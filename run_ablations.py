"""Run paper-supported variants on identical split seeds; B11 choices documented."""
import argparse
from dataclasses import replace
from pathlib import Path
from fedsthg.config import Config
from fedsthg.experiment import run

VARIANTS=['no_temporal','no_spatial','graph_only','hypergraph_only','equal_fusion','no_personalization']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--folds',type=int,nargs='+')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=Config.load(a.config)
    for variant in a.variants:
        run(replace(c,variant=variant),a.manifest,a.cache,Path(a.out)/variant,a.device,a.folds,a.resume)


if __name__=='__main__':main()
