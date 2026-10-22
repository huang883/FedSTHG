"""Sequential simulation of Eqs.35-37, preserving three distinct parameter paths.

Clients run in one process for reproducible experiments, not a deployed network.
Only encoder state_dicts enter the server aggregator in full FedSTHG.
"""
import copy
from dataclasses import asdict
import random
import numpy as np
import torch
from .config import Config
from .model import Encoder, make_local, task_loss


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def clone_state(module):
    return {k: v.detach().clone() for k,v in module.state_dict().items()}


def aggregate(states, sizes, expected_keys):
    if not states or len(states) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError('Need one positive training sample count per state')
    keys=set(expected_keys)
    if any(set(s) != keys for s in states):
        raise ValueError('Aggregation schema mismatch: local parameters must not enter encoder aggregation')
    total=sum(sizes)
    result={}
    for key in states[0]:
        if not states[0][key].is_floating_point():
            if not all(torch.equal(s[key], states[0][key]) for s in states):
                raise ValueError('Nonfloating buffers differ')
            result[key]=states[0][key].clone()
        else:
            result[key]=sum((s[key].detach() * (n/total) for s,n in zip(states,sizes)),
                            torch.zeros_like(states[0][key]))
    return result


def proximal_penalty(encoder, reference):
    return sum((p-reference[name].detach()).square().sum()
               for name,p in encoder.named_parameters())


class FederatedTrainer:
    def __init__(self, cfg: Config, client_names, device='cpu'):
        self.cfg=cfg.validate(); self.device=torch.device(device)
        self.names=sorted(client_names)
        self.global_encoder=Encoder(cfg).to(self.device).requires_grad_(False)
        self.personal={k:copy.deepcopy(self.global_encoder).requires_grad_(True) for k in self.names}
        self.local={k:make_local(cfg,self.global_encoder.width).to(self.device) for k in self.names}
        self.global_local=copy.deepcopy(self.local[self.names[0]]) if cfg.variant=='no_personalization' else None
        self.round=0; self.history=[]

    def _epoch(self, encoder, local, data, reference=None):
        encoder.train(); local.train()
        params=list(encoder.parameters())+list(local.parameters())
        # B09: new optimizer per LocalOptimize; model states, not moments, persist.
        opt=torch.optim.AdamW(params,lr=self.cfg.lr,weight_decay=self.cfg.weight_decay)
        order=torch.randperm(len(data['labels']))
        sums={'loss':0., 'ce':0., 'st':0., 'proximal':0.}
        for idx in order.split(self.cfg.batch_size):
            s=data['static'][idx].to(self.device); d=data['dynamic'][idx].to(self.device)
            y=data['labels'][idx].to(self.device)
            opt.zero_grad(set_to_none=True)
            z,aux=encoder(s,d)
            loss,ce,st=task_loss(local(z),y,aux,self.cfg.consistency)
            prox=loss.new_zeros(()) if reference is None else \
                .5*self.cfg.proximal*proximal_penalty(encoder,reference)
            loss=loss+prox
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; training stopped')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,self.cfg.clip,error_if_nonfinite=True)
            opt.step()
            for name,value in [('loss',loss),('ce',ce),('st',st),('proximal',prox)]:
                sums[name]+=float(value.detach())*len(idx)
        return {k:v/len(order) for k,v in sums.items()}

    def phase_one(self, datasets):
        uploads=[]; sizes=[]; stats={}; local_uploads=[]
        for site in self.names:
            temp=copy.deepcopy(self.global_encoder).requires_grad_(True)
            if self.global_local is not None:
                self.local[site].load_state_dict(self.global_local.state_dict())
            stats[site]=self._epoch(temp,self.local[site],datasets[site])
            uploads.append(clone_state(temp)); sizes.append(len(datasets[site]['labels']))
            if self.global_local is not None:
                local_uploads.append(clone_state(self.local[site]))
        self.global_encoder.load_state_dict(aggregate(uploads,sizes,self.global_encoder.state_dict()))
        if self.global_local is not None:
            # B11 ablation ONLY: global head, no adapter/refinement or Phase II.
            self.global_local.load_state_dict(aggregate(local_uploads,sizes,self.global_local.state_dict()))
        return stats

    def phase_two(self, datasets):
        # Immutable detached reference is the NEW aggregated model, not old global.
        reference={name:p.detach().clone() for name,p in self.global_encoder.named_parameters()}
        return {site:self._epoch(self.personal[site],self.local[site],datasets[site],reference)
                for site in self.names}

    def run_round(self, datasets):
        if set(datasets)!=set(self.names):
            raise ValueError('Client set mismatch')
        shared=self.phase_one(datasets)
        personalized=self.phase_two(datasets) if self.global_local is None else {}
        self.round+=1
        log={'round':self.round,'phase_I':shared,'phase_II':personalized}
        self.history.append(log)
        return log

    @torch.no_grad()
    def predict(self, site, data, batch_size=None):
        if site not in self.local:
            raise ValueError('Unseen site has no personalized model')
        encoder=self.personal[site] if self.global_local is None else self.global_encoder
        local=self.local[site] if self.global_local is None else self.global_local
        encoder.eval(); local.eval()
        probs=[]
        for start in range(0,len(data['labels']),batch_size or self.cfg.batch_size):
            end=start+(batch_size or self.cfg.batch_size)
            z,_=encoder(data['static'][start:end].to(self.device),data['dynamic'][start:end].to(self.device))
            probs.append(local(z).softmax(-1)[:,1].cpu())
        return torch.cat(probs).numpy()

    def save(self,path,provenance=None):
        """Round-boundary checkpoint. All states needed for exact continuation."""
        torch.save({'config':asdict(self.cfg),'names':self.names,'round':self.round,
            'global':self.global_encoder.state_dict(),
            'personal':{k:m.state_dict() for k,m in self.personal.items()},
            'local':{k:m.state_dict() for k,m in self.local.items()},
            'global_local':None if self.global_local is None else self.global_local.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'history':self.history,'provenance':provenance},path)

    @classmethod
    def load(cls,path,device='cpu',restore_rng=True):
        state=torch.load(path,map_location='cpu',weights_only=True)
        obj=cls(Config(**state['config']),state['names'],device)
        obj.global_encoder.load_state_dict(state['global'])
        for k in obj.names:
            obj.personal[k].load_state_dict(state['personal'][k]);obj.local[k].load_state_dict(state['local'][k])
        if obj.global_local is not None:
            obj.global_local.load_state_dict(state['global_local'])
        obj.round=state['round'];obj.history=state['history']
        if restore_rng:
            torch.set_rng_state(state['torch_rng'])
            if state['cuda_rng'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        return obj,state.get('provenance')
