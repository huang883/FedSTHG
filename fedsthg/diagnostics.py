"""Read-only, same-protocol train/validation evaluation; never uses test data."""
import numpy as np
import torch
from torch.nn import functional as F
from .metrics import binary_metrics


@torch.no_grad()
def diagnose(trainer, datasets):
    clients = {}; all_y = []; all_p = []; total_ce = 0.; total_n = 0
    # Preserve mode flags (including individual children) and RNG state.
    modules = [*trainer.personal.values(), *trainer.local.values()]
    flags = {m: m.training for root in modules for m in root.modules()}
    devices = list(range(torch.cuda.device_count())) if trainer.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            for site, data in datasets.items():
                encoder = trainer.personal[site]; local = trainer.local[site]
                encoder.eval(); local.eval()
                probabilities = []; ce = 0.; gate_sum = torch.zeros(2, device=trainer.device)
                st = 0.; n = len(data['labels'])
                for start in range(0, n, trainer.cfg.batch_size):
                    end = min(start + trainer.cfg.batch_size, n)
                    y = data['labels'][start:end].to(trainer.device)
                    z, aux = encoder(data['static'][start:end].to(trainer.device),
                                     data['dynamic'][start:end].to(trainer.device))
                    logits = local(z)
                    ce += F.cross_entropy(logits, y, reduction='sum').item()
                    probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
                    gate_sum += aux['gate'].sum(0)
                    st += aux['consistency'].sum().item()
                y = data['labels'].numpy()
                clients[site] = {**binary_metrics(y, probabilities), 'ce': ce/n,
                                 'gate_mean': (gate_sum/n).cpu().tolist(), 'consistency': st/n}
                all_y.extend(y.tolist()); all_p.extend(probabilities)
                total_ce += ce; total_n += n
    finally:
        for module, flag in flags.items(): module.training = flag
    return {'pooled': {**binary_metrics(all_y, all_p), 'ce': total_ce/total_n},
            'clients': clients, 'dropout': False, 'batch_size': trainer.cfg.batch_size,
            'order': 'saved_split_order', 'checkpoint': 'persistent_personalized'}


def train_validation(trainer, train, validation):
    if trainer.cfg.variant != 'full':
        raise ValueError('Projection diagnostics are defined for full personalized models')
    train_ids = {sid for data in train.values() for sid in data['ids']}
    validation_ids = {sid for data in validation.values() for sid in data['ids']}
    if train_ids & validation_ids:
        raise ValueError('Train/validation subjects overlap')
    return {'round': trainer.round, 'train': diagnose(trainer, train),
            'validation': diagnose(trainer, validation)}
