"""Auditable subject manifests, FC cache and client-wise nonoverlapping splits."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from .preprocessing import read_roi, connectivity


def load_manifest(path, verify=True):
    path = Path(path)
    doc = json.loads(path.read_text(encoding='utf-8'))
    rows = sorted([r for r in doc['records'] if r['status'] == 'ok'],
                  key=lambda r: (r['site'], r['subject_id']))
    ids = [r['subject_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate subject IDs: split at subject level, not scan level')
    for r in rows:
        if r['label'] not in {0, 1}:
            raise ValueError('Expected binary labels, patient=1')
        if verify and sha256((path.parent / r['path']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Checksum mismatch for ' + r['subject_id'])
    return doc, rows


def make_splits(rows, folds=5, seed=42, validation_fraction=.2):
    result = [{'fold': f, 'seed': seed, 'clients': {}} for f in range(folds)]
    for site in sorted({r['site'] for r in rows}):
        local = sorted([r for r in rows if r['site'] == site], key=lambda r: r['subject_id'])
        y = np.array([r['label'] for r in local])
        if min(np.bincount(y, minlength=2)) < folds:
            raise ValueError(f'{site}: too few examples per class for {folds} folds')
        for f, (dev, test) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(y, y)):
            train, val = train_test_split(dev, test_size=validation_fraction,
                                         stratify=y[dev], random_state=seed + f)
            groups = {name: sorted(local[i]['subject_id'] for i in idx)
                      for name, idx in [('train', train), ('val', val), ('test', test)]}
            result[f]['clients'][site] = groups
    audit_splits(rows, result)
    return result


def audit_splits(rows, splits):
    for site in sorted({r['site'] for r in rows}):
        all_ids = {r['subject_id'] for r in rows if r['site'] == site}
        test_ids = []
        for split in splits:
            sets = [set(split['clients'][site][p]) for p in ['train', 'val', 'test']]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError('Train/validation/test overlap')
            if set.union(*sets) != all_ids:
                raise ValueError('Split subjects differ from manifest')
            test_ids += list(sets[2])
        if len(test_ids) != len(set(test_ids)) or set(test_ids) != all_ids:
            raise ValueError('Each subject must be tested exactly once across folds')


def build_cache(manifest, cfg, out):
    """Only per-subject transforms. No dataset-level scaling/feature selection."""
    doc, rows = load_manifest(manifest)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        signature = sha256(json.dumps({'source': r['sha256'], 'M': cfg.aligned_length,
            'window': cfg.window, 'step': cfg.step, 'N': cfg.n_rois, 'algorithm': 'anchor-halfspacing-v1'},
            sort_keys=True).encode()).hexdigest()
        target = out / (r['subject_id'] + '.npz')
        if target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached['signature']) == signature:
                    continue
        x = read_roi(Path(manifest).parent / r['path'], cfg.n_rois)
        s, d = connectivity(x, cfg.aligned_length, cfg.window, cfg.step)
        np.savez_compressed(target, static=s, dynamic=d, signature=signature)
        if (i + 1) % 50 == 0:
            print(f'FC cache {i + 1}/{len(rows)}', flush=True)
    return doc, rows


def synthetic_manifest(out, n_sites=2, per_site=30, n_rois=12, seed=42):
    """Artificial covariance class signal, exclusively for software verification."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for site in range(n_sites):
        for i in range(per_site):
            label = i % 2
            t = int(rng.integers(70, 130))
            x = rng.normal(size=(t, n_rois))
            latent = rng.normal(size=(t, 1))
            x[:, :n_rois // 2] += latent * (1.8 if label else .2)
            x += .1 * site * rng.normal(size=(t, 1))
            sid = f'synthetic_{site}_{i:03d}'
            p = out / (sid + '.npy'); np.save(p, x)
            rows.append({'subject_id': sid, 'site': f'virtual_{site}', 'label': label,
                         'path': p.name, 'sha256': sha256(p.read_bytes()).hexdigest(), 'status': 'ok'})
    doc = {'dataset': 'SYNTHETIC_NOT_MEDICAL_DATA', 'synthetic': True, 'seed': seed, 'records': rows}
    p = out / 'manifest.json'; p.write_text(json.dumps(doc, indent=2), encoding='utf-8')
    return p
