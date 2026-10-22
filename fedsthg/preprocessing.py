"""Subject-local ROI -> FC processing, Eqs. (2)-(7).

This module accepts already preprocessed ROI BOLD. It does NOT claim to perform
slice timing, motion correction or registration on raw NIfTI.
"""
from pathlib import Path
import numpy as np


def read_roi(path, n_rois=116):
    path = Path(path)
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    else:
        x = np.loadtxt(path, comments="#", delimiter="," if path.suffix == ".csv" else None)
    return validate_roi(x, n_rois)


def validate_roi(x, n_rois=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("ROI input must be [time>=2, regions>=2]")
    if n_rois is not None and x.shape[1] != n_rois:
        raise ValueError(f"Expected {n_rois} ROI columns, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite BOLD input; fix/QC upstream, do not silently impute")
    return x


def anchor_align(x, length=116):
    """B02: uniform anchors, half-spacing local mean; nearest sample if empty.

    T==M is identity. Shorter scans repeat samples; not interpolation and not
    information-preserving. No statistics from other subjects are used.
    """
    x = validate_roi(x)
    if length < 2:
        raise ValueError("At least two anchors required")
    centers = np.linspace(0, len(x) - 1, length)
    radius = (len(x) - 1) / (2 * (length - 1))
    out = []
    for center in centers:
        lo = max(0, int(np.ceil(center - radius)))
        hi = min(len(x) - 1, int(np.floor(center + radius)))
        out.append(x[lo:hi + 1].mean(0) if lo <= hi else x[int(np.floor(center + .5))])
    return np.stack(out)


def normalize_roi(x, eps=1e-8):
    x = validate_roi(x)
    return (x - x.mean(0)) / (x.std(0, ddof=0) + eps)


def fisher_fc(x, clip_eps=1e-6):
    """Pearson with defined zero covariance for constant ROI, Fisher z, zero diag."""
    x = validate_roi(x)
    if not 0 < clip_eps < 1:
        raise ValueError("clip_eps must lie in (0,1)")
    xc = x - x.mean(0)
    norms = np.linalg.norm(xc, axis=0)
    denom = np.outer(norms, norms)
    corr = np.divide(xc.T @ xc, denom, out=np.zeros_like(denom), where=denom > 0)
    z = np.arctanh(np.clip(corr, -1 + clip_eps, 1 - clip_eps))
    np.fill_diagonal(z, 0)
    return z.astype(np.float32)


def connectivity(x, aligned_length=116, window=30, step=2):
    if not (2 <= window <= aligned_length and step > 0):
        raise ValueError("Invalid sliding window configuration")
    aligned = normalize_roi(anchor_align(x, aligned_length))
    static = fisher_fc(aligned)
    dynamic = np.stack([fisher_fc(aligned[t:t + window])
                        for t in range(0, aligned_length - window + 1, step)])
    return static, dynamic
