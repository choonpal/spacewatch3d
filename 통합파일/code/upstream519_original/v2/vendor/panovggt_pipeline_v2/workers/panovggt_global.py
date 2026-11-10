"""Numerical registration primitives preserved from the audited PanoVGGT experiment."""

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Config:
    keyframe_stride: int = 3
    min_frame_gap: int = 15
    min_matches: int = 30
    min_inliers: int = 20
    min_inlier_fraction: float = 0.45
    ratio: float = 0.70
    ransac_trials: int = 700
    residual_depth_fraction: float = 0.04
    min_image_bins: int = 4
    max_loops: int = 16
    max_candidates: int = 80
    edge_sample_limit: int = 1600
    rotation_sigma_degrees: float = 2.0
    log_scale_sigma: float = 0.04
    translation_sigma_depth_fraction: float = 0.03
    max_nfev: int = 300
    seed: int = 42
    inner_train_fraction: float = .75
    min_inner_holdout: int = 4
    inner_inlier_fraction: float = .5

def apply(points, transform):
    s, r, t = transform
    return s * (np.asarray(points) @ r.T) + t

def fit_similarity(source, target):
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 4:
        raise ValueError('At least four paired 3D points are required.')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite correspondences.')
    xc, yc = x.mean(0), y.mean(0)
    xx, yy = x - xc, y - yc
    # Planar scenes are valid; collinear correspondences do not constrain rotation.
    for centered in (xx, yy):
        eig = np.linalg.svd(centered, compute_uv=False)
        if eig[0] < 1e-9 or eig[1] < 1e-3 * eig[0]:
            raise ValueError('Degenerate, nearly collinear correspondences.')
    u, singular, vt = np.linalg.svd(yy.T @ xx / len(x))
    correction = np.ones(3)
    correction[-1] = np.sign(np.linalg.det(u @ vt))
    r = (u * correction) @ vt
    s = float(np.sum(singular * correction) / np.mean(np.sum(xx**2, axis=1)))
    if not np.isfinite(s) or not .25 < s < 4:
        raise ValueError('Implausible relative scale.')
    return s, r, yc - s * (r @ xc)

def robust_fit(source, target, threshold, cfg, seed_offset=0):
    """RANSAC with an independent holdout; fit never uses holdout points."""
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('Invalid residual threshold.')
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < cfg.min_matches:
        raise ValueError('Too few 3D matches.')
    rng = np.random.default_rng(cfg.seed + seed_offset)
    order = rng.permutation(len(source)); split = max(4, int(cfg.inner_train_fraction * len(order)))
    train, held = order[:split], order[split:]
    x, y = source[train], target[train]
    best = None; best_count = 0
    for _ in range(cfg.ransac_trials):
        ids = rng.choice(len(x), 4, replace=False)
        try:
            transform = fit_similarity(x[ids], y[ids])
        except ValueError:
            continue
        keep = np.linalg.norm(apply(x, transform)-y, axis=1) < threshold
        if int(keep.sum()) > best_count:
            best, best_count = keep, int(keep.sum())
    if best is None or best_count < max(8, int(cfg.min_inlier_fraction * len(train))):
        raise ValueError('RANSAC has insufficient consensus.')
    for _ in range(3):
        transform = fit_similarity(x[best], y[best])
        best = np.linalg.norm(apply(x, transform)-y, axis=1) < threshold
        if best.sum() < 4:
            raise ValueError('Consensus collapsed during refit.')
    transform = fit_similarity(x[best], y[best])
    errors = np.linalg.norm(apply(source, transform)-target, axis=1)
    inliers = errors < threshold
    if inliers.sum() < cfg.min_inliers or np.mean(inliers) < cfg.min_inlier_fraction:
        raise ValueError('Insufficient 3D inlier coverage.')
    if len(held) < cfg.min_inner_holdout or np.mean(inliers[held]) < cfg.inner_inlier_fraction or np.median(errors[held]) > threshold:
        raise ValueError('Independent 3D holdout validation failed.')
    return transform, inliers, dict(matches=len(source), inliers=int(inliers.sum()),
        inlier_fraction=float(inliers.mean()), median_inlier_residual=float(np.median(errors[inliers])),
        holdout_count=len(held), holdout_inlier_fraction=float(inliers[held].mean()),
        holdout_median_residual=float(np.median(errors[held])), holdout_indices=held.tolist(),
        threshold=float(threshold))

def owners(window_indices):
    result={}
    for wi,ids in enumerate(window_indices):
        for local,frame in enumerate(ids):
            score=abs(local-(len(ids)-1)/2)
            if frame not in result or score < result[frame][0]:
                result[frame]=(score,wi,local)
    return result
