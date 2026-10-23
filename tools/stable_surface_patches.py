"""Geometry-only reference candidates of any orientation, without instance labels."""
import numpy as np
from scipy.spatial import cKDTree
from local_motion_error import normal_families

PATCH_CONFIG = {
    'cell_m': .35, 'minimum_samples': 18, 'minimum_pairs': 12,
    'minimum_tangent_span_m': .12, 'maximum_thickness_m': .025,
    'maximum_normal_angle_deg': 12., 'maximum_pair_distance_m': .12,
    'minimum_bidirectional_coverage': .35, 'minimum_direction_patches': 3,
    'minimum_direction_train_blocks': 3, 'minimum_direction_check_blocks': 2,
    'minimum_direction_span_m': .5,
}


def _plane(points, normals, cfg):
    if len(points) < cfg['minimum_samples']:
        return None
    center = np.median(points, axis=0); keep = np.ones(len(points), bool)
    for _ in range(3):
        p = points[keep]; center = p.mean(0)
        values, vectors = np.linalg.eigh((p-center).T @ (p-center)/len(p))
        normal = vectors[:, 0]
        keep = (np.abs((points-center) @ normal) <= cfg['maximum_thickness_m'])
        keep &= np.abs(normals @ normal) >= np.cos(np.deg2rad(20))
        if keep.sum() < cfg['minimum_samples']:
            return None
    p = points[keep]; center = p.mean(0)
    values, vectors = np.linalg.eigh((p-center).T @ (p-center)/len(p))
    if values[1] <= 1e-6 or values[0]/values[1] > .12:
        return None
    basis = vectors[:, 1:]; uv = (p-center) @ basis
    low, high = np.quantile(uv, [.05, .95], axis=0)
    if np.min(high-low) < cfg['minimum_tangent_span_m']:
        return None
    normal = vectors[:, 0]
    return {'center': center, 'normal': normal, 'basis': basis, 'low': low, 'high': high, 'keep': keep}


def discover_stable_patches(before, after, before_normals, after_normals, config=None):
    """Paired local planar pieces; 'stable' is a screened hypothesis, not a label.

    Only XYZ and surface normals are accepted. Plane offset is not forced to
    zero: a common registration shift must remain measurable by the next stage.
    """
    cfg = {**PATCH_CONFIG, **(config or {})}
    before, after, before_normals, after_normals = map(lambda a: np.asarray(a, float), (before, after, before_normals, after_normals))
    tree = cKDTree(after); cells, membership = np.unique(np.floor(before/cfg['cell_m']).astype(int), axis=0, return_inverse=True)
    order = np.argsort(membership, kind='stable'); groups = np.split(order, np.flatnonzero(np.diff(membership[order]))+1)
    source, target, normals, patch_ids, catalog = [], [], [], [], []
    for cell, ids in zip(cells, groups):
        model = _plane(before[ids], before_normals[ids], cfg)
        if model is None:
            continue
        ids = ids[model['keep']]; p = before[ids]
        center, normal, basis = model['center'], model['normal'], model['basis']
        neighbors = np.array(tree.query_ball_point(center, cfg['cell_m']*1.1+cfg['maximum_pair_distance_m']), int)
        if len(neighbors) < cfg['minimum_samples']:
            continue
        q = after[neighbors]; uv = (q-center) @ basis
        compatible = (np.abs((q-center) @ normal) <= cfg['maximum_pair_distance_m'])
        compatible &= np.abs(after_normals[neighbors] @ normal) >= np.cos(np.deg2rad(cfg['maximum_normal_angle_deg']))
        compatible &= ((uv >= model['low']-.035) & (uv <= model['high']+.035)).all(1)
        neighbors = neighbors[compatible]
        second = _plane(after[neighbors], after_normals[neighbors], cfg)
        if second is None or abs(second['normal'] @ normal) < np.cos(np.deg2rad(cfg['maximum_normal_angle_deg'])):
            continue
        neighbors = neighbors[second['keep']]; q = after[neighbors]
        distance, nearest = cKDTree(p).query(q); reverse = cKDTree(q).query(p)[1]
        paired = (distance <= cfg['maximum_pair_distance_m']) & (reverse[nearest] == np.arange(len(q)))
        count = int(paired.sum())
        coverage = [count/len(p), count/len(q)]
        if count < cfg['minimum_pairs'] or min(coverage) < cfg['minimum_bidirectional_coverage']:
            continue
        a, b = q[paired], p[nearest[paired]]
        n = normal+second['normal']*(1 if normal @ second['normal'] >= 0 else -1)
        n /= np.linalg.norm(n)
        # Reject patch mixtures, but retain a coherent nonzero normal offset.
        residual = (a-b) @ n
        if np.quantile(np.abs(residual-np.median(residual)), .95) > cfg['maximum_thickness_m']:
            continue
        identifier = len(catalog)
        corners = [center+basis @ np.array([u, v]) for u, v in [(model['low'][0], model['low'][1]), (model['high'][0], model['low'][1]), (model['high'][0], model['high'][1]), (model['low'][0], model['high'][1])]]
        catalog.append({'id': identifier, 'normal': n.tolist(), 'center': ((a+b)/2).mean(0).tolist(),
            'corners': np.array(corners).tolist(), 'pair_count': count, 'coverage': coverage,
            'normal_offset_m': float(np.median(residual)), 'cell': cell.tolist()})
        source.append(a); target.append(b); normals.append(np.tile(n, (count, 1))); patch_ids.extend([identifier]*count)
    if not catalog:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.array([], int), []
    return np.concatenate(source), np.concatenate(target), np.concatenate(normals), np.array(patch_ids), catalog


def screen_local_directions(source, target, normals, patch_ids, pivot, config=None):
    """Require independent support for every newly introduced normal family.

    Negative patch IDs are the original structural references. Their existing
    directions are preserved; extra patches cannot manufacture a third axis.
    """
    cfg = {**PATCH_CONFIG, **(config or {})}
    source, target, normals, patch_ids = map(np.asarray, (source, target, normals, patch_ids))
    snapped, families = normal_families(normals)
    cells = np.floor(((source+target)/2-np.asarray(pivot))/.35).astype(np.int64)
    holdout = ((cells @ np.array([73856093, 19349663, 83492791])) % 5) == 0
    keep = np.ones(len(source), bool); diagnostics = []
    for f in np.unique(families):
        chosen = families == f
        old_direction = bool(np.any(chosen & (patch_ids < 0)))
        ids = np.unique(patch_ids[chosen & (patch_ids >= 0)])
        train = len(np.unique(cells[chosen & ~holdout], axis=0)); check = len(np.unique(cells[chosen & holdout], axis=0))
        centers = np.array([((source+target)/2)[chosen & (patch_ids == i)].mean(0) for i in ids])
        span = float(np.linalg.norm(np.ptp(centers, axis=0))) if len(centers) else 0.
        admitted = old_direction or (len(ids) >= cfg['minimum_direction_patches'] and train >= cfg['minimum_direction_train_blocks'] and check >= cfg['minimum_direction_check_blocks'] and span >= cfg['minimum_direction_span_m'])
        if not admitted:
            keep[chosen] = False
        diagnostics.append({'normal': snapped[np.flatnonzero(chosen)[0]].tolist(), 'original_direction': old_direction,
            'patch_count': len(ids), 'train_blocks': train, 'check_blocks': check, 'span_m': span, 'admitted': admitted})
    return keep, diagnostics


def heldout_direction_checks(source, target, normals, patch_ids, pivot, model):
    """A new direction must predict held-out surface pieces, not just fit them."""
    if not model.get('usable'):
        return []
    source, target, normals, patch_ids = map(np.asarray, (source, target, normals, patch_ids))
    _, families = normal_families(normals)
    cells = np.floor(((source+target)/2-np.asarray(pivot))/.35).astype(np.int64)
    check = ((cells @ np.array([73856093, 19349663, 83492791])) % 5) == 0
    correction = np.array(model['correction_after_to_before'])
    moved = source @ correction[:3, :3].T+correction[:3, 3]
    residual = np.abs(np.einsum('ij,ij->i', normals, moved-target))
    results = []
    for f in np.unique(families):
        chosen = families == f
        if np.any(chosen & (patch_ids < 0)):
            continue
        train = residual[chosen & ~check]; test = residual[chosen & check]
        fit = float(np.quantile(train, .95, method='inverted_cdf')) if len(train) else None
        heldout = float(np.quantile(test, .95, method='inverted_cdf')) if len(test) else None
        # This gate is fixed from the patch thickness criterion, not object motion or labels.
        limit = max(2*fit, PATCH_CONFIG['maximum_thickness_m']) if fit is not None else 0.
        results.append({'family': int(f), 'fit_p95_m': fit, 'check_p95_m': heldout,
            'check_limit_m': limit, 'passed': bool(heldout is not None and heldout <= limit)})
    return results
