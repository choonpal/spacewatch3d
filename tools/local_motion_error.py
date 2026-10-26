"""Compare object motion with a local structural-background error model.

The background is assumed stable, not ground truth. Parallel planes never
constrain tangential translation; unresolved directions stay unresolved.
All reported envelopes are empirical screening bounds, not probabilities.
"""
import numpy as np
from scipy.spatial.transform import Rotation

CONFIG = {
    'radii_m': [1., 1.5, 2.5], 'object_exclusion_margin_m': .15,
    'block_size_m': .35, 'minimum_pairs': 80, 'minimum_blocks': 12,
    'minimum_check_pairs': 15, 'minimum_check_blocks': 3,
    'normal_family_angle_deg': 12., 'rank_relative_tolerance': .01,
    'maximum_correction_m': .15, 'maximum_correction_rotation_deg': 5.,
    'minimum_motion_m': .08, 'minimum_rotation_deg': 15.,
    'minimum_object_overlap': .55, 'maximum_object_fit_residual_m': .04,
    'bootstrap_samples': 24, 'seed': 20261019,
}


def skew(p):
    x, y, z = p
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def normal_families(normals, angle_deg=12.):
    """Collapse near-parallel structural normals to avoid spurious constraints."""
    normals = np.asarray(normals, float).copy()
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    families, membership = [], []
    for normal in normals:
        matches = [i for i, a in enumerate(families) if abs(a @ normal) >= np.cos(np.deg2rad(angle_deg))]
        if matches:
            membership.append(matches[0])
        else:
            families.append(normal); membership.append(len(families)-1)
    return np.array([families[i] for i in membership]), np.asarray(membership)


def _fit(a, b, n, weights):
    rotation, translation = np.eye(3), np.zeros(3)
    for _ in range(12):
        moved = a @ rotation.T + translation
        residual = np.einsum('ij,ij->i', n, moved-b)
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))), 1e-5)
        robust = np.minimum(1., 1.5*scale/np.maximum(np.abs(residual), 1e-12))
        design = np.column_stack([n, np.cross(moved, n)])
        root = np.sqrt(weights*robust)
        step = np.linalg.lstsq(design*root[:, None], -residual*root, rcond=.01)[0]
        update = Rotation.from_rotvec(step[3:]).as_matrix()
        rotation = update @ rotation
        translation = update @ translation + step[:3]
        if np.linalg.norm(step) < 1e-7:
            break
    return rotation, translation


def _projector(null, point):
    influence = np.column_stack([np.eye(3), -skew(point)])
    unresolved = influence @ null
    if not unresolved.size:
        return np.eye(3), influence
    u, s, _ = np.linalg.svd(unresolved, full_matrices=False)
    basis = u[:, s > max(s.max()*1e-6, 1e-8)]
    return np.eye(3)-basis @ basis.T, influence


def local_model(source, target, normals, pivot, query, config=None):
    """Fit after->before correction outside the object, with spatial holdouts."""
    cfg = {**CONFIG, **(config or {})}
    a, b, n = map(lambda x: np.asarray(x, float), (source, target, normals))
    pivot, query = np.asarray(pivot, float), np.asarray(query, float)
    result = {'usable': False, 'pair_count': len(a), 'reason': 'insufficient_background'}
    if len(a) < cfg['minimum_pairs']:
        return result
    n, family = normal_families(n, cfg['normal_family_angle_deg'])
    blocks = np.floor(((a+b)/2-pivot)/cfg['block_size_m']).astype(np.int64)
    cells, membership, sizes = np.unique(blocks, axis=0, return_inverse=True, return_counts=True)
    result.update(block_count=len(cells), normal_family_count=len(np.unique(family)))
    if len(cells) < cfg['minimum_blocks']:
        return result
    # Spatial cells, not random correlated mesh samples, are held out together.
    check_cells = ((cells @ np.array([73856093, 19349663, 83492791])) % 5) == 0
    check = check_cells[membership]; train = ~check
    result.update(check_pair_count=int(check.sum()), check_block_count=int(check_cells.sum()))
    if check.sum() < cfg['minimum_check_pairs'] or check_cells.sum() < cfg['minimum_check_blocks']:
        result['reason'] = 'insufficient_spatial_holdout'; return result
    weights = 1/sizes[membership]
    for f in np.unique(family):
        total = weights[(family == f) & train].sum()
        if total <= 0:
            result['reason'] = 'insufficient_spatial_holdout'; return result
        weights[family == f] /= total
    centered_a, centered_b = a-pivot, b-pivot
    rotation, translation = _fit(centered_a[train], centered_b[train], n[train], weights[train])
    correction_angle = float(np.rad2deg(Rotation.from_matrix(rotation).magnitude()))
    if np.linalg.norm(translation) > cfg['maximum_correction_m'] or correction_angle > cfg['maximum_correction_rotation_deg']:
        result['reason'] = 'background_correction_too_large'; return result
    moved = centered_a @ rotation.T + translation
    residual = np.abs(np.einsum('ij,ij->i', n, moved-centered_b))
    # Keep within-cell tails; do not shrink correlated mesh noise with sqrt(n).
    cell_residual = np.array([np.quantile(residual[membership == i], .95, method='inverted_cdf') for i in range(len(cells))])
    fit_p95 = float(np.quantile(cell_residual[~check_cells], .95, method='inverted_cdf'))
    check_p95 = float(np.quantile(cell_residual[check_cells], .95, method='inverted_cdf'))
    noise = max(fit_p95, check_p95)
    design = np.column_stack([n[train], np.cross(moved[train], n[train])])
    root = np.sqrt(weights[train]/weights[train].sum())
    _, singular, vt = np.linalg.svd(design*root[:, None], full_matrices=False)
    observable = singular > max(singular[0]*cfg['rank_relative_tolerance'], 1e-8)
    null = vt[~observable].T
    information_inverse = (vt[observable].T / singular[observable]**2) @ vt[observable]
    q = rotation @ (query-pivot)+translation
    projection, influence = _projector(null, q)
    dimension = int(round(np.trace(projection)))
    amplification = projection @ influence @ information_inverse @ influence.T @ projection
    geometry_bound = noise*np.sqrt(max(float(np.linalg.eigvalsh(amplification)[-1]), 0.))
    rng = np.random.default_rng(cfg['seed'])
    train_cells = np.flatnonzero(~check_cells); variations = []; rotation_variations = []
    for _ in range(cfg['bootstrap_samples']):
        counts = np.bincount(rng.choice(train_cells, len(train_cells), replace=True), minlength=len(cells))
        w = weights*counts[membership]; chosen = w > 0
        r, t = _fit(centered_a[chosen], centered_b[chosen], n[chosen], w[chosen])
        variations.append(np.linalg.norm(projection @ (r @ (query-pivot)+t-q)))
        rotation_variations.append(float(np.rad2deg(Rotation.from_matrix(r @ rotation.T).magnitude())))
    stability_bound = float(np.quantile(variations, .95))
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = pivot+translation-rotation @ pivot
    angular_observable = not null.size or np.linalg.norm(null[3:]) < 1e-6
    rotation_gain = np.sqrt(max(float(np.linalg.eigvalsh(information_inverse[3:, 3:])[-1]), 0.))
    rotation_bound = max(float(np.quantile(rotation_variations, .95)), float(np.rad2deg(noise*rotation_gain)))
    result.update(usable=dimension > 0, reason='local_background_fitted' if dimension else 'unconstrained_background',
        correction_after_to_before=transform.tolist(), observable_projector=projection.tolist(),
        observable_dimensions=dimension, parameter_rank=int(observable.sum()),
        angular_observable=bool(angular_observable), correction_rotation_deg=correction_angle,
        correction_at_pivot_m=float(np.linalg.norm(translation)),
        fit_residual_p95_m=fit_p95, check_residual_p95_m=check_p95,
        displacement_envelope_m=max(float(geometry_bound), stability_bound),
        rotation_envelope_deg=rotation_bound, spatial_model_stability_m=stability_bound,
        directions_complete=dimension == 3)
    return result


def assess_motion(region, model, config=None):
    cfg = {**CONFIG, **(config or {})}
    result = {'region_id': region['id'], 'status': 'uncertain', 'reason': 'local_background_unavailable', 'local_background': model}
    if region['type'] != 'moved':
        result.update(status='not_applicable', reason='presence_change_not_a_motion_test'); return result
    motion = region.get('motion') or {}
    if not motion.get('before_reference_m') or not motion.get('after_reference_m'):
        result['reason'] = 'missing_object_motion'; return result
    a, b = np.array(motion['before_reference_m']), np.array(motion['after_reference_m'])
    result['observed_displacement_m'] = float(np.linalg.norm(b-a))
    if not model.get('usable'):
        result['reason'] = model.get('reason', 'local_background_unavailable'); return result
    correction = np.array(model['correction_after_to_before']); projection = np.array(model['observable_projector'])
    corrected = correction[:3, :3] @ b+correction[:3, 3]-a
    expected = (np.linalg.inv(correction) @ np.r_[a, 1])[:3]-a
    observed_component = projection @ corrected
    error = model['displacement_envelope_m']+2*float(motion.get('surface_residual_m', 0.))
    corrected_rotation = correction[:3, :3] @ np.array(motion['before_to_after'])[:3, :3]
    rotation_deg = float(np.rad2deg(Rotation.from_matrix(corrected_rotation).magnitude()))
    result.update(local_map_drift_m=float(np.linalg.norm(projection @ expected)),
        local_map_drift_vector_m=(projection @ expected).tolist(),
        corrected_observable_displacement_m=float(np.linalg.norm(observed_component)),
        corrected_displacement_m=float(np.linalg.norm(corrected)) if model['directions_complete'] else None,
        unresolved_observed_component_m=float(np.linalg.norm((np.eye(3)-projection) @ (b-a))),
        comparison_envelope_m=float(error), motion_threshold_m=max(cfg['minimum_motion_m'], error),
        corrected_rotation_deg=rotation_deg, rotation_threshold_deg=max(cfg['minimum_rotation_deg'], model['rotation_envelope_deg']),
        reference_radius_m=model.get('radius_m'), reason='within_error_envelope')
    if region.get('identity_ambiguous'):
        result['reason'] = 'ambiguous_object_identity'; return result
    if motion.get('surface_overlap', 0.) < cfg['minimum_object_overlap'] or motion.get('surface_residual_m', 1.) > cfg['maximum_object_fit_residual_m']:
        result['reason'] = 'weak_object_surface_fit'; return result
    translated = result['corrected_observable_displacement_m'] > result['motion_threshold_m']
    rotated = model['angular_observable'] and rotation_deg > result['rotation_threshold_deg']
    if translated or rotated:
        result.update(status='motion_supported', reason='translation_exceeds_local_error' if translated else 'rotation_exceeds_local_error', evidence='translation' if translated else 'rotation')
    elif model['directions_complete'] and np.linalg.norm(corrected) <= error and rotation_deg <= model['rotation_envelope_deg']:
        result.update(status='alignment_explained', reason='difference_explained_by_local_alignment')
    else:
        result['reason'] = 'unconstrained_motion_direction' if not model['directions_complete'] else 'motion_near_error_limit'
    return result
