"""Conservative multiscale screen for reconstructed-surface differences.

This is an application-specific screen after nominal M3C2, not M3C2-EP or a
calibrated change probability. Densified mesh samples are correlated: their
surface spread must not vanish as 1/sqrt(number of sampled mesh vertices).
No floor height, semantic/instance labels, or existing event boxes enter it.
"""
import numpy as np
from scipy.spatial import cKDTree

RADII = (.08, .16, .28)
REGISTRATION_SIGMA_M = .02  # Assumed uncertainty of each reconstruction.
MIN_DISPLACEMENT_M = .06
MIN_SCALES = 2
PATCH_FRACTION = .55


def _nearest_layer(values, minimum=6, gap=.08):
    """Avoid averaging separate layers such as a tabletop and its floor.

    Require six supported surface samples in a layer. An isolated return
    cannot replace the nearest supported layer.
    """
    values = np.sort(values)
    if len(values) < minimum:
        return None
    layers = np.split(values, np.flatnonzero(np.diff(values) > gap) + 1)
    layers = [part for part in layers if len(part) >= minimum]
    return min(layers, key=lambda part: abs(np.median(part))) if layers else None


def screen_surface_changes(queries, normals, surfaces, radii=RADII,
                           registration_sigma=REGISTRATION_SIGMA_M,
                           half_depth=.65, minimum_displacement=MIN_DISPLACEMENT_M):
    """Robust normal-direction distances and noise envelopes at three scales.

    Envelope = 1.96 * sqrt(spread_before² + spread_after² + 2*pose_sigma²).
    Spread uses the larger of MAD and the central-80% width estimate, without
    dividing by the number of mesh samples. The two pose errors are assumed
    independent here; actual camera covariance is unavailable.
    A point must exceed the envelope and the physical distance threshold at
    two scales in the same direction. Unsupported comparisons stay unknown.
    """
    queries = np.asarray(queries, dtype=float)
    normals = np.asarray(normals, dtype=float)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    shape = (len(queries), len(radii))
    centers, spreads, counts = [], [], []
    search_radius = np.hypot(max(radii), half_depth)
    for surface in surfaces:
        surface = np.asarray(surface, dtype=float)
        center = np.full(shape, np.nan)
        spread = np.full(shape, np.nan)
        count = np.zeros(shape, dtype=np.uint32)
        tree = cKDTree(surface)
        for start in range(0, len(queries), 256):
            neighborhoods = tree.query_ball_point(queries[start:start+256], search_radius, workers=4)
            for offset, ids in enumerate(neighborhoods):
                i = start + offset
                if len(ids) < 6:
                    continue
                delta = surface[ids] - queries[i]
                along = delta @ normals[i]
                tangent2 = np.maximum(0, np.einsum('ij,ij->i', delta, delta) - along**2)
                for j, radius in enumerate(radii):
                    layer = _nearest_layer(along[(tangent2 <= radius**2) & (np.abs(along) <= half_depth)])
                    if layer is None:
                        continue
                    median = np.median(layer)
                    mad = 1.4826 * np.median(np.abs(layer - median))
                    low, high = np.quantile(layer, [.1, .9])
                    center[i, j] = median
                    spread[i, j] = max(mad, (high-low)/2.563)
                    count[i, j] = len(layer)
        centers.append(center)
        spreads.append(spread)
        counts.append(count)
    distance = centers[1] - centers[0]
    envelope = 1.96 * np.sqrt(spreads[0]**2 + spreads[1]**2 + 2*registration_sigma**2)
    valid = np.isfinite(distance) & np.isfinite(envelope)
    passes = valid & (np.abs(distance) >= minimum_displacement) & (np.abs(distance) > envelope)
    positive = (passes & (distance > 0)).sum(axis=1)
    negative = (passes & (distance < 0)).sum(axis=1)
    persistent = np.maximum(positive, negative) >= MIN_SCALES
    ratio = np.divide(np.abs(distance), envelope, out=np.full(shape, np.nan), where=valid & (envelope > 0))
    return {'distance_m': distance, 'envelope_m': envelope, 'ratio': ratio,
            'valid': valid, 'persistent': persistent, 'counts': np.minimum(*counts),
            'radii_m': np.asarray(radii)}


def patch_noise_decision(screen, ids, own_observation=None):
    """Insufficient/weak geometry is excluded, rather than published as change.

    Geometry that survives but lacks visibility or unique motion evidence is
    still allowed to remain an uncertain region in the downstream classifier.
    """
    valid = screen['valid'][ids].sum(axis=1) >= MIN_SCALES
    persistent = screen['persistent'][ids]
    fraction = float(persistent.mean()) if len(ids) else 0.
    support = float(valid.mean()) if len(ids) else 0.
    reason = ('insufficient_multiscale_support' if support < PATCH_FRACTION else
              'within_reconstruction_noise' if fraction < PATCH_FRACTION else 'persistent_surface_difference')
    if reason == 'persistent_surface_difference' and own_observation is not None and own_observation['occupied'] < .40:
        reason = 'unsupported_source_surface'
    return {'keep': reason == 'persistent_surface_difference',
            'reason': reason, 'persistent_fraction': fraction, 'supported_fraction': support}
