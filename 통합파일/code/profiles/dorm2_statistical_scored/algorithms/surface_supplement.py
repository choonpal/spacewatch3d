"""Small-size + missing geometric border supplement; development only.

The lower/upper size populations are computed before this feature is examined.
Do not lower the existing ratio10 rule: this is a separate joint hypothesis.
Small relative to the upper population alone is not enough. Require empirical
boundary-angle Q75 < internal NONZERO-bend Q25, i.e. a clear reversed border
contrast, and reject the proposal if a same-cross-section row fits better.
Quartiles and the two growth models are common engineering assumptions, not
calibrated noise probabilities. No IDs, class-specific dimensions or labels.
"""
from pathlib import Path
import sys

import numpy as np

FUSION = Path(__file__).resolve().parent
sys.path.insert(0, str(FUSION))
from tail_guard import lower_components as previous
from fusion_guard import model_error
from statistical_modes import Settings, lower_modes


def lower_components(areas, extents, features, cfg=Settings()):
    a, s = np.asarray(areas, float), np.asarray(extents, float)
    base, old = previous(a, s, cfg)
    if len(features) != len(a):
        raise ValueError('One geometry feature record per component required')
    if len(a) < 2:
        return base, dict(old, new_geometry_indices=[], geometry_checks=[])
    # The upper population's geometric mean is only a candidate gate, NOT an
    # absolute normal-size estimate and never a sufficient exclusion criterion.
    _, area_model = lower_modes(a, cfg)
    center = area_model.get('upper_geometric_mean')
    if center is None:
        return base, dict(old, new_geometry_indices=[], geometry_checks=[])
    x = np.column_stack((.5*np.log(a), np.log(s[:, 1]), np.log(s[:, 2])))
    add = np.zeros(len(a), bool)
    checks = []
    for i, f in enumerate(features):
        b, interior = f.get('boundary_angle_quantiles'), f.get('internal_bend_angle_quantiles')
        reasons = []
        if f.get('uncertain_topology_faces', 0): reasons.append('nonmanifold_or_open_topology_unknown')
        if not a[i] < center: reasons.append('not_smaller_than_upper_population_center')
        if b is None or interior is None: reasons.append('missing_boundary_or_internal_bend_evidence')
        elif not b[3] < interior[1]: reasons.append('no_reversed_boundary_quartile_gap')
        larger = a > a[i]
        row_error = scale_error = None
        if not larger.any(): reasons.append('no_larger_anchor')
        elif not reasons:
            delta = x[larger].mean(axis=0) - x[i]
            row_error = model_error(delta, [.5, 0., 1.])[0]
            scale_error = model_error(delta, [1., 1., 1.])[0]
            tolerance = 32*np.finfo(float).eps*max(1., row_error, scale_error)
            if row_error <= scale_error + tolerance: reasons.append('fusion_alternative_keep')
        add[i] = not reasons and not base[i]
        checks.append(dict(index=i, boundary_q75=None if b is None else b[3],
                           internal_bend_q25=None if interior is None else interior[1],
                           row_error=row_error, isotropic_error=scale_error,
                           reasons_to_keep=reasons, newly_excluded=bool(add[i])))
    return base | add, dict(old, geometry_upper_population_center=center,
                            new_geometry_indices=np.flatnonzero(add).tolist(),
                            geometry_checks=checks, excluded=int((base|add).sum()))
