"""Conservative model comparison on top of the frozen 18-view target rule.

This is an engineering safeguard, not a learned noise probability. In a
connected row of similar objects, area and length both grow roughly linearly,
while the middle extent stays similar. Uniformly smaller fragments instead
shrink sqrt(area), middle extent and length together. Both geometric models
have one nonnegative growth parameter. Prefer KEEP when the row model fits at
least as well. No class name, ID, coordinate, physical size or reference mask
is an input. The existing ratio10 proposal rule is NOT relaxed by this guard.
"""
from pathlib import Path
import sys

import numpy as np

FROZEN = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
from middle_size import lower_components as proposed_components
from statistical_modes import Settings


def model_error(delta, direction):
    """Equal-weight log-linear geometric residual; no fitted physical units."""
    d = np.asarray(delta, dtype=float)
    direction = np.asarray(direction, dtype=float)
    growth = max(0.0, float(d @ direction / (direction @ direction)))
    return float(np.sum((d - growth * direction) ** 2)), growth


def tail_gap(areas, cfg=Settings()):
    """Count-insensitive fallback: a genuinely empty adjacent log-area gap.

    Only proposed when the ordinary lower-mode gate found no separated group.
    The same shared ratio10 is used on the two adjacent observations (a stronger
    condition than separated group MEANS). The later fusion guard still applies.
    """
    a = np.asarray(areas, float)
    flags = np.zeros(len(a), bool)
    info = dict(status='no_separated_adjacent_gap', applied=False)
    if len(a) < cfg.minimum_class_samples:
        return flags, info
    order = np.argsort(a, kind='stable')
    gaps = np.diff(np.log(a[order]))
    i = int(np.argmax(gaps))
    ratio = float(np.exp(gaps[i]))
    info.update(adjacent_area_ratio=ratio,
                minimum_adjacent_area_ratio=cfg.minimum_geometric_mean_ratio)
    if gaps[i] > 1e-12 and ratio >= cfg.minimum_geometric_mean_ratio:
        flags[order[:i+1]] = True
        info.update(status='separated_adjacent_gap', applied=True,
                    boundary_area=float(np.sqrt(a[order[i]] * a[order[i+1]])))
    return flags, info


def lower_components(areas, extents, cfg=Settings(), *, support_tail_gap=False):
    a, s = np.asarray(areas, float), np.asarray(extents, float)
    proposed, previous = proposed_components(a, s, cfg)
    original_proposals = proposed.copy()
    if not len(a):
        return proposed, dict(previous, fusion_model_checks=[], protected_indices=[])
    # middle_size already validates inputs and strictly positive middle/long axes.
    x = np.column_stack((0.5 * np.log(a), np.log(s[:, 1]), np.log(s[:, 2])))
    area_lower = np.asarray(previous['area_flags'], bool)
    secondary_lower = np.asarray(previous['secondary_flags'], bool)
    gap_report = dict(status='not_requested', applied=False)
    gap_flags = np.zeros(len(a), bool)
    if support_tail_gap and not previous['area_gate']:
        gap_flags, gap_report = tail_gap(a, cfg)
        proposed = proposed | gap_flags
        if gap_flags.any():
            area_lower = gap_flags
    protect = np.zeros(len(a), bool)
    checks = []
    for i in np.flatnonzero(proposed):
        # Compare each proposal with the larger population that generated its
        # corresponding area evidence, not with itself or another lower group.
        population = ~(area_lower | secondary_lower) if secondary_lower[i] else ~area_lower
        population = population & (a > a[i])
        if not population.any():
            protect[i] = True
            checks.append(dict(index=int(i), decision='keep_no_larger_anchor', anchors=[]))
            continue
        delta = x[population].mean(axis=0) - x[i]
        isotropic_error, isotropic_growth = model_error(delta, [1., 1., 1.])
        row_error, row_growth = model_error(delta, [.5, 0., 1.])
        # Numerical tie tolerance only; not a size/noise threshold.
        tolerance = 32 * np.finfo(float).eps * max(1., isotropic_error, row_error)
        protect[i] = row_error <= isotropic_error + tolerance
        checks.append(dict(index=int(i), anchors=np.flatnonzero(population).tolist(),
                           log_feature_delta=delta.tolist(),
                           isotropic_error=isotropic_error, row_error=row_error,
                           isotropic_growth=isotropic_growth, row_growth=row_growth,
                           decision='keep_fusion_alternative' if protect[i] else 'retain_exclusion_proposal'))
    result = proposed & ~protect
    return result, dict(previous,
                        method='frozen18 proposals with equal-parameter isotropic-vs-row geometric KEEP safeguard',
                        upstream_proposed_flags=original_proposals.tolist(),
                        fusion_model_checks=checks,
                        tail_gap=gap_report, tail_gap_proposed_flags=gap_flags.tolist(),
                        protected_indices=np.flatnonzero(protect).tolist(),
                        excluded=int(result.sum()),
                        ratio_relaxed=False,
                        limitations=previous.get('limitations', []) + [
                            'Row area proportional to length is an approximation, not true for every fused mesh.',
                            'Equal weights in log-linear geometry and these two models are engineering assumptions.',
                            'A real small item and a similar-size fragment can still be indistinguishable.',
                            'Protection is conservative model comparison, not calibrated noise evidence.'])
