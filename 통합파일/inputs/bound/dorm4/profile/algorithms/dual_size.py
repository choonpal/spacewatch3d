"""Experimental dual-size evidence; no template or reference labels as input.

Observed triangle area can stay relatively large in a short, folded patch.
We supplement its lower natural break with the longest PCA extent squared,
only where a SECOND area split within the first upper population independently
supports that component as small. A long-axis-only split may separate ordinary
single items from fused rows. Two area gates protect the tested configurations.

This is an engineering assumption, not proof that a surface is noise.
"""
import numpy as np
from statistical_modes import Settings, lower_modes


def pca_extents(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected a nonempty finite N x 3 point array')
    p = p - p.mean(axis=0)
    _, axes = np.linalg.eigh(p.T @ p)
    return np.sort(np.ptp(p @ axes, axis=0))


def lower_components(areas, extents, cfg=Settings()):
    a = np.asarray(areas, dtype=float)
    s = np.asarray(extents, dtype=float)
    if s.shape != (len(a), 3) or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError('Expected one finite nonnegative PCA extent triple per area')
    if len(a) and (np.any(s[:, 2] <= 0) or np.any(np.diff(s, axis=1) < -1e-12)):
        raise ValueError('Expected ascending extents with positive largest extent')
    area_drop, area_report = lower_modes(a, cfg)
    span_drop, span_report = lower_modes(s[:, 2] ** 2, cfg)
    area_gate = area_report['status'] == 'separated_lower_natural_break'
    # A long-axis break may separate ordinary single items from fused rows.
    # Require a second separated small-area mode AMONG the first area's upper
    # population before using additional extent exclusions. This is one
    # additional split only, not recursion until a desired object count.
    secondary = np.zeros(len(a), bool)
    secondary_report = None
    if area_gate:
        secondary_flags, secondary_report = lower_modes(a[~area_drop], cfg)
        secondary[~area_drop] = secondary_flags
    extra = span_drop & secondary
    drop = area_drop | extra
    return drop, dict(
        method='area lower break plus secondary-area/longest-extent lower-break intersection',
        reference_labels_used_as_input=False, upper_size_exclusions=False,
        area_gate=area_gate, area=area_report, longest_extent_squared=span_report,
        secondary_area=secondary_report, secondary_flags=secondary.tolist(),
        area_excluded=int(area_drop.sum()), additional_extent_excluded=int(extra.sum()),
        excluded=int(drop.sum()), area_flags=area_drop.tolist(),
        extent_flags=span_drop.tolist(), additional_flags=extra.tolist(),
        limitations=['A real small object may be indistinguishable from a fragment.',
                    'No observed complete-sized population means no complete-size anchor.',
                    'Fused-vs-single protection is tested for specified cases, not guaranteed universally.'])
