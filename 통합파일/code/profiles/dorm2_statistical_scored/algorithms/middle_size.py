"""Scale-free cross-section supplement, local screen only until validated.

The longest extent can grow with a fused row. The middle PCA extent instead
measures its cross-section. A separated lower middle-extent group can add
evidence, but only when the original area's split is itself separated.
No semantic class, physical size, coordinate, ID, or reference mask is input.
"""
import numpy as np
from statistical_modes import Settings,lower_modes
from dual_size import lower_components as guarded


def lower_components(areas,extents,cfg=Settings()):
    base,report=guarded(areas,extents,cfg)
    s=np.asarray(extents,float)
    if len(s) and np.any(s[:,1]<=0):
        raise ValueError('Positive-area component needs positive middle extent')
    cross,details=lower_modes(s[:,1]**2,cfg)
    additional=cross & ~base if report['area_gate'] else np.zeros(len(base),bool)
    combined=np.asarray(report['additional_flags'],bool)|additional
    return base|additional,dict(report,
        method='protected two-area/long-extent evidence plus area-gated middle-extent evidence',
        middle_extent_squared=details,middle_additional_flags=additional.tolist(),
        middle_additional_excluded=int(additional.sum()),
        additional_long_flags=report['additional_flags'],additional_flags=combined.tolist(),
        additional_extent_excluded=int(combined.sum()),excluded=int((base|additional).sum()))
