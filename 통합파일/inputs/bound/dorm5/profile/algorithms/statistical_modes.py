"""Exact lower natural break of log mesh areas; no physical size templates.

This is a two-class within-variance partition (Otsu/Fisher type), extended to
mesh-component sizes. A mesh-specific, dimensionless separation guard is an
engineering assumption, not a calibrated probability of physical noise.
"""
from dataclasses import asdict,dataclass
import numpy as np

@dataclass(frozen=True)
class Settings:
    minimum_class_samples:int=2
    minimum_geometric_mean_ratio:float=10.0

def lower_modes(areas,cfg=Settings()):
    a=np.asarray(areas,float)
    if a.ndim!=1 or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Expected finite strictly positive surface areas')
    remove=np.zeros(len(a),bool)
    report=dict(status='keep_insufficient_or_no_separated_break',n=len(a),settings=asdict(cfg),
                method='exact two-group within-variance partition of log area',
                reference_masks_used_as_input=False,upper_size_exclusions=False)
    if len(a)<cfg.minimum_class_samples:return remove,report
    order=np.argsort(a,kind='stable');x=np.log(a[order]);x=x-x.mean()
    if np.ptp(x)<=1e-12:return remove,report
    counts=np.arange(1,len(a));summed=np.cumsum(x)
    low=summed[:-1]/counts;high=(summed[-1]-summed[:-1])/(len(a)-counts)
    between=counts*(len(a)-counts)/len(a)**2*(high-low)**2
    between[np.diff(x)<=1e-12]=-np.inf
    i=int(np.argmax(between));boundary=i+1
    ratio=float(np.exp(high[i]-low[i]))
    report.update(between_group_variance=float(between[i]),geometric_mean_ratio=ratio,
                  boundary_area=float(np.sqrt(a[order[i]]*a[order[i+1]])),
                  lower_geometric_mean=float(np.exp(np.log(a[order[:boundary]]).mean())),
                  upper_geometric_mean=float(np.exp(np.log(a[order[boundary:]]).mean())))
    if ratio<cfg.minimum_geometric_mean_ratio:return remove,report
    remove[order[:boundary]]=True
    report.update(status='separated_lower_natural_break',excluded=int(remove.sum()),
                  excluded_area_share=float(a[remove].sum()/a.sum()))
    return remove,report
