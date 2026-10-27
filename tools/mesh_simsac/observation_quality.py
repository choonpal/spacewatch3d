"""Check individual mesh-object footprint and viewpoint before RGB voting."""
import numpy as np

from .core import project_in_view

DEFAULTS=dict(min_pixels=64,max_scale=1.5,max_angle=35.,min_inside=.6)


def quality_settings(settings):
    result={k:settings.get('object_pair_'+k,v) for k,v in DEFAULTS.items()}
    if not isinstance(result['min_pixels'],int) or result['min_pixels']<1:
        raise ValueError('object_pair_min_pixels must be a positive integer')
    if not np.isfinite(result['max_scale']) or result['max_scale']<1:
        raise ValueError('object_pair_max_scale must be finite and at least 1')
    if not np.isfinite(result['max_angle']) or not 0<=result['max_angle']<=180:
        raise ValueError('object_pair_max_angle must be between 0 and 180 degrees')
    if not np.isfinite(result['min_inside']) or not 0<result['min_inside']<=1:
        raise ValueError('object_pair_min_inside must be in (0,1]')
    return result


def reference_consensus(support, positive, ratio=.5):
    """A tie between valid references is disagreement, not a positive vote."""
    required=np.maximum(np.ceil(support*ratio),np.floor(support/2)+1)
    return (support>0)&(positive>=required)


def object_pair_quality(query, reference, min_pixels=64, max_scale=1.5, max_angle=35., min_inside=.6):
    """Evaluate the same source surface in both calibrated cameras.

    No opposite semantic label is required: an empty expected object location
    must remain comparable for addition/removal. No RGB crop is rescaled to
    manufacture a size match. Rejected comparisons provide no negative vote.
    """
    eligible=np.ones(query['valid'].shape,bool)
    audits=[]
    source=query['valid']&~query['person']&query.get('segmentation_supported',True)
    for iid in np.unique(query['instance'][source]):
        if iid<=0:continue
        mask=source&(query['instance']==iid)
        y,x=np.nonzero(mask)
        reasons=[];scale=None;angle=None;fraction=0.
        if len(x)<min_pixels:
            reasons.append('object_too_small')
        else:
            points=query['points'][mask]
            regions=query['point_region'][mask] if 'point_region' in query else None
            uv,z,inside=project_in_view(points,reference,regions)
            fraction=float(inside.mean())
            if fraction<min_inside:reasons.append('object_cropped_in_reference')
            if inside.any():
                # Quantiles prevent one coarse edge ray from dominating scale.
                low,high=np.quantile(uv[inside],[.02,.98],axis=0)
                qlow,qhigh=np.quantile(np.column_stack([x,y]),[.02,.98],axis=0)
                qarea=np.prod(np.maximum(1,qhigh-qlow+1))
                rarea=np.prod(np.maximum(1,high-low+1))
                scale=float(np.sqrt(rarea/qarea))
                if not 1/max_scale<=scale<=max_scale:reasons.append('object_scale_mismatch')
                positive_regions=regions[regions>0] if regions is not None else []
                region=int(np.bincount(positive_regions.astype(int)).argmax()) if len(positive_regions) else None
                qpose=query['region_poses'][region] if region is not None else query['pose']
                rpose=reference['region_poses'][region] if region is not None else reference['pose']
                center=np.median(points,axis=0)
                a,b=center-qpose[:3,3],center-rpose[:3,3]
                cosine=float(a@b/max(1e-8,np.linalg.norm(a)*np.linalg.norm(b)))
                angle=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
                if angle>max_angle:reasons.append('object_viewpoint_mismatch')
        if reasons:eligible[query['instance']==iid]=False
        audits.append(dict(instance_id=int(iid),query_pixels=len(x),reference_scale_ratio=scale,
            viewpoint_degrees=angle,reference_inside_fraction=fraction,accepted=not reasons,reasons=reasons))
    return eligible,audits
