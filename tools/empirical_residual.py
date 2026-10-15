"""Conditional empirical residual references from explicit fit/check ROIs.

No alignment is estimated. The operator, normals and radii are identical in
fit/check/target areas. Reference residuals already mix reconstruction and
alignment effects; they replace the threshold, never the M3C2 sigma term.
"""
import numpy as np

NO_EVIDENCE=0
CANDIDATE=1
UNASSESSABLE=2


def in_bounds(points,bounds):
    return ((points>=bounds['min'])&(points<=bounds['max'])).all(axis=1)


def spatial_quantile(values,points,probability=.95,block_size=.16):
    """Individual distances with equal total weight per spatial block.

    Do not take block medians first: that narrows the fitting distribution
    relative to the individual target distances. No n^-1/2 uncertainty shrink.
    """
    values=np.asarray(values);points=np.asarray(points)
    keep=np.isfinite(values);values=values[keep];points=points[keep]
    if not len(values):return None,0
    _,membership,counts=np.unique(np.floor(points/block_size).astype(int),axis=0,return_inverse=True,return_counts=True)
    weights=1./counts[membership]
    order=np.argsort(values,kind='stable');cdf=np.cumsum(weights[order])
    index=min(np.searchsorted(cdf,probability*cdf[-1],side='left'),len(values)-1)
    return float(values[order[index]]),len(counts)


def exceeds(distance,threshold):
    """Strict physical comparison with only floating-point equality tolerance."""
    distance,threshold=np.broadcast_arrays(np.asarray(distance),np.asarray(threshold))
    tolerance=np.maximum(16*np.spacing(np.abs(threshold)),1e-12)
    return np.isfinite(distance)&np.isfinite(threshold)&(np.abs(distance)>threshold+tolerance)


def persistent_exceedance(distance,threshold,valid):
    over=valid&exceeds(distance,threshold)
    consistent=np.maximum(np.sum(over&(distance>0),axis=1),np.sum(over&(distance<0),axis=1))>=2
    return over,consistent


def validate_split(config):
    """Fit/check footprints must not share a measurement neighborhood."""
    fit=[r for r in config['rois'] if r['role']=='fit'];check=[r for r in config['rois'] if r['role']=='check']
    separation=2*max(config['radii_m'])
    for a in fit:
        for b in check:
            # These manual floor ROIs have nearly parallel vertical normals:
            # require separation in XY, not a misleading vertical gap.
            gap=np.maximum(0,np.maximum(np.array(a['bounds']['min'])[:2]-np.array(b['bounds']['max'])[:2],
                np.array(b['bounds']['min'])[:2]-np.array(a['bounds']['max'])[:2]))
            if np.linalg.norm(gap)<=separation:
                raise ValueError('Fit/check measurement neighborhoods overlap: '+a['id']+' / '+b['id'])


def feature_summary(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return {'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95))} if len(values) else None


def median_supported(values,axis):
    # Ignore truly unsupported rows without producing all-NaN warnings.
    values=np.asarray(values)
    if axis!=1:raise ValueError('Only the analysis-radius axis is supported')
    result=np.full((values.shape[0],)+values.shape[2:],np.nan)
    if values.ndim==3:
        for e in range(values.shape[2]):
            good=np.isfinite(values[:,:,e]).any(axis=1)
            result[good,e]=np.nanmedian(values[good,:,e],axis=1)
    return result


def fit_models(points,measurement,quality,normal_agreement,paired_observed,config,photo_review=None,support_disjoint=None):
    validate_split(config)
    distance=measurement['distance_m'];normals=measurement['normals']
    roughness=median_supported(measurement['spreads_m'],axis=1).max(axis=1)
    density=np.nanmedian(measurement['counts_by_epoch'].min(axis=-1),axis=1)
    axis=np.asarray(config['normal_axis'],float)
    normal_ok=np.abs(normals@axis)>=np.cos(np.deg2rad(config['maximum_normal_angle_deg']))
    shape_ok=normal_ok&(quality>=.35)&(normal_agreement>=np.cos(np.deg2rad(20)))
    valid=np.isfinite(distance)&(measurement['counts_by_epoch'].min(axis=-1)>=6)&paired_observed
    stable_support=shape_ok&(valid.sum(axis=1)>=2)
    models=[]
    for roi in [r for r in config['rois'] if r['role']=='fit']:
        inside=in_bounds(points,roi['bounds']);ids=np.flatnonzero(inside&stable_support)
        thresholds=[];block_counts=[]
        for j in range(distance.shape[1]):
            selected=ids[valid[ids,j]]
            t,blocks=spatial_quantile(np.abs(distance[selected,j]),points[selected],block_size=config['reference_block_m'])
            thresholds.append(t);block_counts.append(blocks)
        enough=(len(ids)>=config['minimum_reference_cores'] and all(x>=config['minimum_reference_blocks'] for x in block_counts))
        photos=photo_review is None or photo_review.get(roi['id'],False)
        disjoint=support_disjoint is None or support_disjoint.get(roi['id'],False)
        models.append({'id':roi['model'],'roi_id':roi['id'],'fit_core_ids':ids.tolist(),
            'thresholds_m':thresholds,'fit_core_count':len(ids),'reference_block_counts':block_counts,
            'raw_roi_core_count':int(inside.sum()),'geometry_observed_core_count':len(ids),
            'fit_ready':bool(enough and photos and disjoint),'photos_reviewable':bool(photos),'support_disjoint':bool(disjoint),'enabled':False,
            'fit_reason':'fit_ready' if enough and photos and disjoint else 'fit_check_surface_support_overlap' if not disjoint else 'paired_original_photos_unavailable' if not photos else 'insufficient_reference_support',
            'roughness_m':feature_summary(roughness[ids]),'density':feature_summary(density[ids]),
            'quality':feature_summary(quality[ids]),'center':np.median(points[ids],axis=0).tolist() if len(ids) else None,
            'reference_distribution':[{'radius_m':float(r),'absolute_residuals_m':np.abs(distance[ids[valid[ids,j]],j]).tolist(),
                'summary':feature_summary(np.abs(distance[ids[valid[ids,j]],j]))} for j,r in enumerate(config['radii_m'])]})
    return models,valid,roughness,density,shape_ok


def applicable(points,measurement,quality,normal_agreement,roughness,density,model,config):
    if not model['fit_ready']:return np.zeros(len(points),bool)
    n=measurement['normals'];axis=np.asarray(config['normal_axis'])
    shape=(np.abs(n@axis)>=np.cos(np.deg2rad(config['maximum_normal_angle_deg'])))
    shape&=(quality>=max(.35,model['quality']['p05']*.8))&(normal_agreement>=np.cos(np.deg2rad(20)))
    # Numerical roughness floor is used for ratios, never as a distance threshold.
    rough=roughness>=max(.001,model['roughness_m']['p05'])*config['roughness_ratio_range'][0]
    rough&=roughness<=max(.001,model['roughness_m']['p95'])*config['roughness_ratio_range'][1]
    dense=(density>=model['density']['p05']*config['density_ratio_range'][0])&(density<=model['density']['p95']*config['density_ratio_range'][1])
    center=np.array(model['center'])
    nearby=np.linalg.norm(points-center,axis=1)<=config['maximum_reference_distance_m']
    height=np.abs((points-center)@axis)<=config['maximum_reference_height_difference_m']
    return shape&rough&dense&nearby&height


def evaluate(points,measurement,quality,normal_agreement,paired_observed,config,photo_review=None,support_disjoint=None):
    models,valid,roughness,density,shape_ok=fit_models(points,measurement,quality,normal_agreement,paired_observed,config,photo_review,support_disjoint)
    distance=measurement['distance_m'];n=len(points);threshold=np.full(distance.shape,np.nan)
    model_index=np.full(n,-1,int);provisional=np.full(distance.shape,np.nan)
    reasons=np.full(n,'no_comparable_reference',dtype='U64')
    model_masks=[]
    for index,model in enumerate(models):
        domain=applicable(points,measurement,quality,normal_agreement,roughness,density,model,config)
        model_masks.append(domain)
        checks=[]
        for roi in [r for r in config['rois'] if r['role']=='check' and r['model']==model['id']]:
            inside=in_bounds(points,roi['bounds']);ids=np.flatnonzero(inside&domain&(valid.sum(1)>=2))
            t=np.array([v if v is not None else np.nan for v in model['thresholds_m']])
            _,over=persistent_exceedance(distance[ids],t,valid[ids])
            fraction=float(over.mean()) if len(ids) else None
            coverage=float(len(ids)/max(int(inside.sum()),1))
            enough=len(ids)>=config['minimum_holdout_valid_cores'] and coverage>=config['minimum_paired_observed_fraction']
            photo_ok=photo_review is None or photo_review.get(roi['id'],False)
            passed=bool(enough and photo_ok and fraction<=config['maximum_holdout_exceedance_fraction'])
            failures=[]
            if not photo_ok:failures.append('paired_original_photos_unavailable')
            if not enough:failures.append('insufficient_check_support')
            if len(ids)>=config['minimum_holdout_valid_cores'] and fraction>config['maximum_holdout_exceedance_fraction']:
                failures.append('reference_does_not_transfer_to_check')
            checks.append({'roi_id':roi['id'],'valid_core_ids':ids.tolist(),'valid_cores':len(ids),'total_cores':int(inside.sum()),
                'comparable_geometry_cores':int(np.sum(inside&domain)),
                'valid_fraction':coverage,'exceedance_fraction':fraction,'passed':passed,
                'reason':'check_passed' if passed else failures[-1] if failures else 'insufficient_check_support',
                'failure_reasons':failures,
                'absolute_residual_summary':[feature_summary(np.abs(distance[ids,j])) for j in range(distance.shape[1])]})
        model['checks']=checks;model['enabled']=bool(model['fit_ready'] and checks and all(c['passed'] for c in checks))
        model['check_reason']='reference_accepted_on_spatial_check' if model['enabled'] else model['fit_reason'] if not model['fit_ready'] else next((c['reason'] for c in checks if not c['passed']),'missing_spatial_check')
        t=np.array([v if v is not None else np.nan for v in model['thresholds_m']])
        fill=domain&(model_index<0)
        provisional[fill]=t
        reasons[fill]=model['check_reason']
        if model['enabled']:
            threshold[fill]=t;model_index[fill]=index
    eligible=(valid.sum(1)>=2)&(model_index>=0)
    reasons[valid.sum(1)<2]='insufficient_paired_surface_observation'
    over,persistent=persistent_exceedance(distance,threshold,valid)
    status=np.full(n,UNASSESSABLE,dtype=np.uint8)
    status[eligible]=NO_EVIDENCE;status[eligible&persistent]=CANDIDATE
    reasons[eligible&~persistent]='below_empirical_reference'
    reasons[eligible&persistent]='exceeds_comparable_reference'
    provisional_over,provisional_persistent=persistent_exceedance(distance,provisional,valid)
    return {'models':models,'valid':valid,'eligible':eligible,'threshold_m':threshold,
        'provisional_threshold_m':provisional,'exceeds':over,'persistent':persistent&eligible,
        'provisional_exceeds':provisional_over,'provisional_persistent':provisional_persistent,
        'status':status,'reason':reasons,'model_index':model_index,'roughness_m':roughness,
        'density':density,'shape_supported':shape_ok}


def region_decision(valid_fraction,exceed_fraction,largest_component_cores,exceed_area_m2,config):
    if valid_fraction<config['minimum_paired_observed_fraction']:
        return 'unassessable','insufficient_observation_or_comparable_reference'
    if exceed_fraction>=config['minimum_region_exceedance_fraction'] and largest_component_cores>=config['minimum_connected_exceedance_cores'] and exceed_area_m2>=config['minimum_connected_exceedance_area_m2']:
        return 'change_candidate','connected_reference_exceedance'
    return 'insufficient_change_evidence','insufficient_connected_exceedance'


def fixed_control(measurement,paired_observed):
    """Existing fixed-2cm surface rule on the same distances and observations."""
    d=measurement['distance_m'];s=measurement['spreads_m']
    threshold=1.96*np.sqrt(s[:,:,0]**2+s[:,:,1]**2+2*.02**2)
    valid=np.isfinite(d)&np.isfinite(threshold)&paired_observed
    over=valid&(np.abs(d)>=.06)&(np.abs(d)>threshold)
    persistent=np.maximum((over&(d>0)).sum(1),(over&(d<0)).sum(1))>=2
    eligible=valid.sum(1)>=2
    status=np.full(len(d),UNASSESSABLE,dtype=np.uint8);status[eligible]=NO_EVIDENCE;status[persistent]=CANDIDATE
    return {'threshold_m':threshold,'exceeds':over,'persistent':persistent,'eligible':eligible,
        'valid':valid,'status':status}
