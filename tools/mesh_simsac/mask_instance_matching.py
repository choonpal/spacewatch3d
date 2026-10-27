"""Existing mask-assisted rigid matching gates; no room IDs or event overrides."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .core import transform

PARAMETERS={'voxel_h':.0064,'point_radius_h':.35,'vertical_radius_h':.45,
    'minimum_partition_faces':12,'minimum_seed_separation_h':.12,
    'shape_tolerance_h':.045,'minimum_shape_f1':.65,'minimum_directional_coverage':.50,
    'minimum_shape_gain_over_stationary_surface':.10,'minimum_assignment_margin':.05,
    'minimum_destination_vacancy_views':2,
    'movement_error_multiplier':3.,'minimum_movement_h':.15,
    'yaw_seeds_degrees':list(range(0,360,45)),'icp_iterations':28,'icp_trim_fraction':.85}

def rigid_shape_fit(reference,source,height):
    """Return an upright rigid reference->source fit with symmetric coverage."""
    a=np.asarray(reference);b=np.asarray(source);tolerance=height*PARAMETERS['shape_tolerance_h']
    if min(len(a),len(b))<20:return None
    tree=cKDTree(b);best=None
    for degrees in PARAMETERS['yaw_seeds_degrees']:
        angle=np.deg2rad(degrees);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=b.mean(0)-rotation@a.mean(0)
        for _ in range(PARAMETERS['icp_iterations']):
            moved=transform(a,matrix);distance,index=tree.query(moved)
            keep=distance<=np.quantile(distance,PARAMETERS['icp_trim_fraction'])
            x=moved[keep];y=b[index[keep]];cx=x.mean(0);cy=y.mean(0)
            xx=(x-cx)[:,[0,2]];yy=(y-cy)[:,[0,2]]
            angle=np.arctan2(np.sum(xx[:,1]*yy[:,0]-xx[:,0]*yy[:,1]),np.sum(xx*yy))
            rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=cy-rotation@cx;matrix=delta@matrix
            if np.linalg.norm(delta[:3,3])<1e-6 and abs(angle)<1e-6:break
        moved=transform(a,matrix);distance=tree.query(moved)[0];opposite=cKDTree(moved).query(b)[0]
        coverage=[float(np.mean(d<=tolerance)) for d in [distance,opposite]]
        f1=2*coverage[0]*coverage[1]/max(sum(coverage),1e-12)
        inliers=np.r_[distance[distance<=tolerance],opposite[opposite<=tolerance]]
        result={'shape_f1':float(f1),'directional_coverage':coverage,
            'shape_rmse_u':float(np.sqrt(np.mean(inliers**2))) if len(inliers) else None,
            'reference_to_source_rigid':matrix.tolist(),'object_scale_fitted':False}
        if best is None or (result['shape_f1'],-(result['shape_rmse_u'] or 1e3))>(best['shape_f1'],-(best['shape_rmse_u'] or 1e3)):best=result
    return best

def assignment_gate(pair,height,error):
    """Reject a fit to a different chair that still occupies the old location."""
    reasons=[]
    if pair['shape_f1']<PARAMETERS['minimum_shape_f1']:reasons.append('shape_f1')
    if min(pair['directional_coverage'])<PARAMETERS['minimum_directional_coverage']:reasons.append('bidirectional_coverage')
    if pair['displacement_u']<=max(PARAMETERS['movement_error_multiplier']*error,PARAMETERS['minimum_movement_h']*height):reasons.append('within_registration_error')
    if pair['origin_shape_gain']<PARAMETERS['minimum_shape_gain_over_stationary_surface']:reasons.append('origin_surface_still_present')
    if pair['destination_shape_gain']<PARAMETERS['minimum_shape_gain_over_stationary_surface']:reasons.append('destination_surface_already_present')
    if pair['destination_vacancy_views']<PARAMETERS['minimum_destination_vacancy_views']:reasons.append('destination_vacancy_unsupported')
    if pair['same_position_partner']:reasons.append('already_stationary_matched')
    return reasons

def select_assignments(pairs):
    """One-to-one assignment with null matches and a competing-fit margin."""
    eligible=[p for p in pairs if not p['rejection_reasons']]
    if not eligible:return []
    for p in eligible:
        competitors=[q['shape_f1'] for q in eligible if q is not p and (q['reference']==p['reference'] or q['source']==p['source'])]
        p['assignment_margin']=float(p['shape_f1']-max(competitors,default=0.))
        if p['assignment_margin']<PARAMETERS['minimum_assignment_margin']:p['rejection_reasons'].append('ambiguous_assignment')
    eligible=[p for p in eligible if not p['rejection_reasons']]
    if not eligible:return []
    aa=sorted({p['reference'] for p in eligible});bb=sorted({p['source'] for p in eligible})
    score=np.zeros((len(aa)+len(bb),len(bb)+len(aa)));score[:len(aa),:len(bb)]=-1e3
    by_pair={(p['reference'],p['source']):p for p in eligible}
    for p in eligible:score[aa.index(p['reference']),bb.index(p['source'])]=p['shape_f1']
    accepted=[]
    for i,j in zip(*linear_sum_assignment(-score)):
        if i>=len(aa) or j>=len(bb) or score[i,j]<=0:continue
        p=by_pair[(aa[i],bb[j])]
        accepted.append(p)
    return accepted
