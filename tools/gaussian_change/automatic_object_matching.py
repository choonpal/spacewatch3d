"""Automatic mask-assisted component partition and fixed-frame object matching.

No region-specific labels, manual IDs or object scale fitting. The camera/map
registration stays frozen. Uncertain assignments remain unclassified.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from prepare import transform,epoch_to_common

PARAMETERS={'voxel_h':.0064,'point_radius_h':.35,'vertical_radius_h':.45,
    'minimum_partition_faces':12,'minimum_seed_separation_h':.12,
    'shape_tolerance_h':.045,'minimum_shape_f1':.65,'minimum_directional_coverage':.50,
    'minimum_shape_gain_over_stationary_surface':.10,'minimum_assignment_margin':.05,
    'minimum_destination_vacancy_views':2,
    'movement_error_multiplier':3.,'minimum_movement_h':.15,
    'yaw_seeds_degrees':list(range(0,360,45)),'icp_iterations':28,'icp_trim_fraction':.85}


def track_points(track,ctx,height):
    raw=[o['raw_points'] for o in track['observations'] if o.get('raw_points')]
    if not raw:return np.empty((0,3))
    points=transform(np.concatenate(raw),epoch_to_common(ctx,track['epoch'],track['zone']))
    delta=points-np.asarray(track['center'])
    points=points[(np.linalg.norm(delta[:,[0,2]],axis=1)<PARAMETERS['point_radius_h']*height)
                  &(abs(delta[:,1])<PARAMETERS['vertical_radius_h']*height)]
    if not len(points):return points
    # Retain actual observed coordinates, with one representative per voxel.
    _,indices=np.unique(np.floor(points/(PARAMETERS['voxel_h']*height)),axis=0,return_index=True)
    return points[np.sort(indices)]


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


def infer(all_objects,prepared,tracks,stationary_pairs,meshes,ctx,height,zones):
    """Return derived mesh subsets and automatically accepted motion matches."""
    supported=[t for t in tracks if t['supported'] and t['zone'] in zones and t['class_name'] in ('chair','table','cabinet','dining table')]
    points={t['id']:track_points(t,ctx,height) for t in supported}
    supported=[t for t in supported if len(points[t['id']])>=20]
    mapping={};attached={}
    for t in supported:
        family='table' if t['class_name']=='dining table' else t['class_name']
        candidates=[o for o in all_objects.values() if o['epoch']==t['epoch'] and o['zone']==t['zone'] and o['class_name']==family]
        distances=[(float(cKDTree(prepared[o['id']]['vertices']).query(t['center'])[0]),o['id']) for o in candidates]
        if not distances:continue
        distance,parent=min(distances)
        limit=ctx['methods']['structure']['errors']['regions'][t['zone']]['tolerance_u']+t['spread_u']
        if distance>limit:continue
        attached.setdefault(parent,[]).append(t)
    derived={};geometry={}
    for parent,own in attached.items():
        original=all_objects[parent];geo=prepared[parent]
        if len(own)==1:mapping[own[0]['id']]=parent;continue
        centers=np.array([t['center'] for t in own]);separation=np.linalg.norm(centers[:,None,[0,2]]-centers[None,:,[0,2]],axis=-1)
        np.fill_diagonal(separation,np.inf)
        if np.min(separation)<PARAMETERS['minimum_seed_separation_h']*height:
            # Duplicate tracks cannot establish distinct mesh instances.
            continue
        centroids=geo['centroids'];cost=[]
        for t in own:
            distances=cKDTree(points[t['id']]).query(centroids,k=3)[0].mean(1)
            cost.append(distances+.1*np.linalg.norm(centroids-np.asarray(t['center']),axis=1))
        assigned=np.argmin(cost,axis=0)
        for index,t in enumerate(own):
            subset=np.flatnonzero(assigned==index)
            if len(subset)<PARAMETERS['minimum_partition_faces']:continue
            used,inverse=np.unique(geo['faces'][subset],return_inverse=True)
            vertices=geo['vertices'][used];faces=inverse.reshape(-1,3);id=parent+'@'+t['id']
            original_faces=np.asarray(geo['original_face_ids'])[subset].tolist()
            original_vertices=np.asarray(geo['original_vertex_ids'])[used].tolist()
            triangles=vertices[faces];areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
            observed=meshes[t['epoch']]['face_observations'][original_faces]>=2
            geometry[id]={'vertices':vertices,'faces':faces,'centroids':vertices[faces].mean(1),
                          'original_face_ids':original_faces,'original_vertex_ids':original_vertices}
            derived[id]={**original,'id':id,'parent_instance':parent,'image_track':t['id'],
                'partition_method':'automatic_multiview_mask_points_nearest_face','automatic_partition':True,
                'physical_instance_id':False,'center':np.median(vertices,axis=0).tolist(),
                'bounds':[vertices.min(0).tolist(),vertices.max(0).tolist()],'faces':len(faces),
                'multi_observation_fraction':float(areas[observed].sum()/max(areas.sum(),1e-12)),
                'tracks':[o for o in original['tracks'] if o['id']==t['id']],
                'area_u2':float(areas.sum()),
                'mesh_file':'object-meshes/'+id+'.json'}
            mapping[t['id']]=id
    registry={**all_objects,**derived};trees={};stationary={}
    for pair in stationary_pairs:
        stationary[pair['reference']]=pair['source'];stationary[pair['source']]=pair['reference']
    audit=[]
    for a in [t for t in supported if t['epoch']==0 and t['id'] in mapping]:
        key=(a['zone'],1)
        if key not in trees:trees[key]=cKDTree(transform(meshes[1]['p'],epoch_to_common(ctx,1,a['zone'])))
        stationary_fraction=float(np.mean(trees[key].query(points[a['id']])[0]<=height*PARAMETERS['shape_tolerance_h']))
        for b in [t for t in supported if t['epoch']==1 and t['id'] in mapping and t['zone']==a['zone'] and t['class_name']==a['class_name']]:
            # Static assignments are protected; keep their exclusion in the audit.
            if a['id'] in stationary or b['id'] in stationary:
                audit.append({'reference':mapping[a['id']],'source':mapping[b['id']],
                    'reference_track':a['id'],'source_track':b['id'],'reference_zone':a['zone'],'source_zone':b['zone'],
                    'rejection_reasons':['already_stationary_matched'],'same_position_partner':stationary.get(a['id']) or stationary.get(b['id'])})
                continue
            fit=rigid_shape_fit(points[a['id']],points[b['id']],height)
            if fit is None:continue
            if (b['zone'],0) not in trees:trees[(b['zone'],0)]=cKDTree(meshes[0]['p'])
            destination_fraction=float(np.mean(trees[(b['zone'],0)].query(points[b['id']])[0]<=height*PARAMETERS['shape_tolerance_h']))
            pair={**fit,'reference':mapping[a['id']],'source':mapping[b['id']],
                'reference_track':a['id'],'source_track':b['id'],'reference_zone':a['zone'],'source_zone':b['zone'],
                'displacement_u':float(np.linalg.norm(np.asarray(b['center'])-a['center'])),
                'origin_stationary_surface_fraction':stationary_fraction,
                'origin_shape_gain':float(fit['directional_coverage'][0]-stationary_fraction),
                'destination_stationary_surface_fraction':destination_fraction,
                'destination_shape_gain':float(fit['directional_coverage'][1]-destination_fraction),
                'destination_vacancy_views':next((t['vacancy_views'] for t in registry[mapping[b['id']]]['tracks'] if t['id']==b['id']),0),
                'same_position_partner':None,'already_paired_with_other':None,
                'origin_free_views':next((t['vacancy_views'] for t in registry[mapping[a['id']]]['tracks'] if t['id']==a['id']),0),
                'origin_vacancy_confirmed':False,'physical_id_confirmed':False,
                'method':'automatic_instance_track_rigid_assignment'}
            error=ctx['methods']['structure']['errors']['regions'][a['zone']]['tolerance_u']
            pair['rejection_reasons']=assignment_gate(pair,height,error);audit.append(pair)
    accepted=select_assignments([p for p in audit if 'shape_f1' in p])
    return derived,geometry,{'automatic':True,'manual_overrides_used':False,'parameters':PARAMETERS,
        'track_to_mesh':mapping,'derived_instances':len(derived),'comparisons':audit,'accepted_matches':accepted}
