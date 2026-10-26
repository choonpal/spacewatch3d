"""Direction-independent surface changes with an empirical floor noise budget.

The floor provides a scalar noise reference, not a 3D covariance or a rule
that furniture must be parallel to it. All local surface directions are used.
Annotation identities are accepted only by the separate evaluation function.
"""
import numpy as np
from scipy.spatial import cKDTree

RADII=(.08,.16,.28)


def clearance_measurements(queries,normals,owner,surfaces,surface_normals,opposite_clearance,radii=RADII):
    """Robust multiscale distance from each source surface to the other mesh.

    Clearance is computed against actual triangles by the caller, in both
    directions. Neighbor normals are compared to the query's own normal,
    never to the floor. Fine-scale plane residuals describe local roughness.
    """
    queries=np.asarray(queries,float);normals=np.asarray(normals,float)
    shape=(len(queries),len(radii));distance=np.full(shape,np.nan);lower=distance.copy()
    count=np.zeros(shape,np.uint32);coherence=np.zeros(shape);roughness=np.full(len(queries),np.nan)
    for e in (0,1):
        selected=np.flatnonzero(owner==e)
        tree=cKDTree(surfaces[e])
        for start in range(0,len(selected),256):
            ids=selected[start:start+256]
            neighbors=tree.query_ball_point(queries[ids],max(radii),workers=4)
            for i,group in zip(ids,neighbors):
                group=np.asarray(group,int)
                aligned=np.abs(surface_normals[e][group]@normals[i])>=np.cos(np.deg2rad(50))
                group=group[aligned]
                if len(group)<6:continue
                delta=surfaces[e][group]-queries[i];norm=np.linalg.norm(delta,axis=1)
                fine=delta[norm<=.10]
                if len(fine)>=6:
                    centered=fine-fine.mean(0)
                    _,vectors=np.linalg.eigh(centered.T@centered)
                    residual=centered@vectors[:,0];median=np.median(residual)
                    lo,hi=np.quantile(residual,[.1,.9])
                    roughness[i]=max(1.4826*np.median(np.abs(residual-median)),(hi-lo)/2.563)
                else:continue
                for j,radius in enumerate(radii):
                    values=opposite_clearance[e][group[norm<=radius]]
                    values=values[np.isfinite(values)]
                    count[i,j]=len(values)
                    if len(values)<6:continue
                    distance[i,j]=np.median(values);lower[i,j]=np.quantile(values,.25)
                    coherence[i,j]=np.mean(values>=.06)
    return {'distance_m':distance,'lower_quartile_m':lower,'counts':count,
        'roughness_m':roughness,'coherence':coherence,'normals':normals,'radii_m':np.array(radii)}


def classify_clearance(measurement,floor_budget,background,source_distance):
    """Apply the measured noise budget to all directions; no paired-layer gate."""
    d=measurement['distance_m'];rough=measurement['roughness_m']
    reference=np.broadcast_to(np.asarray(floor_budget,float),d.shape)
    # The maximum combines two noise screens without adding a second copy of
    # the floor residual variance. Mesh samples never reduce noise by sqrt(n).
    threshold=np.maximum(reference,(1.96*np.sqrt(2)*rough)[:,None])
    valid=np.isfinite(d)&np.isfinite(threshold)&(measurement['counts']>=6)&(np.asarray(source_distance)[:,None]<=.045)
    over=valid&(d>=.06)&(d>threshold+1e-12)
    persistent=over.sum(1)>=2
    eligible=(valid.sum(1)>=2)&(background<0)
    status=np.full(len(d),2,np.uint8);status[eligible]=0;status[eligible&persistent]=1;status[background>=0]=3
    reason=np.full(len(d),'insufficient_source_surface_support',dtype='U64')
    if not np.isfinite(reference).any():reason[:]='no_comparable_reference'
    reason[eligible]='within_local_noise_budget';reason[eligible&persistent]='all_direction_surface_difference'
    reason[background>=0]='common_structural_surface_excluded'
    ratios=np.divide(d,threshold,out=np.zeros(d.shape),where=np.isfinite(threshold)&(threshold>1e-12))
    ratio=np.median(np.where(valid,ratios,0),axis=1)
    coherence=np.median(measurement['coherence'],axis=1)
    score=100*(.45*np.clip(ratio/3,0,1)+.35*over.sum(1)/len(reference[0])+.20*coherence)
    score[~eligible]=0
    return {'threshold_m':threshold,'reference_threshold_m':reference,'valid':valid,'eligible':eligible,
        'persistent':persistent&eligible,'exceeds':over&(background<0)[:,None],
        'status':status,'reason':reason,'corrected_distance_m':d,'score':score}


def connected_ids(points,ids,radius=.16):
    if not len(ids):return []
    parent=np.arange(len(ids));tree=cKDTree(points[ids])
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in tree.query_pairs(radius):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    membership=np.array([find(i) for i in range(len(ids))])
    return [ids[membership==i] for i in np.unique(membership)]


def evaluate_reference_items(items,points,candidates,owner,nearest_instance,source_supported,areas):
    """Post-detection localization audit on source instance surfaces, not boxes.

    A captured item needs a connected patch of >=9 source-owned core points
    and >=0.04m2 assigned triangle area. This is localization on the existing
    twelve comparison items, not automatic object identity/type recognition.
    """
    result=[];own_area=areas[owner,np.arange(len(points))]
    for item in items:
        match=np.zeros(len(points),bool)
        for e,key in enumerate(('before_instance_id','after_instance_id')):
            instance=item.get(key)
            if instance is not None:match|=(owner==e)&(nearest_instance==instance)
        match &= source_supported
        ids=np.flatnonzero(match);detected=np.flatnonzero(match&candidates)
        groups=connected_ids(points,detected)
        qualifying=[g for g in groups if len(g)>=9 and own_area[g].sum()>=.04]
        outcome='captured' if qualifying else 'partial' if len(detected) else 'missed'
        result.append({'id':item['id'],'name':item['name'],'reference_type':item['type'],
            'outcome':outcome,'source_cores':len(ids),'candidate_cores':len(detected),
            'coverage_fraction':float(len(detected)/len(ids)) if len(ids) else None,
            'candidate_area_m2':float(own_area[detected].sum()),
            'largest_connected_cores':max(map(len,groups),default=0),'core_ids':ids.tolist(),
            'candidate_core_ids':detected.tolist(),'qualifying_components':len(qualifying)})
    return result
