"""Geometry-only common floor/wall references on already aligned meshes.

Structural planes are an explicit stable-background assumption, not semantic
ground truth. No poses, meshes, or object transformations are optimized here.
Only mutually supported parts of matched planes are excluded from candidates.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from empirical_residual import spatial_quantile, persistent_exceedance, feature_summary
from mesh_change_noise import _nearest_layer, RADII

BACKGROUND = 3


def canonical(normal):
    normal = np.asarray(normal, float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    return normal if normal[np.argmax(np.abs(normal))] >= 0 else -normal


def plane_basis(normal):
    normal = np.asarray(normal)
    # For floors u follows world X; for walls u follows their horizontal span.
    u = np.array([1., 0, 0]) if abs(normal[2]) > .9 else np.cross([0., 0, 1], normal)
    u -= normal * (u @ normal)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(normal, u)])


def extract_planes(points, normals, seed=20261016, tolerance=.035, diagnostics=None):
    """Large, floor-adjacent vertical planes and the lowest broad horizontal plane.

    RANSAC only proposes planes. Normal consistency, geometric extent and
    floor contact reject tabletops/chair backs; no instance labels are read.
    """
    import open3d as o3d
    points = np.asarray(points); normals = np.asarray(normals)
    o3d.utility.random.seed(seed)
    candidates = []
    for kind, direction, maximum in [('floor', np.abs(normals[:, 2]) >= np.cos(np.deg2rad(15)), 8),
                                      ('wall', np.abs(normals[:, 2]) <= np.sin(np.deg2rad(15)), 18)]:
        remaining = np.flatnonzero(direction)
        for _ in range(maximum):
            if len(remaining) < 120:
                break
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining]))
            coefficients, chosen = cloud.segment_plane(tolerance, 3, 700)
            chosen = np.asarray(chosen, int)
            if len(chosen) < 100:
                break
            candidate_ids = remaining[chosen]
            remaining = np.delete(remaining, chosen)
            normal = canonical(coefficients[:3])
            if (kind == 'floor' and abs(normal[2]) < np.cos(np.deg2rad(15))) or (kind == 'wall' and abs(normal[2]) > np.sin(np.deg2rad(15))):
                continue
            ids = candidate_ids[np.abs(normals[candidate_ids] @ normal) >= np.cos(np.deg2rad(20))]
            if len(ids) < 100:
                continue
            p = points[ids]
            for _ in range(2):
                center = p.mean(0)
                _, vectors = np.linalg.eigh((p-center).T @ (p-center))
                normal = canonical(vectors[:, 0]); offset = -center @ normal
                keep = (np.abs(p @ normal + offset) <= tolerance) & (np.abs(normals[ids] @ normal) >= np.cos(np.deg2rad(20)))
                ids = ids[keep]; p = points[ids]
            if len(ids) < 100:
                continue
            basis = plane_basis(normal); uv = p @ basis
            span = np.ptp(uv, axis=0)
            area = len(np.unique(np.floor(uv/.12).astype(int), axis=0)) * .12**2
            if diagnostics is not None:
                diagnostics.append({'kind':kind,'samples':len(ids),'area':area,'span':span.tolist(),
                    'z_bounds':[float(p[:,2].min()),float(p[:,2].max())], 'normal':normal.tolist()})
            if kind == 'floor' and (area < 1.5 or min(span) < 1.):
                continue
            if kind == 'wall' and (area < .8 or np.ptp(p[:, 2]) < 1.1 or max(span) < 1.5 or (np.ptp(p[:,2])<1.5 and span[0]<2.)):
                continue
            candidates.append({'kind': kind, 'normal': normal.tolist(), 'offset_m': float(offset),
                'center': p.mean(0).tolist(), 'basis': basis.tolist(), 'sample_ids': ids,
                'support_grid_area_m2': area, 'span_m': span.tolist(),
                'bounds': {'min': p.min(0).tolist(), 'max': p.max(0).tolist()}})
    floors = [p for p in candidates if p['kind'] == 'floor']
    if not floors:
        return []
    floor_height = min(p['center'][2] for p in floors)
    result = []
    for p in candidates:
        if p['kind'] == 'floor' and p['center'][2] > floor_height + .20:
            continue
        # A wall's lower edge can be hidden by furniture. Require a tall plane
        # reaching room-wall height, rather than a directly visible floor seam.
        if p['kind'] == 'wall' and (p['bounds']['min'][2] > floor_height+.55 or p['bounds']['max'][2] < floor_height+1.2):
            continue
        result.append(p)
    # Noisy reconstruction may split one physical floor into several RANSAC
    # slabs. Join only low broad floor slabs, never elevated furniture tops.
    floors=[p for p in result if p['kind']=='floor']
    if len(floors)>1:
        ids=np.unique(np.concatenate([p['sample_ids'] for p in floors]));p=points[ids];center=p.mean(0)
        _,vectors=np.linalg.eigh((p-center).T@(p-center));normal=canonical(vectors[:,0]);basis=plane_basis(normal);uv=p@basis
        floor={'kind':'floor','normal':normal.tolist(),'offset_m':float(-center@normal),'center':center.tolist(),
            'basis':basis.tolist(),'sample_ids':ids,'support_grid_area_m2':len(np.unique(np.floor(uv/.12).astype(int),axis=0))*.12**2,
            'span_m':np.ptp(uv,axis=0).tolist(),'bounds':{'min':p.min(0).tolist(),'max':p.max(0).tolist()}}
        result=[floor]+[p for p in result if p['kind']!='floor']
    return result


def match_planes(planes, surfaces):
    """One-to-one normal/offset/footprint pairing; never an alignment step."""
    if not planes[0] or not planes[1]:
        return []
    costs = np.full((len(planes[0]), len(planes[1])), 1e6); possibilities = {}
    for i, a in enumerate(planes[0]):
        for j, b in enumerate(planes[1]):
            na, nb = np.array(a['normal']), np.array(b['normal'])
            if a['kind'] != b['kind'] or abs(na @ nb) < np.cos(np.deg2rad(8)):
                continue
            if na @ nb < 0:
                nb = -nb
            normal = canonical(na+nb); basis = plane_basis(normal)
            offset_gap = abs((np.array(a['center'])-b['center']) @ normal)
            if offset_gap > .12:
                continue
            uv = [surfaces[e][p['sample_ids']] @ basis for e, p in enumerate((a,b))]
            fractions = [(cKDTree(uv[1-e]).query(uv[e])[0] < .12).mean() for e in (0,1)]
            common_area = min(a['support_grid_area_m2']*fractions[0], b['support_grid_area_m2']*fractions[1])
            if min(fractions) < .2 or common_area < .6:
                continue
            costs[i,j] = offset_gap + .1*(1-min(fractions))
            possibilities[i,j] = {'kind':a['kind'], 'normal':normal.tolist(), 'basis':basis.tolist(),
                'planes':[a,b], 'offset_gap_m':float(offset_gap), 'overlap_fraction':list(map(float,fractions)),
                'common_support_grid_area_m2':float(common_area)}
    rows, cols = linear_sum_assignment(costs)
    pairs = [possibilities[i,j] for i,j in zip(rows,cols) if (i,j) in possibilities]
    pairs.sort(key=lambda p:(p['kind']!='floor',-p['common_support_grid_area_m2']))
    for i,pair in enumerate(pairs):
        pair['id'] = ('floor' if pair['kind']=='floor' else 'wall')+'-'+str(i+1)
    return pairs


def background_assignment(points, normals, pairs, surfaces, maximum_distance=.08, require_both=True):
    """Only the surface itself is excluded, not its 3D bounding box/interior."""
    result = np.full(len(points), -1, int); best = np.full(len(points), np.inf)
    for k,pair in enumerate(pairs):
        normal = np.array(pair['normal'])
        d=[];normal_offsets=[]
        for e,plane in enumerate(pair['planes']):
            support=surfaces[e][plane['sample_ids']]
            distance,nearest=cKDTree(support).query(points,workers=4)
            d.append(distance);normal_offsets.append(np.abs((points-support[nearest])@normal))
        distance = np.maximum(*d) if require_both else np.minimum(*d)
        # A mesh sample must actually lie on one epoch's structural surface.
        # Proximity alone would erase a low horizontal furniture base.
        own_surface=np.minimum(*normal_offsets)<=.025
        supported = own_surface & (distance <= maximum_distance) & (np.abs(normals @ normal) >= np.cos(np.deg2rad(20)))
        take = supported & (distance < best); result[take]=k; best[take]=distance[take]
    return result


def candidate_components(points, result, areas, radius=.16, minimum_cores=9, minimum_area=.04):
    """Connected surface evidence, not whole-object or occupancy labels."""
    ids=np.flatnonzero(result['persistent'])
    if not len(ids):return []
    tree=cKDTree(points[ids]);parent=np.arange(len(ids))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in tree.query_pairs(radius):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    labels=np.array([find(i) for i in range(len(ids))]);groups=[]
    accepted=np.zeros(len(points),bool)
    for label in np.unique(labels):
        group=ids[labels==label]
        if len(group)>=minimum_cores and areas[:,group].mean(0).sum()>=minimum_area:
            groups.append(group);accepted[group]=True
    rejected=result['persistent']&~accepted
    result['raw_persistent']=result['persistent'].copy()
    result['persistent']=accepted;result['status'][rejected]=0
    result['reason'][rejected]='insufficient_connected_exceedance'
    groups.sort(key=lambda g:-areas[:,g].mean(0).sum())
    return groups


def reference_split(uv, stripe_m=1.5, buffer_m=.30):
    """Fixed spatial stripes, with a 60cm gap between fit and check centers."""
    coordinate = np.asarray(uv)[:,0]
    stripe = np.floor(coordinate/stripe_m).astype(int)
    within = coordinate-stripe*stripe_m
    interior = (within >= buffer_m) & (within <= stripe_m-buffer_m)
    return interior & (stripe % 3 != 1), interior & (stripe % 3 == 1)


def oriented_measurements(queries, normals, surfaces, surface_normals, radii=RADII, half_depth=.65):
    """Same-layer distances with normal-gated neighborhoods.

    A vertical furniture leg can no longer bridge floor and tabletop layers.
    Reference calls additionally pass only samples belonging to that plane.
    """
    queries=np.asarray(queries,float);normals=np.asarray(normals,float)
    normals=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    shape=(len(queries),len(radii)); centers=[];spreads=[];counts=[]
    for surface,sn in zip(surfaces,surface_normals):
        center=np.full(shape,np.nan);spread=np.full(shape,np.nan);count=np.zeros(shape,dtype=np.uint32)
        tree=cKDTree(surface)
        for start in range(0,len(queries),256):
            lists=tree.query_ball_point(queries[start:start+256],np.hypot(max(radii),half_depth),workers=4)
            for row,ids in enumerate(lists):
                i=start+row;ids=np.asarray(ids,int)
                ids=ids[np.abs(sn[ids]@normals[i])>=np.cos(np.deg2rad(25))]
                if len(ids)<6:continue
                delta=surface[ids]-queries[i];along=delta@normals[i]
                tangent=np.maximum(0,np.einsum('ij,ij->i',delta,delta)-along**2)
                for j,radius in enumerate(radii):
                    layer=_nearest_layer(along[(tangent<=radius**2)&(np.abs(along)<=half_depth)])
                    if layer is None:continue
                    median=np.median(layer);low,high=np.quantile(layer,[.1,.9])
                    center[i,j]=median;spread[i,j]=max(1.4826*np.median(np.abs(layer-median)),(high-low)/2.563)
                    count[i,j]=len(layer)
        centers.append(center);spreads.append(spread);counts.append(count)
    return {'distance_m':centers[1]-centers[0], 'centers_m':np.stack(centers,axis=-1),
        'spreads_m':np.stack(spreads,axis=-1), 'counts_by_epoch':np.stack(counts,axis=-1),
        'normals':normals,'radii_m':np.array(radii)}


def calibrate_plane(uv, distance, valid, radii=RADII):
    fit,check=reference_split(uv)
    enough=valid.sum(1)>=2;fit &= enough;check &= enough
    xyz=np.column_stack([uv,np.zeros(len(uv))]);thresholds=[];raw_thresholds=[];blocks=[];distributions=[];coefficients=[];direct_thresholds=[]
    corrected=np.full(distance.shape,np.nan)
    for j,radius in enumerate(radii):
        ids=np.flatnonzero(fit & valid[:,j])
        design=np.column_stack([np.ones(len(uv)),uv])
        if len(ids):
            _,membership,counts=np.unique(np.floor(uv[ids]/.16).astype(int),axis=0,return_inverse=True,return_counts=True)
            weights=np.sqrt(1./counts[membership])
            coefficient=np.linalg.lstsq(design[ids]*weights[:,None],distance[ids,j]*weights,rcond=None)[0]
            corrected[:,j]=distance[:,j]-design@coefficient
        else:coefficient=np.full(3,np.nan)
        coefficients.append([float(v) if np.isfinite(v) else None for v in coefficient])
        raw,_=spatial_quantile(np.abs(distance[ids,j]),xyz[ids]);raw_thresholds.append(raw)
        direct,b=spatial_quantile(np.abs(corrected[ids,j]),xyz[ids]);direct_thresholds.append(direct)
        # A fitted plane trend's training residuals understate spatial prediction
        # error. Estimate that error by leaving 1.5m fit blocks out, with an
        # additional 56cm support buffer. Check stripes never train this bound.
        cross=np.full(len(ids),np.nan)
        cells=np.floor(uv[ids]/1.5).astype(int)
        for cell in np.unique(cells,axis=0):
            held=(cells==cell).all(1)
            low,high=cell*1.5,(cell+1)*1.5
            gap=np.maximum(0,np.maximum(low-uv[ids],uv[ids]-high))
            train=(~held)&(np.linalg.norm(gap,axis=1)>.56)
            if train.sum()<10:continue
            selected=ids[train]
            _,membership,counts=np.unique(np.floor(uv[selected]/.16).astype(int),axis=0,return_inverse=True,return_counts=True)
            w=np.sqrt(1./counts[membership]);coef=np.linalg.lstsq(design[selected]*w[:,None],distance[selected,j]*w,rcond=None)[0]
            cross[held]=distance[ids[held],j]-design[ids[held]]@coef
        predictive,_=spatial_quantile(np.abs(cross),xyz[ids])
        t=max(direct,predictive) if direct is not None and predictive is not None else None
        thresholds.append(t);blocks.append(b)
        distributions.append({'radius_m':float(radius),'absolute_residuals_m':np.abs(cross[np.isfinite(cross)]).tolist(),
            'summary':feature_summary(np.abs(cross))})
    t=np.array([v if v is not None else np.nan for v in thresholds])
    _,persistent=persistent_exceedance(corrected,t,valid)
    fraction=float(persistent[check].mean()) if check.any() else None
    fit_ready=fit.sum()>=20 and min(blocks)>=6 and all(t is not None and np.isfinite(t) for t in thresholds)
    check_ready=check.sum()>=10
    enabled=bool(fit_ready and check_ready and fraction<=.2)
    if not fit_ready:fraction=None
    return {'thresholds_m':thresholds,'raw_thresholds_m':raw_thresholds,'direct_fit_thresholds_m':direct_thresholds,'bias_coefficients_m':coefficients,
        'reference_block_counts':blocks,'fit_core_count':int(fit.sum()),
        'fit_mask':fit,'check_mask':check,'reference_distribution':distributions,
        'fit_ready':bool(fit_ready),'enabled':enabled,
        'check_reason':'reference_accepted_on_spatial_check' if enabled else 'insufficient_reference_support' if not fit_ready else 'insufficient_check_support' if not check_ready else 'reference_does_not_transfer_to_check',
        'checks':[{'valid_cores':int(check.sum()),'exceedance_fraction':fraction,'passed':enabled,
            'absolute_residual_summary':[feature_summary(np.abs(corrected[check,j])) for j in range(len(radii))],
            'raw_absolute_residual_summary':[feature_summary(np.abs(distance[check,j])) for j in range(len(radii))]}]}


def classify_targets(measurement, reference, paired_observed, background, bias=None):
    d=measurement['distance_m']-(0 if bias is None else bias);s=measurement['spreads_m']
    local=1.96*np.sqrt(s[:,:,0]**2+s[:,:,1]**2)
    threshold=np.maximum(reference,local)
    valid=np.isfinite(d)&np.isfinite(threshold)&paired_observed&(measurement['counts_by_epoch'].min(-1)>=6)
    over,persistent=persistent_exceedance(d,threshold,valid)
    # Same application resolution as the fixed-2cm control, independently of
    # the estimated background error. This is not a fitted registration sigma.
    over &= np.abs(d)>=.06
    persistent=np.maximum((over&(d>0)).sum(1),(over&(d<0)).sum(1))>=2
    eligible=(valid.sum(1)>=2)&(background<0)
    status=np.full(len(d),2,np.uint8);status[eligible]=0;status[eligible&persistent]=1;status[background>=0]=BACKGROUND
    reason=np.full(len(d),'no_comparable_reference',dtype='U64')
    reason[~paired_observed.any(1)]='insufficient_paired_surface_observation'
    reason[eligible]='below_empirical_reference';reason[eligible&persistent]='exceeds_comparable_reference'
    reason[background>=0]='common_structural_surface_excluded'
    return {'threshold_m':threshold,'valid':valid,'eligible':eligible,'persistent':persistent&eligible,
        'exceeds':over&(background<0)[:,None],'status':status,'reason':reason,'reference_threshold_m':reference,
        'corrected_distance_m':d}
