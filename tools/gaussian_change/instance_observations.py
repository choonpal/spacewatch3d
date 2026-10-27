"""Connect unary instance-mask observations to the supplied mesh and source cameras.

No previous cross-capture assignments, event labels or historical room IDs enter
this adapter. Tracks are rebuilt in each capture's own mesh coordinates.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tools.mesh_simsac.core import read, sha, transform, camera_transform
from .registration import assign

CLASSES = {'chair':(19,'의자'),'bench':(23,'벤치'),'couch':(23,'소파'),'dining table':(15,'테이블'),
           'refrigerator':(50,'냉장고'),'tv':(89,'TV'),'laptop':(64,'노트북'),'suitcase':(37,'가방')}


def packet(side):
    value = side.get('_instance_segmentation')
    if value is None and side.get('instance_observations'):
        value = read(side['instance_observations'])
    if value is None:
        return None
    if value.get('schema') != 'mesh-instance-observations-v1' or value.get('coordinate_frame') != 'mesh':
        raise ValueError('Unsupported instance-observation coordinate/schema')
    if value.get('mesh_sha256') != sha(side['mesh']):
        raise ValueError('Instance observations do not belong to this exact input mesh')
    if value.get('cross_capture_pairing_included') or value.get('event_labels_included'):
        raise ValueError('Only unary segmentation evidence is accepted; no previous event decisions')
    return value


def common_points(points, side, epoch, registration):
    p = transform(points,side['world_from_mesh'])
    if not epoch:
        return p
    labels = assign(transform(p,registration['global_after_to_before']),registration['centers'])
    out = p.copy()
    for index,region in enumerate(registration['regions']):
        out[labels==index] = transform(p[labels==index],region['matrix_after_to_common'])
    return out


def rebuild_tracks(observations,height):
    tracks = []
    for observation in sorted(observations,key=lambda o:(int(o['frame']),str(o['id']))):
        points = np.asarray(observation.get('raw_points',[]),float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<12 or not np.isfinite(points).all():
            continue
        center = np.median(points,axis=0)
        delta = points-center
        points = points[(np.linalg.norm(delta[:,[0,2]],axis=1)<.35*height)&(abs(delta[:,1])<.45*height)]
        if len(points)<12 or np.ptp(points,axis=0).max()<.04*height:
            continue
        center = np.median(points,axis=0); choices=[]
        for i,t in enumerate(tracks):
            if t['class_name']!=observation['class_name']:
                continue
            distance=np.linalg.norm(center-t['center'])
            if distance>.18*height:
                continue
            nearest=cKDTree(t['points']).query(points)[0]
            overlap=float(np.mean(nearest<.05*height))
            if overlap>=.15 or distance<.07*height:
                choices.append((float(np.median(nearest))+.2*distance,i))
        if choices:
            t=tracks[min(choices)[1]]
            p=np.concatenate([t['points'],points]);_,ix=np.unique(np.floor(p/(.008*height)).astype('i8'),axis=0,return_index=True)
            p=p[np.sort(ix)];t['points']=p[::max(1,len(p)//2500)];t['center']=np.median(t['points'],axis=0)
            t['observations'].append(observation)
        else:
            tracks.append({'class_name':observation['class_name'],'center':center,'points':points,'observations':[observation]})
    return sorted(tracks,key=lambda t:(t['class_name'],*np.round(t['center'],5)))


def build_instances(side,epoch,registration):
    source=packet(side)
    if source is None:
        return None,None
    h=registration['height_u']
    with np.load(side['mesh'],allow_pickle=False) as d:
        vertices=(d['p'] if 'p' in d else d['vertices']).astype(float);faces=d['faces'].copy()
        semantics=d['face_semantic'].copy() if 'face_semantic' in d else np.full(len(faces),255)
        rgb=d['rgb'].copy() if 'rgb' in d else None
        observations=d['face_observations'].copy() if 'face_observations' in d else np.ones(len(faces))
    tracks=rebuild_tracks(source['observations'],h)
    centers=vertices[faces].mean(1);cost=np.full(len(faces),np.inf);owner=np.full(len(faces),-1)
    for index,t in enumerate(tracks):
        delta=centers-t['center']
        near=(np.linalg.norm(delta[:,[0,2]],axis=1)<.42*h)&(abs(delta[:,1])<.5*h)&~np.isin(semantics,[0,3,5,12])
        ids=np.flatnonzero(near)
        if not len(ids):continue
        dist=cKDTree(t['points']).query(centers[ids])[0]
        score=dist+.08*np.linalg.norm(delta[ids],axis=1)
        good=(dist<.085*h)&(score<cost[ids])
        cost[ids[good]]=score[good];owner[ids[good]]=index
    # Preserve complete semantic components and partition merged objects using
    # the existing mask-assisted nearest-face rule, rather than clipping every
    # instance to only the surface pixels seen from one viewing direction.
    for cls in set(CLASSES.get(t['class_name'],(-1,''))[0] for t in tracks):
        if cls<0:continue
        fi=np.flatnonzero(semantics==cls)
        if not len(fi):continue
        vertices_flat=faces[fi].ravel();local_face=np.repeat(np.arange(len(fi)),3);order=np.argsort(vertices_flat,kind='stable');same=vertices_flat[order[1:]]==vertices_flat[order[:-1]]
        graph=coo_matrix((np.ones(int(same.sum())),(local_face[order[:-1]][same],local_face[order[1:]][same])),shape=(len(fi),len(fi)))
        _,labels=connected_components(graph,directed=False);parents=[fi[labels==g] for g in np.unique(labels) if np.sum(labels==g)>=6];attached={}
        for index,t in enumerate(tracks):
            if CLASSES.get(t['class_name'],(-1,''))[0]!=cls or len({o['frame'] for o in t['observations']})<2:continue
            choices=[(float(cKDTree(vertices[np.unique(faces[part])]).query(t['center'])[0]),k) for k,part in enumerate(parents)]
            if not choices:continue
            distance,k=min(choices)
            if distance<.2*h:attached.setdefault(k,[]).append(index)
        for k,indices in attached.items():
            part=parents[k]
            if len(indices)>1:
                seeds=np.array([tracks[i]['center'] for i in indices]);separation=np.linalg.norm(seeds[:,None,[0,2]]-seeds[None,:,[0,2]],axis=-1);np.fill_diagonal(separation,np.inf)
                if separation.min()<.12*h:continue
            costs=[]
            for i in indices:
                t=tracks[i];dist=cKDTree(t['points']).query(centers[part],k=min(3,len(t['points'])))[0]
                costs.append(dist.mean(1)+.1*np.linalg.norm(centers[part]-t['center'],axis=1))
            owner[part]=np.asarray(indices)[np.argmin(np.stack(costs),axis=0)]
    rng=np.random.default_rng(20261027+epoch);objects=[];audit=[]
    for index,t in enumerate(tracks):
        own=np.flatnonzero(owner==index)
        frame_ids=sorted({int(o['frame']) for o in t['observations']})
        row={'class_name':t['class_name'],'source_frames':frame_ids,'observations':len(t['observations']),'assigned_faces':len(own)}
        if len(own)<12 or len(frame_ids)<2:
            row['rejected']='insufficient_mesh_or_independent_source_frames';audit.append(row);continue
        tri=common_points(vertices[faces[own]].reshape(-1,3),side,epoch,registration).reshape(-1,3,3)
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        if area.sum()<.002*h*h:
            row['rejected']='mesh_area_too_small';audit.append(row);continue
        picks=rng.choice(len(tri),512,p=area/area.sum());uv=rng.random((512,2));a=np.sqrt(uv[:,0]);w=np.c_[1-a,a*(1-uv[:,1]),a*uv[:,1]]
        points=np.einsum('ni,nij->nj',w,tri[picks]);bounds=np.stack([tri.min((0,1)),tri.max((0,1))]);region=int(assign(points.mean(0)[None],registration['centers'])[0])
        cls,label=CLASSES.get(t['class_name'],(1000+int(t['observations'][0].get('class_id',0)),t['class_name']))
        color=None
        if rgb is not None:
            pixels=rgb[np.unique(faces[own])].astype(float)
            if pixels.max()<=1:pixels*=255
            histogram=np.concatenate([np.histogram(pixels[:,k],bins=16,range=(0,256))[0] for k in [0,1,2]]).astype(float)
            color=histogram/max(np.linalg.norm(histogram),1e-12)
        raw_tri=vertices[faces[own]][picks]
        native_points=transform(np.einsum('ni,nij->nj',w,raw_tri),side['world_from_mesh'])
        obj={'id':f'I{epoch}-{len(objects)+1:04d}','epoch':epoch,'class_id':cls,'label':label,'class_name':t['class_name'],
             'faces':len(own),'original_face_ids':own.tolist(),'source_frames':frame_ids,'source_observation_ids':[o['id'] for o in t['observations']],
             'source_observation_count':len(frame_ids),'source_mesh_multi_observation_fraction':float(np.mean(observations[own]>=2)),
             'segmentation':'unary_image_instance_masks_partitioned_on_input_mesh','bounds':bounds.tolist(),'center':bounds.mean(0).tolist(),
             'region_index':region,'regional_supported':registration['regions'][region]['supported'],'_points':points,'_native_points':native_points,'_mesh_appearance':color,
             'candidate_ids':[],'link_scores':{}}
        row['instance_id']=obj['id'];audit.append(row);objects.append(obj)
    return objects,{'used':True,'input_observations':len(source['observations']),'rebuilt_tracks':len(tracks),'mesh_instances':len(objects),
                    'mesh_sha256':source['mesh_sha256'],'source_observations_sha256':source.get('source_observations_sha256'),
                    'source_cameras':len(source.get('cameras',[])),'tracks':audit,'previous_assignments_consumed':False}


def verification_observations(objects,sides,registration):
    """First-hit tests from actual opposite-capture source cameras, in common space."""
    import open3d as o3d
    h=registration['height_u'];scenes=[];poses=[]
    for epoch,side in enumerate(sides):
        source=packet(side)
        if source is None:return
        with np.load(side['mesh'],allow_pickle=False) as d:
            p=(d['p'] if 'p' in d else d['vertices']).astype(float);f=d['faces'].astype('u4')
            support=d['face_observations'].copy() if 'face_observations' in d else np.ones(len(f))
        p=p.astype('f4')
        scene=o3d.t.geometry.RaycastingScene();mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(p),o3d.core.Tensor(f));scene.add_triangles(mesh);scenes.append((scene,support))
        rows=[]
        for i,row in enumerate(source.get('cameras',[])):
            pose=np.asarray(row['camera_to_world'],float)
            rows.append((row.get('frame',i),pose))
        poses.append(rows)
    for obj in objects:
        opposite=1-obj['epoch'];scene,support=scenes[opposite]
        # Project the expected location through its own regional inverse and cast
        # against the original mesh/camera. A camera-origin regional transform
        # distorts a ray when the observed object is in another region.
        matrix=sides[opposite]['world_from_mesh'].copy()
        if opposite:
            matrix=np.asarray(registration['regions'][obj['region_index']]['matrix_after_to_common'])@matrix
        points=transform(obj['_points'],np.linalg.inv(matrix));center=np.median(points,axis=0)
        scale=float(np.cbrt(np.linalg.det(matrix[:3,:3])))
        nearest=sorted(poses[opposite],key=lambda row:np.linalg.norm(row[1][:3,3]-center));selected=[]
        for frame,pose in nearest:
            distance=np.linalg.norm(pose[:3,3]-center)
            if distance<.2*h/scale or distance>5*h/scale:continue
            if any(np.linalg.norm(pose[:3,3]-q[1][:3,3])<.1*h/scale for q in selected):continue
            selected.append((frame,pose))
            if len(selected)>=16:break
        field=registration['regions'][obj['region_index']]['validation'];error=field.get('plane_scatter_p90_u') or .025*h;tolerance=max(.05*h,2*error)/scale
        for frame,pose in selected:
            delta=points-pose[:3,3];expected=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(expected[:,None],1e-8)
            rays=np.c_[np.repeat(pose[None,:3,3],len(points),axis=0),direction].astype('f4');hit=scene.cast_rays(o3d.core.Tensor(rays))
            depth=hit['t_hit'].numpy();face=hit['primitive_ids'].numpy();valid=np.isfinite(depth)&(depth>.01*h)&(face<len(support))
            good=np.flatnonzero(valid);valid[good]&=support[face[good]]>=2;denom=max(int(valid.sum()),1)
            occupied=valid&(abs(depth-expected)<=tolerance);free=valid&(depth>expected+tolerance);occluded=valid&(depth<expected-tolerance)
            q=delta@pose[:3,:3];longitude=np.arctan2(q[:,0],q[:,2]);latitude=np.arctan2(q[:,1],np.hypot(q[:,0],q[:,2]));pixels=np.floor((longitude/(2*np.pi)+.5)*1036).astype(int)+1036*np.floor((latitude/np.pi+.5)*518).astype(int)
            obj['opposite_observations'].append({'frame_id':f'source:{opposite}:{frame}','view':None,'source_frame':frame,'source_epoch':opposite,
                'camera_position':pose[:3,3].tolist(),'observed_pixels':len(np.unique(pixels)), 'usable':bool(valid.mean()>=.2 and len(np.unique(pixels))>=12),
                'free_fraction':float(free.sum()/denom),'occluded_fraction':float(occluded.sum()/denom),'occupied_fraction':float(occupied.sum()/denom),
                'source':'actual_source_camera_mesh_first_hit','separation_u':.1*h})
        free=[r for r in obj['opposite_observations'] if r.get('source')=='actual_source_camera_mesh_first_hit' and r['usable'] and r['free_fraction']>=.65 and r['occluded_fraction']<=.2]
        obj['source_free_frames']=[r['frame_id'] for r in free]
