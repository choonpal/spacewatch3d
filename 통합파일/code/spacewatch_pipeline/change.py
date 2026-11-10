"""Automatic structural alignment plus the pinned Task5a surface algorithm."""
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .io import read,save,result,sha
from .geometry import transform,similarity
from spacewatch3d.tasks.task_5a_change_detection.surface_geometry import (
    face_geometry,triangle_scene,compare_surfaces,error_baseline,coherent_patches,classify_patch)

def pair(job):
    captures=job['config']['captures']
    return [job['dependencies'][c['id']+'.semantic'] for c in captures]

def structural_cloud(seg,voxel):
    data=np.load(seg['arrays']);v,f,l=data['vertices'],data['faces'],data['face_class_id']
    names={r['id']:r['name'] for r in read(seg['vocabulary'])}
    ids=[k for k,val in names.items() if val in ['wall','floor','ceiling']]
    vertices=np.unique(f[np.isin(l,ids)])
    if len(vertices)<100:raise ValueError('Insufficient structural mesh for alignment')
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v[vertices]))
    pc=pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4,max_nn=40))
    return pc

def align(job):
    out=Path(job['output']);before,after=pair(job)
    opt=job['config'].get('alignment',{});voxel=opt.get('voxel_m',.08)
    if opt.get('prepared_source'):
        source=Path(opt['prepared_source'])
        registration=read(source/'registration.json')
        expected=registration['inputs']
        for capture,prior in zip(job['config']['captures'],expected):
            current=job['dependencies'][capture['id']+'.mesh']
            if current['geometry_sha256']!=prior['geometry_file_sha256']:
                raise ValueError('Prepared regional alignment belongs to a different mesh. Remove prepared_source for a new capture.')
        files=[str(p) for name in ['maps','new-inputs','object-comparison-v1/photos']
               for p in (source/name).rglob('*') if p.is_file()]
        files += [str(source/n) for n in ['registration.json','alignment-context.json','input-provenance.json']]
        result(out,status='prepared_regional',prepared_source=str(source),
               external_artifacts=files,method='Existing geometry-bound regional registration; Task5a applies its full regional matrices',
               new_registration_estimated=False,metric_scale_verified=False)
        return
    try:
        target=structural_cloud(before,voxel);source=structural_cloud(after,voxel)
    except ValueError as exc:
        if 'Insufficient structural mesh' not in str(exc):raise
        save(out/'alignment.json',dict(status='unavailable',reason=str(exc),baseline_from_current=None))
        result(out,alignment=str(out/'alignment.json'),status='unavailable',reason=str(exc),baseline_from_current=None)
        return
    s=np.asarray(source.points);t=np.asarray(target.points)
    # Holdout structural vertices never enter ICP, but still share reconstruction inputs.
    rng=np.random.default_rng(5201);sh=rng.random(len(s))<.2;th=rng.random(len(t))<.2
    fit_s=source.select_by_index(np.flatnonzero(~sh));fit_t=target.select_by_index(np.flatnonzero(~th))
    seeds=[]
    if opt.get('initial_transform'):
        seeds=[similarity(opt['initial_transform'])[0]]
    else:
        for yaw in [0,np.pi/2,np.pi,3*np.pi/2]:
            r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
            m=np.eye(4);m[:3,:3]=r;m[:3,3]=np.median(t,axis=0)-np.median(s,axis=0)@r.T
            seeds.append(m)
        seeds.append(np.eye(4))
        if opt.get('fpfh',True):
            fs=o3d.pipelines.registration.compute_fpfh_feature(fit_s,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=100))
            ft=o3d.pipelines.registration.compute_fpfh_feature(fit_t,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=100))
            o3d.utility.random.seed(5201)
            match=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                fit_s,fit_t,fs,ft,True,voxel*2,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel*2)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(20000,.999))
            seeds.append(match.transformation)
    rows=[]
    for seed in seeds:
        tf=seed
        for distance in [.6,.3,.15]:
            registration=o3d.pipelines.registration.registration_icp(fit_s,fit_t,distance,tf,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            tf=registration.transformation
        moved=transform(s[sh],tf)
        distances=cKDTree(t).query(moved,workers=4)[0]
        reverse=cKDTree(transform(s,tf)).query(t[th],workers=4)[0]
        overlap=np.r_[distances,reverse]<opt.get('evaluation_radius_m',.3)
        d=np.r_[distances,reverse]
        rows.append(dict(matrix=tf.tolist(),overlap=float(overlap.mean()),
                         p90=float(np.quantile(d[overlap],.9)) if overlap.any() else 1e9,
                         median=float(np.median(d)),fitness=float(registration.fitness)))
    rows.sort(key=lambda r:(-r['overlap'],r['p90']))
    best=rows[0];tf=np.array(best['matrix'])
    distinct=[r for r in rows[1:] if np.linalg.norm(np.array(r['matrix'])-tf)>.25]
    ambiguous=bool(distinct and best['overlap']-distinct[0]['overlap']<opt.get('ambiguity_overlap_margin',.02))
    good=best['overlap']>=opt.get('min_overlap',.55) and best['p90']<=opt.get('max_p90_m',.15) and not ambiguous
    # Point-to-plane observability: parallel corridor surfaces cannot constrain every direction.
    moved=transform(np.asarray(fit_s.points),tf);d,ix=cKDTree(np.asarray(fit_t.points)).query(moved)
    keep=d<.15;n=np.asarray(fit_t.normals)[ix[keep]];p=moved[keep]
    centered=p-p.mean(0) if len(p) else p
    J=np.c_[np.cross(centered,n),n]
    singular=np.linalg.svd(J,compute_uv=False) if len(J)>=6 else np.zeros(6)
    ratio=float(singular[-1]/max(singular[0],1e-12));good &= ratio>opt.get('min_observability_ratio',.001)
    save(out/'alignment.json',dict(status='accepted' if good else 'underconstrained',
         baseline_from_current=tf.tolist(),before_capture=job['config']['captures'][0]['id'],
         after_capture=job['config']['captures'][1]['id'],method='multi-initialization structural SE3 ICP',
         metrics=best,ambiguous=ambiguous,observability_ratio=ratio,candidates=rows,
         coordinate_units='estimated meters unless input scale is verified',
         limitation='Global rigid alignment; local reconstruction drift is not corrected. Heldout vertices share reconstruction inputs.'))
    result(out,alignment=str(out/'alignment.json'),status='accepted' if good else 'underconstrained',
           baseline_from_current=tf.tolist(),registration_error_m=max(best['p90'],.02))

def cameras(job,index,alignment):
    c=job['config']['captures'][index];src=job['dependencies'][c['id']+'.reconstruct']
    root=Path(src['source_run']);tf,scale=similarity(read(src['coordinates'])['source_to_mesh'])
    raw=read(root/'result/camera_poses.json');manifest=read(root/'input/manifest.json')
    tf=alignment@tf
    rows=[]
    for i,row in enumerate(raw):
        pose=np.array(row['camera_to_world']);center=transform(pose[None,:3,3],tf)[0]
        rows.append(dict(frame=i,timestamp_s=manifest['frames'][i]['timestamp_s'],camera_in_map=center.tolist(),
                         source_image=str(root/'input'/manifest['frames'][i]['file'])))
    return rows

def views(points,rows,scene,tolerance,height):
    if len(points)>128:points=points[np.linspace(0,len(points)-1,128).astype(int)]
    order=np.argsort([np.linalg.norm(np.asarray(r['camera_in_map'])-np.median(points,0)) for r in rows])[:32]
    values=[]
    for ix in order:
        row=rows[ix];origin=np.array(row['camera_in_map']);delta=points-origin;depth=np.linalg.norm(delta,axis=1)
        rays=np.c_[np.broadcast_to(origin,points.shape),delta/np.maximum(depth[:,None],1e-12)].astype('f4')
        hits=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy();finite=np.isfinite(hits)
        margin=hits-depth;visible=finite&(abs(margin)<=max(.025*height,tolerance*.5))&(depth>.1*height)
        free=finite&(margin>tolerance);occluded=finite&(margin< -tolerance)
        values.append(dict(**row,visible_fraction=float(visible.mean()),free_fraction=float(free.mean()),
                           occluded_fraction=float(occluded.mean()),finite_fraction=float(finite.mean()),
                           score=float(visible.mean()+free.mean()*.5)))
    return values

def changes(job):
    out=Path(job['output']);segments=pair(job);alignment=job['dependencies']['pair.align']
    if alignment['status']=='unavailable':
        save(out/'changes.json',dict(scope='surface_regions_not_object_events',status='unavailable',alignment=alignment,events=[],audit=[],no_object_identity_claim=True))
        result(out,changes=str(out/'changes.json'),candidates=0,deferred=0,alignment_status='unavailable',comparison_performed=False)
        return
    if alignment.get('prepared_source'):
        from spacewatch3d.tasks.task_5a_change_detection.dorm_surface import main
        source=Path(alignment['prepared_source'])
        main(source,out/'published')
        tour=read(out/'published/tour.json')
        rows=tour.get('events',[])+tour.get('records',[])
        result(out,changes=str(out/'published/tour.json'),
               candidates=len(tour.get('events',[])),deferred=sum(r.get('decision',r.get('status'))=='deferred' for r in rows),
               alignment_status='prepared_regional',backend='upstream_task5a_dorm_surface',
               limitations='Existing source regional alignment reused after exact whole-mesh hash verification')
        return
    tf=np.array(alignment['baseline_from_current']);identity=np.eye(4)
    meshes=[]
    for i,seg in enumerate(segments):
        z=np.load(seg['arrays']);meshes.append((transform(z['vertices'],tf if i else identity),z['faces']))
    scenes=[triangle_scene(v,f) for v,f in meshes]
    rows=[cameras(job,i,tf if i else identity) for i in range(2)]
    height=job['config'].get('changes',{}).get('reference_height_m',2.7)
    events=[];audit=[]
    for epoch,(v,f) in enumerate(meshes):
        other=1-epoch;ov,of=meshes[other]
        centers,normals,area=face_geometry(v,f)
        distances,target_normals,_=compare_surfaces(v,f,ov,of,scenes[other])
        baseline=error_baseline(distances,normals,target_normals,area,alignment['registration_error_m'],height)
        patches,counts=coherent_patches(v,f,distances,baseline['threshold_u'],height)
        audit.append(dict(epoch=epoch,**baseline,**counts))
        for p in patches:
            q=centers[p['face_ids']]
            own=views(q,rows[epoch],scenes[epoch],baseline['threshold_u'],height)
            opposite=views(q,rows[other],scenes[other],baseline['threshold_u'],height)
            status,reason,own_count,free_count=classify_patch([dict(own_views=own,opposite_views=opposite)],
                alignment['status']=='accepted' and baseline['stable_surface_supported'],height)
            events.append(dict(id=f'surface_{epoch}_{len(events):04d}',epoch=epoch,status=status,reason=reason,
                face_ids=p['face_ids'].tolist(),bounds=p['bounds'].tolist(),area_m2=p['area_u2'],
                distance_p50_m=p['distance_p50_u'],distance_p90_m=p['distance_p90_u'],
                own_views=sorted(own,key=lambda r:-r['score'])[:4],opposite_views=sorted(opposite,key=lambda r:-r['score'])[:4],
                independent_own_views=own_count,independent_free_views=free_count))
    save(out/'changes.json',dict(scope='surface_regions_not_object_events',alignment=alignment,events=events,audit=audit,
         no_object_identity_claim=True,algorithm='pinned SpaceWatch3D Task5a surface_geometry',
         limitations=['Reconstruction-based visibility; no independent RGB-D truth',
                     'Global SE3 adapter; the upstream Dorm regional alignment is not generalized here']))
    result(out,changes=str(out/'changes.json'),candidates=sum(e['status']=='candidate' for e in events),
           deferred=sum(e['status']=='deferred' for e in events),alignment_status=alignment['status'])
