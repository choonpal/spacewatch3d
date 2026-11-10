"""Stable point cloud, early-semantic-guided meshing, and exact multi-frame voting."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from .io import ROOT, read, save, sha, digest, result

def transform(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]

def similarity(matrix):
    m=np.asarray(matrix,float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1]):
        raise ValueError('Invalid homogeneous transform')
    scale=np.linalg.norm(m[:3,0])
    r=m[:3,:3]/scale
    if scale<=0 or np.linalg.det(r)<.99999 or not np.allclose(r.T@r,np.eye(3),atol=1e-6):
        raise ValueError('Expected positive uniform-scale rigid transform')
    return m,float(scale)

def context(job):
    c=job['capture']; deps=job['dependencies']; prefix=c['id']
    return c,Path(job['output']),deps,prefix

def reconstruct(job):
    c,out,_,_=context(job)
    if c.get('source_run'):
        source=Path(c['source_run'])
        if read(source/'status.json')['status']!='complete':
            raise ValueError('Source reconstruction is not complete')
    else:
        cfg=read(c['reconstruction_config'])
        cfg['execution']['physical_gpu']=job['gpu']['index']
        if c['id']+'.masks' in job['dependencies']:
            cfg['semantic_masks']=job['dependencies'][c['id']+'.masks']['masks']
        save(out/'generator.json',cfg)
        source=out/'reconstruction'
        subprocess.run([sys.executable,str(ROOT/'vendor/panovggt_pipeline_v2/run.py'),
                        '--config',str(out/'generator.json'),'--output',str(source)],check=True)
    settings=read(source/'config.json')
    if settings.get('generator_version')!='2.0.0':
        raise ValueError('Integration v2 requires point cloud stable v2.0.0')
    required=['result/map.ply','result/camera_poses.json','input/manifest.json','config.json',
              'prepared.npz','optimized.npz','component_filter/keep_mask.npy','component_filter/report.json',
              'unfiltered_result/map.ply']
    external=[str(source/p) for p in required]
    # The hash records of all windows bind the projection caches to reconstruction.
    external += [str(p) for p in sorted((source/'windows').glob('*.json'))]
    manifest=read(source/'input/manifest.json')
    poses=read(source/'result/camera_poses.json')
    if len(manifest['frames'])!=len(poses):
        raise ValueError('Frame/pose count mismatch')
    for row in manifest['frames']:
        p=source/'input'/row['file']
        if sha(p)!=row['sha256']:
            raise ValueError(f'Frame hash mismatch: {p}')
        external.append(str(p))
    external += [str(p) for p in sorted((source/'masks').glob('*.png'))]
    coords=read(c['coordinates']);tf,scale=similarity(coords['source_to_mesh'])
    if coords.get('up') != [0,0,1]:
        raise ValueError('Coordinates must declare metric/estimated metric +Z up')
    external.append(c['coordinates'])
    bridges=read(source/'unverified_bridges.json') if (source/'unverified_bridges.json').exists() else None
    result(out,source_run=str(source),point_cloud=str(source/'result/map.ply'),
           coordinates=str(Path(c['coordinates']).resolve()),frames=len(poses),
           metric_scale_verified=coords.get('metric_scale_verified',False),unverified_bridges=bridges,
           external_artifacts=external)

def mesh(job):
    import open3d as o3d
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    import trimesh
    c,out,deps,prefix=context(job)
    from .early_semantics import filtered_source
    src=filtered_source(deps,prefix);source=Path(src['source_run'])
    coords=read(src['coordinates']);tf,scale=similarity(coords['source_to_mesh'])
    if c.get('mesh_arrays'):
        path=Path(c['mesh_arrays'])
        data=np.load(path)
        v,f,rgb=data['vertices'],data['faces'],data['rgb']
        external=[str(path)]
        report=read(c['mesh_report'])
        if report['source_sha256']!=sha(src['point_cloud']):
            raise ValueError('Cached mesh was made from another point cloud')
        if not any(r.get('geometry_sha256')==sha(path) for r in report['variants']):
            raise ValueError('Cached mesh geometry hash mismatch')
        external.append(c['mesh_report'])
    else:
        opts=job['config'].get('mesh',{}); voxel=opts.get('voxel_source_units',.025)
        cloud=PlyData.read(src['point_cloud'],mmap='r')['vertex'].data
        xyz=np.column_stack([cloud[k] for k in 'xyz']).astype(float)
        rgb=np.column_stack([cloud[k] for k in ['red','green','blue']])/255.
        pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz));pc.colors=o3d.utility.Vector3dVector(rgb)
        protected=None
        if src.get('protected_points'):
            protected=np.load(src['protected_points'],mmap_mode='r')
            # Sample observed small-object support more densely, without creating geometry.
            coarse=pc.select_by_index(np.flatnonzero(~protected)).voxel_down_sample(voxel)
            fine=pc.select_by_index(np.flatnonzero(protected)).voxel_down_sample(voxel*opts.get('protected_voxel_ratio',.5))
            protected_samples=np.r_[np.zeros(len(coarse.points),bool),np.ones(len(fine.points),bool)]
            pc=coarse+fine
        else:pc=pc.voxel_down_sample(voxel)
        pc.transform(tf)
        p=np.asarray(pc.points)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4*scale,max_nn=40))
        poses=np.array([r['camera_to_world'] for r in read(source/'result/camera_poses.json')])
        centers=transform(poses[:,:3,3],tf)
        distance,ix=cKDTree(centers).query(p,k=min(3,len(centers)),workers=4)
        if ix.ndim==1: ix,distance=ix[:,None],distance[:,None]
        view=((centers[ix]-p[:,None])/np.maximum(distance[:,:,None]**2,1e-12)).sum(1)
        normals=np.asarray(pc.normals);normals[(normals*view).sum(1)<0]*=-1
        print('POISSON',len(p),flush=True)
        raw,_=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc,depth=opts.get('poisson_depth',9),
                                                                       scale=1.1,n_threads=4)
        smooth=raw.filter_smooth_taubin(number_of_iterations=opts.get('taubin_iterations',5))
        v=np.asarray(smooth.vertices).astype(np.float32);f=np.asarray(smooth.triangles).astype(np.uint32)
        _,ix=cKDTree(p).query(v,workers=4)
        if protected is not None:
            preserve=protected_samples[ix]
            v[preserve]=np.asarray(raw.vertices,dtype=np.float32)[preserve]
        rgb=np.rint(np.asarray(pc.colors)[ix]*255).clip(0,255).astype(np.uint8)
        path=out/'geometry.npz';np.savez_compressed(path,vertices=v,faces=f,rgb=rgb)
        external=[]
        trimesh.Trimesh(v,f,vertex_colors=rgb,process=False).export(out/'mesh_rgb.glb')
    if not len(f) or not np.isfinite(v).all() or f.max()>=len(v):
        raise ValueError('Invalid triangle mesh')
    result(out,geometry=str(path),coordinates=src['coordinates'],vertices=len(v),faces=len(f),
           geometry_sha256=sha(path),external_artifacts=external,semantic_labels_used_for_meshing=bool(src.get('protected_points')))

def infer(job):
    c,out,deps,prefix=context(job)
    data=deps.get(prefix+'.inputs')
    source=Path(c['source_run']) if c.get('source_run') else None
    catalog=read(job['config']['vocabulary'])
    names=[v['name'] for v in catalog]
    if [v['id'] for v in catalog]!=list(range(len(catalog))) or len(names)>254 or len(names)<2:
        raise ValueError('Vocabulary requires 2..254 contiguous IDs; 255 is unknown')
    manifest=read(data['manifest']) if data else read(source/'input/manifest.json')
    if c.get('predictions'):
        dest=Path(c['predictions'])
        if read(dest/'complete.json')['vocabulary_sha256']!=sha(job['config']['vocabulary']):
            raise ValueError('Cached OOOPS vocabulary mismatch')
        original=read(c['prediction_manifest'])
        if original['source_video_sha256']!=manifest['source_sha256'] or len(original['frames'])!=len(manifest['frames']):
            raise ValueError('Cached OOOPS source frames mismatch')
        for a,b in zip(original['frames'],manifest['frames']):
            if a['source_sha256']!=b['sha256']:
                raise ValueError('OOOPS frame pixel identity mismatch')
        external=[str(dest/f'{i:05d}.npz') for i in range(len(manifest['frames']))]
        external += [str(dest/'complete.json'),c['prediction_manifest'],job['config']['vocabulary']]
    else:
        from .ooops import predict
        dest=out/'predictions';dest.mkdir()
        predict(job,[Path(p) for p in data['images']] if data else [source/'input'/row['file'] for row in manifest['frames']],dest,names)
        external=[job['config']['vocabulary']]
    result(out,predictions=str(dest),frames=len(manifest['frames']),classes=names,
           vocabulary=job['config']['vocabulary'],external_artifacts=external)

def export_semantic(path,arrays,catalog):
    """Portable meshes with exact face labels; GLB corners preserve hard color boundaries."""
    import trimesh
    from plyfile import PlyData,PlyElement
    v,f,rgb,labels=(arrays[k] for k in ['vertices','faces','rgb','face_class_id'])
    palette=np.full((256,3),145,np.uint8)
    for c in catalog:palette[c['id']]=c['rgb']
    vertices=np.empty(len(v),dtype=[(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ['red','green','blue']])
    for i,k in enumerate('xyz'):vertices[k]=v[:,i]
    for i,k in enumerate(['red','green','blue']):vertices[k]=rgb[:,i]
    faces=np.empty(len(f),dtype=[('vertex_indices','<i4',(3,)),('semantic_id','<u2')])
    faces['vertex_indices']=f;faces['semantic_id']=labels
    PlyData([PlyElement.describe(vertices,'vertex'),PlyElement.describe(faces,'face')],text=False).write(str(path/'mesh_semantic.ply'))
    corners=v[f].reshape(-1,3);triangles=np.arange(len(corners)).reshape(-1,3)
    trimesh.Trimesh(corners,triangles,vertex_colors=np.repeat(palette[labels],3,axis=0),process=False,
                    metadata={'classes':catalog,'attribute_scope':'face labels in companion PLY/NPZ'}).export(path/'mesh_semantic.glb')
    trimesh.Trimesh(v,f,vertex_colors=rgb,process=False).export(path/'mesh_rgb.glb')
    return {name:str(path/file) for name,file in [('semantic_mesh','mesh_semantic.glb'),('semantic_ply','mesh_semantic.ply'),('rgb_mesh','mesh_rgb.glb')]}

def semantic(job):
    import open3d as o3d
    c,out,deps,prefix=context(job)
    sys.path.insert(0,str(ROOT/'vendor/segmentation'))
    from geometry import source_frames,build_index
    from core import voxel_keys,buckets,exact_lookup,resize_labels_to_source,frame_votes,fuse_votes,VOTE_DTYPE
    from .early_semantics import filtered_source
    src=filtered_source(deps,prefix);mesh=deps[prefix+'.mesh'];prediction=deps[prefix+'.infer']
    if src.get('source_index'):c=dict(c,source_index=src['source_index'])
    source=Path(src['source_run']);data=np.load(mesh['geometry'])
    v,f,rgb=data['vertices'],data['faces'],data['rgb'];nf=len(f)
    options=job['config'].get('semantic',{})
    tf,scale=similarity(read(mesh['coordinates'])['source_to_mesh'])
    external=[]
    if c.get('semantic_arrays'):
        cached=Path(c['semantic_arrays']);a=np.load(cached)
        rep=read(c['semantic_report'])
        if rep['geometry_sha256']!=mesh['geometry_sha256'] or not np.array_equal(a['vertices'],v) or not np.array_equal(a['faces'],f):
            raise ValueError('Semantic cache geometry mismatch')
        if [x['name'] for x in rep['classes']] != prediction['classes']:
            raise ValueError('Semantic cache vocabulary mismatch')
        external=[str(cached),c['semantic_report']]
        exports=export_semantic(out,a,read(prediction['vocabulary']))
        result(out,arrays=str(cached),coordinates=mesh['coordinates'],vocabulary=prediction['vocabulary'],
               external_artifacts=external,report=rep,**exports)
        return
    projection_config=dict(source_run=str(source),output=str(out))
    if c.get('source_index'):
        indexdir=Path(c['source_index'])
        # Existing source index must have exact v2 verification and every file intact.
        complete=read(indexdir/'complete.json')
        for p,r in complete['files'].items():
            if sha(p)!=r['sha256']:
                raise ValueError('Source index cache changed: '+p)
            external.append(p)
        index=read(indexdir/'index.json')
        from filtered_source import source_layout,build_row_map
        from plyfile import PlyData
        layout=source_layout(projection_config,read(source/'config.json'))
        # Verify the cached row mapping belongs to this exact source, even when point counts coincide.
        proof=build_row_map(PlyData.read(layout['unfiltered'],mmap='r')['vertex'].data,
                            PlyData.read(layout['target'],mmap='r')['vertex'].data,
                            np.load(layout['keep_mask'],mmap_mode='r'),out/'verified_rowmap.npy')
        if not np.array_equal(np.load(out/'verified_rowmap.npy',mmap_mode='r'),np.load(index['unfiltered_to_target'],mmap_mode='r')):
            raise ValueError('Cached source index maps another reconstruction')
    else:
        build_index(projection_config,dict(identity=digest([src,mesh]),source_frames=src['frames']))
        indexdir=out/'source_index';index=read(indexdir/'index.json')
    keys=[np.load(r['path'],mmap_mode='r') for r in index['buckets']]
    rowmap=np.load(index['unfiltered_to_target'],mmap_mode='r')
    raycaster=o3d.t.geometry.RaycastingScene(nthreads=4)
    raycaster.add_triangles(o3d.core.Tensor(v.astype('f4')),o3d.core.Tensor(f.astype('u4')))
    poses=np.array([r['camera_to_world'] for r in read(source/'result/camera_poses.json')])
    classes=read(prediction['vocabulary']);excluded=[r['id'] for r in classes if not r.get('static_mesh',True)]
    records=[];coverage=np.zeros(nf,np.uint32)
    for frame,xyz,mask,_ in source_frames(projection_config):
        cells=voxel_keys(xyz,index['voxel']);bucket=buckets(cells,len(keys));ids=np.empty(len(xyz),np.uint32)
        for k,base in enumerate(keys):
            selected=bucket==k
            if selected.any(): ids[selected]=exact_lookup(base,cells[selected])+index['buckets'][k]['offset']
        retained=rowmap[ids]!=np.uint32(2**32-1)
        p=transform(xyz[retained],tf);origin=transform(poses[frame,None,:3,3],tf)[0]
        delta=p-origin;distance=np.linalg.norm(delta,axis=1)
        rays=np.c_[np.broadcast_to(origin,p.shape),delta/np.maximum(distance[:,None],1e-12)].astype('f4')
        hit=raycaster.cast_rays(o3d.core.Tensor(rays),nthreads=4)
        depth=hit['t_hit'].numpy();faceids=hit['primitive_ids'].numpy()
        valid=np.isfinite(depth)&(np.abs(depth-distance)<=options.get('depth_tolerance_source_units',.1)*scale)&(distance>0)
        pixels=np.flatnonzero(mask)[retained][valid];faceids=faceids[valid]
        pred=np.load(Path(prediction['predictions'])/f'{frame:05d}.npz')
        labels,support=resize_labels_to_source(pred['labels'],pred['support_fraction'],*mask.shape)
        _,margin=resize_labels_to_source(pred['labels'],pred['margin_fraction'],*mask.shape)
        labels=labels.ravel()[pixels];support=support.ravel()[pixels];margin=margin.ravel()[pixels]
        eligible=(support>=options.get('pixel_support',.35))&(margin>=options.get('pixel_margin',.1))&~np.isin(labels,excluded)
        support[~eligible]=0
        vote=frame_votes(faceids,labels,support,options.get('pixel_support',.35))
        counts=np.bincount(faceids[eligible],minlength=nf)
        vote=vote[(vote['weight']>0)&(counts[vote['index']]>=options.get('minimum_pixels',2))]
        records.append(vote);coverage[np.unique(faceids)]+=1
        if frame%20==0: print('FACE_VOTES',frame,len(vote),flush=True)
    ids,win,agreement,mean,obs=fuse_votes(np.concatenate(records) if records else np.empty(0,VOTE_DTYPE),
                                       options.get('agreement',.55),options.get('mean_support',.2),options.get('minimum_frames',2))
    labels=np.full(nf,255,np.uint16);labels[ids]=win
    a=np.zeros(nf,np.float32);a[ids]=agreement;s=np.zeros(nf,np.float32);s[ids]=mean
    observations=np.zeros(nf,np.uint32);observations[ids]=obs
    target=out/'mesh_arrays.npz'
    np.savez_compressed(target,vertices=v,faces=f,rgb=rgb,face_class_id=labels,vote_agreement=a,
                        mean_winning_support=s,observations=observations,geometric_observations=coverage)
    report=dict(faces=nf,labeled_faces=int((labels!=255).sum()),geometry_unchanged=True,
                segmentation_type='semantic classes; connected regions are not verified physical instances',
                score_type='relative OOOPS class support, not calibrated probability')
    exports=export_semantic(out,np.load(target),classes)
    result(out,arrays=str(target),coordinates=mesh['coordinates'],vocabulary=prediction['vocabulary'],
           external_artifacts=external,report=report,**exports)
