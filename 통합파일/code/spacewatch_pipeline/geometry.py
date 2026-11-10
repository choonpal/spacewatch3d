"""Point cloud, label-independent meshing, and exact multi-frame mesh voting."""
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
        save(out/'generator.json',cfg)
        source=out/'reconstruction'
        subprocess.run([sys.executable,str(ROOT/'vendor/panovggt_pipeline_v2/run.py'),
                        '--config',str(out/'generator.json'),'--output',str(source)],check=True)
    settings=read(source/'config.json')
    if settings.get('generator_version')!='2.0.0':
        raise ValueError('Integration v1 requires point cloud stable v2.0.0')
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
    src=deps[prefix+'.reconstruct'];source=Path(src['source_run'])
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
        pc=pc.voxel_down_sample(voxel)
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
        rgb=np.rint(np.asarray(pc.colors)[ix]*255).clip(0,255).astype(np.uint8)
        path=out/'geometry.npz';np.savez_compressed(path,vertices=v,faces=f,rgb=rgb)
        external=[]
        trimesh.Trimesh(v,f,vertex_colors=rgb,process=False).export(out/'mesh_rgb.glb')
    if not len(f) or not np.isfinite(v).all() or f.max()>=len(v):
        raise ValueError('Invalid triangle mesh')
    result(out,geometry=str(path),coordinates=src['coordinates'],vertices=len(v),faces=len(f),
           geometry_sha256=sha(path),external_artifacts=external,semantic_labels_used_for_meshing=False)

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

