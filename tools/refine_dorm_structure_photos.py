"""Resolve repeated-corridor alignment from photo features on geometric structure.

Features are admitted only where original-mesh rays hit extracted floor/wall
support. No semantic/instance labels or changed-object associations are read.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial import cKDTree
from dorm_mesh_geometry import transform,plane_samples
from run_dorm_mesh_comparison import ROOTS,SOURCES,load,save,load_frames


def umeyama(source,target):
    a,b=source.mean(0),target.mean(0);x,y=source-a,target-b
    u,s,vt=np.linalg.svd(y.T@x/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    rotation=u@np.diag(sign)@vt;scale=float(np.sum(s*sign)/np.mean(np.sum(x*x,axis=1)))
    matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=b-scale*rotation@a
    return matrix


def run(previous,output,deps,frame_offset=0,check_seed=None):
    sys.path.insert(0,str(deps))
    import cv2
    import open3d as o3d
    metadata=load(previous/'alignment.json');h=metadata['quality']['reference_height_units']
    if output.exists():raise ValueError('New output required')
    output.mkdir(parents=True)
    frames=load_frames(np.eye(4));cache=previous/('structural-photo-features.npz' if frame_offset==0 else f'structural-photo-features-offset{frame_offset}.npz');samples=np.load(previous/'alignment-samples.npz')
    if cache.exists():
        saved=np.load(cache);features=[tuple(saved[f'{name}_{e}'] for name in ('xyz','descriptors','frames')) for e in (0,1)]
    else:
        features=[];samples=np.load(previous/'alignment-samples.npz')
        sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.015)
        for e,root in enumerate(ROOTS):
            mesh=o3d.io.read_triangle_mesh(str(root/'meshes/scene.ply'));scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            groups=[p for p in metadata['planes'][e] if p['kind'] in ('floor','wall')]
            points=samples['before' if e==0 else 'after'];ids=np.concatenate([np.array(p['ids'],int) for p in groups])
            support=cKDTree(points[ids]);xyz=[];descriptors=[];frame_ids=[]
            for frame in frames[e][frame_offset::5]:
                image=cv2.imread(frame['source_path'],cv2.IMREAD_GRAYSCALE);keys,desc=sift.detectAndCompute(image,None)
                if desc is None:continue
                pixels=np.array([key.pt for key in keys]);w,hp=frame['width'],frame['height']
                lon=((pixels[:,0]+.5)/w-.5)*2*np.pi;lat=((pixels[:,1]+.5)/hp-.5)*np.pi
                directions=np.c_[np.sin(lon)*np.cos(lat),np.sin(lat),np.cos(lon)*np.cos(lat)]
                pose=np.array(frame['camera_to_world']);directions=directions@pose[:3,:3].T
                origins=np.tile(pose[:3,3],(len(directions),1));hit=scene.cast_rays(o3d.core.Tensor(np.c_[origins,directions].astype('f4')))
                distance=hit['t_hit'].numpy();valid=np.isfinite(distance)&(distance>h*.1)&(distance<h*8)
                indices=np.flatnonzero(valid);locations=origins[valid]+directions[valid]*distance[valid,None]
                close=support.query(locations)[0]<h*.05
                locations=locations[close];indices=indices[close]
                xyz.append(locations);descriptors.append(desc[indices]);frame_ids.extend([frame['sample_index']]*len(indices))
            features.append((np.vstack(xyz),np.vstack(descriptors).astype('f4'),np.array(frame_ids)))
            print('STRUCTURAL PHOTO FEATURES',e,len(frame_ids),'source frames',len(set(frame_ids)),flush=True)
        np.savez_compressed(cache,**{f'{name}_{e}':array for e,group in enumerate(features) for name,array in zip(('xyz','descriptors','frames'),group)})
    before,desc_before,frames_before=features[0];after,desc_after,frames_after=features[1]
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=8),dict(checks=128))
    nearest=matcher.knnMatch(desc_after,desc_before,k=min(24,len(desc_before)));pairs=[]
    for i,matches in enumerate(nearest):
        if not matches or matches[0].distance>220:continue
        best=matches[0];competitors=[m.distance for m in matches[1:] if np.linalg.norm(before[m.trainIdx]-before[best.trainIdx])>h*.12]
        if competitors and best.distance>=.72*min(competitors):continue
        if not competitors and best.distance>120:continue
        pairs.append((i,best.trainIdx,best.distance))
    pairs.sort(key=lambda pair:pair[2]);unique=set();kept=[]
    for row in pairs:
        key=tuple(np.floor(after[row[0]]/(h*.05)).astype(int))+tuple(np.floor(before[row[1]]/(h*.05)).astype(int))
        if key not in unique:kept.append(row);unique.add(key)
    if len(kept)<30:raise ValueError('Not enough floor/wall photo feature correspondences')
    pairs=np.array(kept);source=after[pairs[:,0].astype(int)];target=before[pairs[:,1].astype(int)]
    print('CANDIDATE STRUCTURAL FEATURE PAIRS',len(pairs),flush=True)
    if len(pairs)<30:raise ValueError('Not enough floor/wall photo feature correspondences')
    np.savez_compressed(output/'structural-photo-correspondences.npz',source=source,target=target,pairs=pairs,
        source_frame=frames_after[pairs[:,0].astype(int)],target_frame=frames_before[pairs[:,1].astype(int)])
    if check_seed:
        hypotheses=[np.array(metadata['after_to_before']),np.array(load(check_seed)['after_to_before'])];checks=[]
        for matrix in hypotheses:
            distance=np.linalg.norm(transform(source,matrix)-target,axis=1);keep=distance<h*.10
            checks.append({'inliers':int(keep.sum()),'median_residual_units':float(np.median(distance[keep])) if keep.any() else None,
                'source_frames':len(np.unique(frames_after[pairs[keep,0].astype(int)])),'target_frames':len(np.unique(frames_before[pairs[keep,1].astype(int)]))})
        record={'correspondences':len(pairs),'frame_offset':frame_offset,'fitted_to_these_frames':False,'checks':checks,
            'acceptance_rule':'At least 15 corroborating pairs across >=3 source and >=3 target frames; >=2x the other hypothesis; geometric overlap >=.25',
            'threshold_units':h*.10}
        save(output/'independent-photo-check.json',record);print('INDEPENDENT PHOTO CHECK',json.dumps(record),flush=True);return
    rng=np.random.default_rng(20261019);best=None;tolerance=h*.10
    for _ in range(30000):
        ids=rng.choice(len(source),3,replace=False)
        if np.linalg.svd(source[ids]-source[ids].mean(0),compute_uv=False)[1]<h*.03:continue
        matrix=umeyama(source[ids],target[ids]);scale=np.cbrt(np.linalg.det(matrix[:3,:3]))
        if not .4<scale<2.5 or matrix[1,1]/scale<.92:continue
        distance=np.linalg.norm(transform(source,matrix)-target,axis=1);inlier=distance<tolerance
        cells=len(np.unique(np.floor(target[inlier]/(h*.2)).astype(int),axis=0))
        score=inlier.sum()+.5*cells
        if best is None or score>best[0]:best=(score,matrix,inlier)
    if best is None:raise ValueError('No supported photo-structure similarity')
    _,matrix,inlier=best
    print('RANSAC CONSENSUS',int(inlier.sum()),'matrix',matrix.tolist(),flush=True)
    for _ in range(5):
        fitted=umeyama(source[inlier],target[inlier]);distance=np.linalg.norm(transform(source,fitted)-target,axis=1)
        updated=distance<tolerance
        if updated.sum()<inlier.sum():break
        matrix,inlier=fitted,updated
    distance=np.linalg.norm(transform(source,matrix)-target,axis=1)
    save(output/'feature-alignment-audit.json',{'candidates':len(pairs),'inliers':int(inlier.sum()),'after_to_before':matrix.tolist(),
        'residual_quantiles':np.quantile(distance,[.1,.5,.9]).tolist(),'inlier_threshold_units':tolerance})
    if inlier.sum()<25:raise ValueError('Photo correspondence consensus is too small')
    a,na=plane_samples(samples['before'],metadata['planes'][0]);b,nb=plane_samples(samples['after'],metadata['planes'][1])
    moved=transform(b,matrix);d,near=cKDTree(a).query(moved);opposite=cKDTree(moved).query(a)[0]
    scale=float(np.cbrt(np.linalg.det(matrix[:3,:3])));rotation=matrix[:3,:3]/scale
    good=(d<h*.05)&(np.abs(np.einsum('ij,ij->i',nb@rotation.T,na[near]))>.9)
    residual=np.abs(np.einsum('ij,ij->i',na[near],moved-a[near]))
    quality={'structural_overlap':float((good.mean()+(opposite<h*.05).mean())/2),
        'normal_residual_median_units':float(np.median(residual[good])),'reference_height_units':h,
        'relative_scale_after_to_before':scale,'reference_points':[len(a),len(b)],
        'structural_photo_feature_inliers':int(inlier.sum()),'structural_photo_feature_candidates':len(source),
        'structural_photo_median_residual_units':float(np.median(distance[inlier])),
        'source_frames_used':[len(np.unique(frames_before[pairs[inlier,1].astype(int)])),len(np.unique(frames_after[pairs[inlier,0].astype(int)]))],
        'method':'Similarity RANSAC on geometric floor/wall photo correspondences; no object/semantic labels'}
    metadata.update(after_to_before=matrix.tolist(),quality=quality,geometry_only_initial_alignment=metadata['after_to_before'])
    save(output/'alignment.json',metadata);shutil.copy2(previous/'alignment-samples.npz',output/'alignment-samples.npz')
    protocol=load(previous/'protocol.json');protocol['structural_photo_refinement']='SIFT on original panoramas, admitted only on rays hitting geometric floor/wall support'
    save(output/'protocol.json',protocol)
    np.savez_compressed(output/'structural-photo-correspondences.npz',source=source,target=target,inliers=inlier)
    print('PHOTO STRUCTURE ALIGNMENT',json.dumps(quality),'matrix',matrix.tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('previous','output','deps'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--frame-offset',type=int,default=0);parser.add_argument('--check-seed',type=Path)
    args=parser.parse_args();run(args.previous,args.output,args.deps,args.frame_offset,args.check_seed)
