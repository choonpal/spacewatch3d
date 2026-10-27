"""Bind Gaussian RGB change pixels to visible input-mesh instance faces.

Gaussian expected depth is an approximate lift. A small 3D distance threshold
must not silently discard an otherwise visible segmented object in that pixel.
"""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .instance_observations import common_points


def link(objects,sides,registration,folder,result,cameras):
    import open3d as o3d
    scenes=[];owners=[];semantics=[];votes={};surfaces={};totals={};view_votes={}
    for epoch,side in enumerate(sides):
        with np.load(side['mesh'],allow_pickle=False) as d:
            p=d['p'] if 'p' in d else d['vertices'];f=d['faces'].astype('u4');semantic=d['face_semantic'] if 'face_semantic' in d else np.full(len(f),255)
        p=common_points(p,side,epoch,registration).astype('f4')
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(p),o3d.core.Tensor(f)))
        owner=np.full(len(f),-1,'i4')
        for index,obj in enumerate(objects):
            if obj['epoch']==epoch:owner[np.asarray(obj['original_face_ids'],int)]=index
        scenes.append(scene);owners.append(owner);semantics.append(semantic)
    with np.load(folder/'changes-3d.npz') as d:points=d['points'];labels=d['candidate_index']
    if not len(points):return {'used':True,'instance_pixel_votes':{},'candidate_surface_counts':{}}
    tree=cKDTree(points);h=registration['height_u']
    for view,(camera,observation) in enumerate(zip(cameras,result['observations'])):
        pose=np.asarray(camera['camera_to_common_cv']);directory=folder/observation['directory']
        for epoch,phase in enumerate(['before','after']):
            mask=np.asarray(Image.open(directory/(phase+'-change.png')))>0
            y,x=np.nonzero(mask)
            if not len(x):continue
            height,width=mask.shape;longitude=((x+.5)/width-.5)*2*np.pi;latitude=((y+.5)/height-.5)*np.pi
            rays=np.c_[np.sin(longitude)*np.cos(latitude),np.sin(latitude),np.cos(longitude)*np.cos(latitude)]@pose[:3,:3].T
            with np.load(directory/(phase+'-geometry.npz')) as d:depth=d['depth'][y,x]
            lifted=pose[:3,3]+rays*depth[:,None];distance,nearest=tree.query(lifted);candidate=labels[nearest];valid=distance<.05*h
            cast=np.c_[np.repeat(pose[None,:3,3],len(rays),axis=0),rays].astype('f4');hit=scenes[epoch].cast_rays(o3d.core.Tensor(cast));face=hit['primitive_ids'].numpy();has_face=face<len(owners[epoch])
            for ci in np.unique(candidate[valid]):
                choose=valid&(candidate==ci);key=str(int(ci));totals[(int(ci),epoch)]=totals.get((int(ci),epoch),0)+int(choose.sum())
                ids=np.flatnonzero(choose&has_face)
                if not len(ids):continue
                values,count=np.unique(semantics[epoch][face[ids]],return_counts=True);bucket=surfaces.setdefault(key,{})
                for value,n in zip(values,count):bucket[str(int(value))]=bucket.get(str(int(value)),0)+int(n)
                own,count=np.unique(owners[epoch][face[ids]],return_counts=True)
                for obj,n in zip(own,count):
                    if obj<0:continue
                    pair=(int(obj),int(ci));votes[pair]=votes.get(pair,0)+int(n);view_votes.setdefault(pair,set()).add(view)
    for obj in objects:obj['proximity_candidate_ids']=obj['candidate_ids'];obj['candidate_ids']=[];obj['link_scores']={};obj['change_pixel_votes']={}
    for (index,ci),n in votes.items():
        obj=objects[index];total=totals.get((ci,obj['epoch']),0);fraction=n/max(total,1);candidate=result['candidates'][ci]['id']
        if n>=6 and fraction>=.15:
            obj['candidate_ids'].append(candidate);obj['link_scores'][candidate]=fraction
            obj['change_pixel_votes'][candidate]={'pixels':n,'phase_changed_pixels':total,'fraction':fraction,'views':sorted(view_votes[(index,ci)])}
    return {'used':True,'method':'common-camera first-hit face of each actual Gaussian-RGB SimSaC changed pixel',
            'instance_pixel_votes':{objects[i]['id']+':'+result['candidates'][c]['id']:n for (i,c),n in votes.items()},
            'candidate_surface_counts':{result['candidates'][int(k)]['id']:v for k,v in surfaces.items()}}
