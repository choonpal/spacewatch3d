"""Identity descriptors from freshly rendered native Gaussian object views.

Uses the same frozen SimSaC VGG features as the existing mesh SimSaC appearance
stage. Original video RGB is never read by this adapter.
"""
import numpy as np
from pathlib import Path
from tools.mesh_simsac.core import camera_transform, write, sha
from .instance_observations import packet
from .render import Renderer


def extract(objects,sides,height,folder,model,gaussian_models):
    import torch
    import torch.nn.functional as F
    import cv2
    folder=Path(folder);directory=folder/'instance-renders';directory.mkdir(exist_ok=True)
    renderers=[Renderer(g,np.zeros(len(g['xyz']),'i4'),[s['world_from_gaussian']]) for g,s in zip(gaussian_models,sides)]
    poses=[]
    for side in sides:
        source=packet(side)
        poses.append({int(row.get('frame',i)):camera_transform(row['camera_to_world'],side['world_from_mesh']) for i,row in enumerate(source.get('cameras',[]))} if source else {})
    audit={};arrays={};size=384;focal=.5*size/np.tan(np.deg2rad(100)/2)
    for index,obj in enumerate(objects):
        points=obj.get('_native_points')
        if points is None:continue
        center=np.median(points,axis=0);rows=[(frame,poses[obj['epoch']][frame]) for frame in obj.get('source_frames',[]) if frame in poses[obj['epoch']]]
        rows.sort(key=lambda row:np.linalg.norm(row[1][:3,3]-center));selected=[]
        for frame,pose in rows:
            if any(np.linalg.norm(pose[:3,3]-p[:3,3])<.1*height for _,p in selected):continue
            selected.append((frame,pose))
            if len(selected)>=3:break
        features=[];records=[]
        for frame,pose in selected:
            camera=pose.copy();forward=center-camera[:3,3];forward/=max(np.linalg.norm(forward),1e-8)
            right=np.cross(np.array([0.,-1.,0.]),forward)
            if np.linalg.norm(right)<.1:right=np.cross(np.array([1.,0.,0.]),forward)
            right/=max(np.linalg.norm(right),1e-8);down=np.cross(forward,right);camera[:3,:3]=np.column_stack([right,down,forward])
            image,depth,alpha=renderers[obj['epoch']].face(camera,size)
            q=(points-camera[:3,3])@camera[:3,:3];z=q[:,2]
            uv=focal*q[:,:2]/np.maximum(z[:,None],1e-8)+size/2-.5
            u,v=np.floor(uv+.5).astype(int).T
            valid=(z>.01*height)&(u>=0)&(u<size)&(v>=0)&(v<size)
            ids=np.flatnonzero(valid);u,v=u[ids],v[ids];seen=(alpha[v,u]>=.85)&(abs(depth[v,u]-z[ids])<.08*height)
            mask=np.zeros((size,size),'u1');mask[v[seen],u[seen]]=1
            mask=cv2.dilate(mask,np.ones((7,7),'u1'))&(alpha>=.85).astype('u1')
            if mask.sum()<24:continue
            full,_=model.prep(image)
            with torch.inference_mode():
                pyramid=model.model.pyramid(full,eigth_resolution=True)[-2:];parts=[]
                for feature in pyramid:
                    weight=F.interpolate(torch.from_numpy(mask.astype('f4'))[None,None].to(model.device),size=feature.shape[-2:],mode='area')
                    pooled=(feature*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1e-8)
                    parts.append(F.normalize(pooled,dim=1)[0].cpu().numpy())
                vector=np.concatenate(parts);vector/=max(np.linalg.norm(vector),1e-8)
            name=obj['id']+f'-F{frame:05d}';path=directory/(name+'.png');cv2.imwrite(str(path),image[...,::-1]);cv2.imwrite(str(directory/(name+'-mask.png')),mask*255)
            features.append(vector);records.append({'source_frame':frame,'image':path.relative_to(folder).as_posix(),'image_sha256':sha(path),'mask_pixels':int(mask.sum()),'camera_to_native_world_cv':camera.tolist()})
        if features:
            obj['_embeddings']=np.stack(features).astype('f4');arrays[obj['id']]=obj['_embeddings'];obj['appearance_source']='fresh_native_gaussian_simsac_vgg';obj['appearance_source_frames']=[r['source_frame'] for r in records]
        audit[obj['id']]=records
        print('INSTANCE_APPEARANCE',index+1,len(objects),obj['id'],len(features),flush=True)
    np.savez_compressed(folder/'instance-appearance.npz',**arrays)
    model_audit={**model.audit,'input_source':'Fresh native Gaussian RGB renders at segmentation-source cameras','gaussian_inputs_used':True}
    output={'model':model_audit,'objects':audit,'descriptor_file':'instance-appearance.npz','descriptor_sha256':sha(folder/'instance-appearance.npz'),
            'input':'fresh native Gaussian RGB at real segmentation-source camera positions','pooling':'same frozen SimSaC VGG quarter/eighth mask means','original_video_rgb_read':False}
    write(folder/'instance-appearance.json',output)
    return output


def similarity(a,b):
    if a is None or b is None:return 0.
    scores=np.clip(a@b.T,-1,1)
    return float((np.median(scores.max(1))+np.median(scores.max(0)))/2)
