"""Calibrated ERP -> six overlapping pinhole views, with whole-time holdout."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from .io import read,save,result
from .geometry import context,transform,similarity

def rotations():
    values=[]
    for name,z,y in [('front',[0,0,1],[0,1,0]),('right',[1,0,0],[0,1,0]),
                     ('back',[0,0,-1],[0,1,0]),('left',[-1,0,0],[0,1,0]),
                     ('up',[0,-1,0],[0,0,1]),('down',[0,1,0],[0,0,-1])]:
        z,y=np.array(z,float),np.array(y,float)
        values.append((name,np.stack([np.cross(y,z),y,z],1)))
    return values

def pinhole_maps(size,width,height,fov=100.):
    focal=size/(2*np.tan(np.deg2rad(fov)/2));yy,xx=np.mgrid[:size,:size]
    rays=np.stack([(xx+.5-size/2)/focal,(yy+.5-size/2)/focal,np.ones_like(xx)],-1)
    maps={}
    for name,r in rotations():
        ray=rays@r.T;ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
        u=(np.arctan2(ray[...,0],ray[...,2])/(2*np.pi)+.5)*width-.5
        v=(np.arcsin(np.clip(ray[...,1],-1,1))/np.pi+.5)*height-.5
        maps[name]=(u.astype('f4'),np.clip(v,0,height-1).astype('f4'))
    return maps,np.array([[focal,0,size/2],[0,focal,size/2],[0,0,1.]])

def gs_prepare(job):
    import cv2
    c,out,deps,prefix=context(job);cfg=job['config'].get('gaussian',{})
    src=deps[prefix+'.reconstruct'];root=Path(src['source_run'])
    manifest=read(root/'input/manifest.json');frames=manifest['frames']
    raw=np.array([r['camera_to_world'] for r in read(root/'result/camera_poses.json')])
    tf,scale=similarity(read(src['coordinates'])['source_to_mesh'])
    poses=raw.copy();poses[:,:3,:3]=tf[:3,:3]/scale@raw[:,:3,:3];poses[:,:3,3]=transform(raw[:,:3,3],tf)
    size=cfg.get('view_size',512);source_video=root/'source/video.mp4'
    use_original=cfg.get('original_video',True) and source_video.exists()
    cap=cv2.VideoCapture(str(source_video)) if use_original else None
    if cap is not None and not cap.isOpened():raise ValueError('Cannot decode original video')
    selected=list(range(0,len(frames),cfg.get('frame_stride',1)))
    if len(selected)<3:raise ValueError('3DGS requires at least three panorama times for train/holdout')
    # Keep all six directions of a timestamp in the same split.
    holdout=set(selected[8::16]) or {selected[len(selected)//2]}
    views=[];maps=None;last_shape=None
    (out/'images').mkdir();(out/'masks').mkdir()
    for i in selected:
        row=frames[i]
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_index']);ok,image=cap.read()
            if not ok:raise ValueError(f'Video decode failed at frame {i}')
        else:
            image=cv2.imread(str(root/'input'/row['file']))
        h,w=image.shape[:2]
        if last_shape!=(h,w):maps,K=pinhole_maps(size,w,h);last_shape=(h,w)
        bad=cv2.imread(str(root/'masks'/(Path(row['file']).stem+'.png')),0)
        if bad is None:raise ValueError('Missing reconstruction exclusion mask')
        bad=cv2.resize(bad,(w,h),interpolation=cv2.INTER_NEAREST)
        bad[:int(h*.03)]=255
        for name,r in rotations():
            mx,my=maps[name]
            im=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
            valid=(cv2.remap(bad,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_WRAP)==0).astype('u1')*255
            valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
            if (valid>0).mean()<.1:continue
            stem=f'{i:05d}_{name}'
            cv2.imwrite(str(out/'images'/f'{stem}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(out/'masks'/f'{stem}.png'),valid)
            pose=poses[i].copy();pose[:3,:3]=pose[:3,:3]@r
            views.append(dict(image=f'images/{stem}.jpg',mask=f'masks/{stem}.png',panorama_index=i,
                              split='val' if i in holdout else 'train',camera_to_world=pose.tolist()))
        if i%20==0:print('GS_PREPARE',i,len(frames),flush=True)
    if cap is not None:cap.release()
    save(out/'views.json',dict(size=size,K=K.tolist(),frames=views,frame_id=prefix+'_mesh',
                              pose_convention='camera X right, Y down, Z forward; world metric +Z up'))
    cloud=PlyData.read(src['point_cloud'],mmap='r')['vertex'].data
    rng=np.random.default_rng(42);ids=np.sort(rng.choice(len(cloud),min(len(cloud),1500000),replace=False))
    xyz=transform(np.c_[cloud['x'][ids],cloud['y'][ids],cloud['z'][ids]],tf)
    rgb=np.c_[cloud['red'][ids],cloud['green'][ids],cloud['blue'][ids]].astype('f4')/255
    _,keep=np.unique(np.floor(xyz/.035).astype('i8'),axis=0,return_index=True)
    if len(keep)>cfg.get('initial_gaussians',180000):keep=rng.choice(keep,cfg.get('initial_gaussians',180000),replace=False)
    xyz=xyz[keep].astype('f4');rgb=rgb[keep]
    if len(xyz)<4:raise ValueError('Too few initialization points')
    d,_=cKDTree(xyz).query(xyz,k=4,workers=4)
    scales=np.sqrt((d[:,1:]**2).mean(1)).clip(.005,.15).astype('f4')
    np.savez_compressed(out/'initial.npz',xyz=xyz,rgb=rgb,scales=scales)
    np.save(out/'panorama_poses.npy',poses)
    result(out,dataset=str(out),views=len(views),train_views=sum(r['split']=='train' for r in views),
           validation_views=sum(r['split']=='val' for r in views),original_resolution=use_original,
           initial_gaussians=len(xyz),holdout_limit='Photometric holdout shares reconstruction poses and initialization',
           external_artifacts=[str(source_video.resolve())] if use_original else [])
