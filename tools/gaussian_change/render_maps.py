"""Export real SH3 Gaussians and render both frozen-aligned maps on the GPU.

Only derived files are written. Means, covariance rotations and scales follow the
existing mesh transforms; view-dependent SH is evaluated in each original frame.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from gsplat import rasterization,spherical_harmonics
from prepare import FrozenPartition,transform,membership

ROOT=Path('/workspace/gaussian_change_20261025')
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def quaternion_product(a,b):
    w=a[...,0]*b[...,0]-(a[...,1:]*b[...,1:]).sum(-1)
    v=a[...,0,None]*b[...,1:]+b[...,0,None]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([w[...,None],v],-1)
def write_ply(path,params,mask):
    fields=['x','y','z']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    values=np.concatenate([params['means'][mask],params['features_dc'][mask],params['features_rest'][mask].transpose(0,2,1).reshape(-1,45),params['opacities'][mask],params['scales'][mask],params['quats'][mask]],1).astype('<f4')
    assert values.shape[1]==len(fields) and np.isfinite(values).all()
    header='ply\nformat binary_little_endian 1.0\ncomment Original trained SH3 Gaussian parameters; no point decimation\nelement vertex '+str(len(values))+'\n'+''.join('property float '+x+'\n' for x in fields)+'end_header\n'
    with path.open('wb') as f:f.write(header.encode());f.write(values.tobytes())
    return dict(file='models/'+path.name,gaussians=len(values),bytes=path.stat().st_size,sha256=sha(path))

def main():
    start=time.time();torch.set_num_threads(4)
    protocol=read(ROOT/'protocol.json');out=ROOT/'published';out.mkdir(exist_ok=True)
    for name in ['models','renders']:(out/name).mkdir(exist_ok=True)
    classifier=FrozenPartition({k:v for k,v in np.load(ROOT/'partition.npz').items()})
    source=Path(protocol['model_root'])/'vendor/dorm1_3dgs_integration/src/render_360_keyframes.py'
    spec=importlib.util.spec_from_file_location('linked_cube_renderer',source);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    W,H=protocol['decisions']['render_size'];F=protocol['decisions']['cube_face_size'];fov=protocol['decisions']['cube_fov_degrees'];focal=.5*F/np.tan(np.deg2rad(fov)/2)
    yy,xx=np.mgrid[:H,:W];lo=((xx+.5)/W-.5)*2*np.pi;la=((yy+.5)/H-.5)*np.pi
    rays=np.stack([np.sin(lo)*np.cos(la),np.sin(la),np.cos(lo)*np.cos(la)],-1)
    bases={k:renderer.face_basis(*v) for k,v in renderer.FACE_SPECS.items()}
    maps={}
    for name,basis in bases.items():
        d=rays@basis;z=np.maximum(d[...,2],1e-8)
        maps[name]=(d,(focal*d[...,0]/z+F/2-.5).astype('f4'),(focal*d[...,1]/z+F/2-.5).astype('f4'))
    dominant=np.argmax(np.stack([maps[k][0][...,2] for k in bases],-1),-1)
    views=protocol.get('views',protocol['zones'])
    dataset=dict(schema='registered-gaussian-pair-renders-v1',scenes=protocol['scenes'],models={'before':[],'after':[]},zones=views,registration_zones=protocol['zones'],baseline_boxes=protocol['baseline_boxes'],height_u=protocol['height_u'],camera_checks=protocol['camera_checks'],decisions=protocol['decisions'],renders={},source_hashes=protocol['source_hashes'],region_membership_counts={})
    if 'keyframe_sources' in protocol:
        dataset.update(keyframe_sources=protocol['keyframe_sources'],viewpoint_mode='tour_keyframes',keyframe_camera_checks=protocol['keyframe_camera_checks'])
    for epoch,scene in enumerate(protocol['scenes']):
        name=scene['capture'];phase=['before','after'][epoch]
        save(ROOT/'render-status.json',dict(state='loading',capture=name))
        assert sha(scene['checkpoint'])==scene['checkpoint_sha256']
        ckpt=torch.load(scene['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        params={k:ckpt['pipeline']['_model.gauss_params.'+k].detach().numpy().copy() for k in ['means','features_dc','features_rest','opacities','quats','scales']}
        del ckpt;gc.collect();N=len(params['means']);assert N==scene['gaussians']
        native_to_mesh=np.array(scene['native_to_mesh']);native_old=transform(params['means'],native_to_mesh)
        labels=membership(protocol,native_old,epoch,classifier)
        dataset['region_membership_counts'][phase]={f'R{i:02d}':int((labels==i).sum()) for i in range(1,14)}
        world=np.zeros_like(params['means']);qworld=np.zeros_like(params['quats']);sworld=np.zeros_like(params['scales']);rotations=np.zeros((N,3,3),dtype='f4')
        for zone in protocol['zones']:
            mask=(labels==int(zone['id'][1:])) if epoch else np.ones(N,bool)
            if not mask.any():continue
            m=np.array(zone['native_to_common'][epoch]);scale=np.cbrt(np.linalg.det(m[:3,:3]));rotation=m[:3,:3]/scale
            quaternion=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            world[mask]=transform(params['means'][mask],m)
            qworld[mask]=quaternion_product(quaternion,params['quats'][mask]);qworld[mask]/=np.linalg.norm(qworld[mask],axis=1,keepdims=True)
            sworld[mask]=params['scales'][mask]+np.log(scale);rotations[mask]=rotation
            asset=write_ply(out/'models'/('after-'+zone['id']+'.ply' if epoch else 'before.ply'),params,mask)
            asset.update(matrix=m.tolist(),zone=zone['id'] if epoch else None,bounds=np.stack([world[mask].min(0),world[mask].max(0)]).tolist(),sh_degree=3,source_epoch=epoch)
            dataset['models'][phase].append(asset)
            if not epoch:break
        assert sum(x['gaussians'] for x in dataset['models'][phase])==N
        means=torch.from_numpy(world).cuda();quats=torch.from_numpy(qworld).cuda();scales=torch.from_numpy(np.exp(sworld)).cuda()
        opacities=torch.from_numpy(1/(1+np.exp(-params['opacities'].reshape(-1)))).cuda()
        coeff=torch.from_numpy(np.concatenate([params['features_dc'][:,None,:],params['features_rest']],1)).cuda();rotations_gpu=torch.from_numpy(rotations).cuda()
        for number,zone in enumerate(views):
            directory=out/'renders'/zone['id'];directory.mkdir(exist_ok=True)
            pose=np.array(zone['camera_to_common_cv']);center=torch.from_numpy(pose[:3,3].astype('f4')).cuda()
            # Evaluate original directional SH in each Gaussian's native frame.
            directions=means-center
            directions=torch.einsum('ni,nij->nj',directions,rotations_gpu)
            directions=directions/torch.linalg.norm(directions,dim=1,keepdim=True).clamp_min(1e-8)
            with torch.inference_mode():colors=torch.clamp(spherical_harmonics(3,directions,coeff)+.5,min=0)
            rgb_faces={};aux_faces={}
            for face,basis in bases.items():
                camera=pose.copy();camera[:3,:3]=pose[:3,:3]@basis
                view=torch.from_numpy(np.linalg.inv(camera).astype('f4'))[None].cuda()
                K=torch.tensor([[[focal,0,F/2],[0,focal,F/2],[0,0,1]]],dtype=torch.float32,device='cuda')
                with torch.inference_mode():
                    rendered,alpha,_=rasterization(means,quats,scales,opacities,colors,view,K,F,F,near_plane=.01,far_plane=500.,packed=True,render_mode='RGB+ED',rasterize_mode='classic',backgrounds=torch.zeros((1,3),device='cuda'))
                rgb=np.clip(rendered[0,...,:3].cpu().numpy()*255+.5,0,255).astype('u1');depth=rendered[0,...,3].cpu().numpy();a=alpha[0,...,0].cpu().numpy()
                rgb_faces[face]=rgb;aux_faces[face]=(depth,a)
                cv2.imwrite(str(directory/(phase+'-'+face+'.jpg')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
            pano=renderer.stitch_equirectangular(rgb_faces,bases,W,fov)
            depth=np.zeros((H,W),'f4');alpha=np.zeros((H,W),'f4')
            for i,(face,(dep,a)) in enumerate(aux_faces.items()):
                d,mx,my=maps[face];choose=dominant==i
                z=cv2.remap(dep,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                av=cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                depth[choose]=z[choose]/np.maximum(d[...,2][choose],1e-8);alpha[choose]=av[choose]
            image=directory/(phase+'.jpg');aux=directory/(phase+'-geometry.npz')
            cv2.imwrite(str(image),cv2.cvtColor(pano,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);np.savez_compressed(aux,depth=depth,alpha=alpha)
            dataset['renders'].setdefault(zone['id'],{})[phase]=dict(image=str(image.relative_to(out)),geometry=str(aux.relative_to(out)),image_sha256=sha(image),geometry_sha256=sha(aux),size=[W,H],camera_to_common_cv=pose.tolist(),checkpoint_sha256=scene['checkpoint_sha256'],same_registered_world_pose=True,reference_sharpening=False)
            save(ROOT/'render-status.json',dict(state='rendering',capture=name,zone=zone['id'],completed=number+1,total=len(views)));print('RENDER',phase,zone['id'],flush=True)
        assert sha(scene['checkpoint'])==scene['checkpoint_sha256']
        del means,quats,scales,opacities,coeff,rotations_gpu,params;gc.collect();torch.cuda.empty_cache()
    save(out/'gaussian-data.json',dataset)
    save(ROOT/'render-status.json',dict(state='complete',seconds=time.time()-start,views=2*len(views),source_gaussians=[s['gaussians'] for s in protocol['scenes']],all_gaussians_exported=True,original_models_unchanged=True))
    print('RENDER COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);ROOT=parser.parse_args().root
    try:main()
    except Exception:
        import traceback
        save(ROOT/'render-status.json',dict(state='error',traceback=traceback.format_exc()));raise
