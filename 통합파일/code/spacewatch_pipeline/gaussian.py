"""Masked 3D Gaussian training and novel-view ERP rendering with gsplat."""
import copy
import math
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gsplat import rasterization,DefaultStrategy,export_splats
from gsplat.strategy.ops import reset_opa
from .io import read,save,result
from .geometry import context
from .gaussian_data import rotations
from .gaussian_inputs import CudaViewLoader

def ssim(x,y,mask):
    x=x.permute(2,0,1)[None];y=y.permute(2,0,1)[None];m=mask[None,None].float()
    moments=F.avg_pool2d(torch.cat([x,y,x*x,y*y,x*y],1),7,1)
    a,b,aa,bb,ab=moments.chunk(5,1)
    score=((2*a*b+.01**2)*(2*(ab-a*b)+.03**2))/((a*a+b*b+.01**2)*((aa-a*a).clamp_min(0)+(bb-b*b).clamp_min(0)+.03**2))
    valid=(F.avg_pool2d(m,7,1)>.999).float()
    return (score*valid).sum()/(valid.sum()*3).clamp_min(1)

def draw(splats,pose,K,size,degree,viewmat=None,background=None):
    colors,alpha,info=rasterization(means=splats['means'],quats=splats['quats'],scales=splats['scales'].exp(),
        opacities=splats['opacities'].sigmoid(),colors=torch.cat([splats['sh0'],splats['shN']],1),
        viewmats=(torch.linalg.inv(pose) if viewmat is None else viewmat)[None],Ks=K[None],width=size,height=size,sh_degree=degree,
        packed=False,near_plane=.15,far_plane=200.,rasterize_mode='classic',
        backgrounds=torch.zeros((1,3),device='cuda') if background is None else background)
    return colors[0],alpha[0],info

def gs_train(job):
    c,out,deps,prefix=context(job);cfg=job['config'].get('gaussian',{})
    dataset=Path(deps[prefix+'.gs_prepare']['dataset']);meta=read(dataset/'views.json')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one GPU is required')
    views=meta['frames'];train=[i for i,v in enumerate(views) if v['split']=='train'];val=[i for i,v in enumerate(views) if v['split']=='val']
    if not train or not val:raise ValueError('Missing disjoint train/validation views')
    size=meta['size'];K=torch.tensor(meta['K'],device='cuda',dtype=torch.float32)
    poses=torch.tensor([v['camera_to_world'] for v in views],device='cuda',dtype=torch.float32)
    viewmats=torch.linalg.inv(poses);background=torch.zeros((1,3),device='cuda')
    loader=CudaViewLoader(dataset,views,size,int(cfg.get('decoded_cache_mib',2048))*1024**2)
    with np.load(dataset/'initial.npz') as z:
        xyz=torch.tensor(z['xyz'],device='cuda');rgb=torch.tensor(z['rgb'],device='cuda');scales=torch.tensor(z['scales'],device='cuda')
    n=len(xyz);steps=cfg.get('steps',30000)
    splats=nn.ParameterDict(dict(means=nn.Parameter(xyz),scales=nn.Parameter(scales.log()[:,None].repeat(1,3)),
        quats=nn.Parameter(F.normalize(torch.randn(n,4,device='cuda'),dim=-1)),
        opacities=nn.Parameter(torch.full((n,),math.log(.1/.9),device='cuda')),
        sh0=nn.Parameter(((rgb-.5)/.28209479177387814)[:,None]),shN=nn.Parameter(torch.zeros(n,15,3,device='cuda'))))
    rates=dict(means=4.8e-4,scales=.0025,quats=.001,opacities=.025,sh0=.0025,shN=.0025/20)
    optimizers={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in splats.items()}
    strategy=DefaultStrategy(refine_start_iter=500,refine_stop_iter=min(15000,max(501,steps//2)),refine_every=100,
                             reset_every=3000,pause_refine_after_reset=len(train),verbose=False)
    strategy_state=strategy.initialize_state(scene_scale=3.)
    rng=np.random.default_rng(42);start_step=0;best=-math.inf
    previous=sorted(out.parent.glob('attempt-*/last.pt'),key=lambda p:p.stat().st_mtime)
    if previous:
        ck=torch.load(previous[-1],map_location='cuda',weights_only=False)
        if ck['dataset']!=str(dataset) or ck['steps']!=steps:raise ValueError('Training checkpoint inputs changed')
        splats=nn.ParameterDict({k:nn.Parameter(v) for k,v in ck['splats'].items()})
        optimizers={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in splats.items()}
        for k,opt in optimizers.items():opt.load_state_dict(ck['optimizers'][k])
        strategy_state=ck['strategy_state'];start_step=ck['step'];best=ck['best_psnr']
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        if previous[-1].with_name('best.pt').exists():
            import shutil
            shutil.copyfile(previous[-1].with_name('best.pt'),out/'best.pt')
    strategy.check_sanity(splats,optimizers)
    started=time.monotonic();history=[]
    def checkpoint(step,name):
        temporary=out/(name+'.partial')
        torch.save(dict(step=step,steps=steps,splats={k:v.detach() for k,v in splats.items()},
            optimizers={k:o.state_dict() for k,o in optimizers.items()},strategy_state=strategy_state,
            best_psnr=best,dataset=str(dataset),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),frame_id=prefix+'_mesh'),temporary)
        temporary.replace(out/(name+'.pt'))
    @torch.no_grad()
    def evaluate(step):
        rows=[]
        for index in val:
            target,valid=loader.get(index)
            rendered,alpha,_=draw(splats,poses[index],K,size,min(step//1000,3),viewmats[index],background);rendered=rendered.clamp(0,1)
            mse=((rendered-target)**2)[valid].mean();psnr=-10*torch.log10(mse.clamp_min(1e-12))
            rows.append(dict(index=index,psnr=float(psnr),ssim=float(ssim(rendered,target,valid)),alpha=float(alpha[...,0][valid].mean())))
        report=dict(step=step,views=len(rows),mean_psnr=float(np.mean([r['psnr'] for r in rows])),
                    mean_ssim=float(np.mean([r['ssim'] for r in rows])),per_view=rows)
        save(out/f'validation_{step:06d}.json',report)
        return report
    for step in range(start_step,steps):
        index=int(rng.choice(train));target,valid=loader.get(index)
        if step+1<steps:
            # Look ahead without advancing the checkpointed training RNG.
            lookahead=copy.deepcopy(rng)
            loader.prefetch(int(lookahead.choice(train)))
        rendered,alpha,info=draw(splats,poses[index],K,size,min(step//1000,3),viewmats[index],background)
        strategy.step_pre_backward(splats,optimizers,strategy_state,step,info)
        loss=.8*(rendered-target).abs()[valid].mean()+.2*(1-ssim(rendered,target,valid))
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite 3DGS loss at step {step}')
        loss.backward();optimizers['means'].param_groups[0]['lr']=rates['means']*(.01**(step/steps))
        for opt in optimizers.values():opt.step();opt.zero_grad(set_to_none=True)
        if len(splats['means'])>=cfg.get('growth_cap',750000):strategy.grow_grad2d=math.inf
        strategy.step_post_backward(splats,optimizers,strategy_state,step,info,packed=False)
        if step>0 and step<strategy.refine_stop_iter and step%3000==0:reset_opa(splats,optimizers,strategy_state,value=.01)
        with torch.no_grad():splats['scales'].clamp_(-9,math.log(.15));splats['opacities'].clamp_(-12,12)
        loader.stage_pending()
        done=step+1
        if done%100==0 or done==steps:
            row=dict(step=done,steps=steps,gaussians=len(splats['means']),loss=float(loss.detach()),seconds=time.monotonic()-started)
            save(out/'progress.json',row);print('GS_TRAIN',row,flush=True)
        evaluating=done%cfg.get('evaluate_every',3000)==0 or done==steps
        if not evaluating and done%cfg.get('checkpoint_every',1000)==0:checkpoint(done,'last')
        if evaluating:
            report=evaluate(done);history.append(report)
            if report['mean_psnr']>best:best=report['mean_psnr'];checkpoint(done,'best')
            checkpoint(done,'last')
    # A worker can stop after the final atomic last.pt write but before its
    # first validation/best.pt. Resume must finish that pending validation.
    if start_step==steps or not (out/'best.pt').exists():
        report=evaluate(steps)
        if report['mean_psnr']>best or not (out/'best.pt').exists():
            best=report['mean_psnr'];checkpoint(steps,'best')
        checkpoint(steps,'last')
    best_ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    export_splats(**best_ck['splats'],format='ply',save_to=str(out/'gaussians.ply'))
    loader.close()
    result(out,checkpoint=str(out/'best.pt'),gaussians=str(out/'gaussians.ply'),steps=steps,
           selected_step=best_ck['step'],validation_psnr=best,training_seconds=time.monotonic()-started,
           frame_id=prefix+'_mesh',fixed_reconstruction_camera_poses=True,
           input_cache=loader.statistics(),
           quality_validated=False,profile='smoke' if steps<3000 else 'training')

def stitch(faces,alphas,width,height,size,fov=100.):
    yy,xx=np.mgrid[:height,:width]
    lon=((xx+.5)/width-.5)*2*np.pi;lat=((yy+.5)/height-.5)*np.pi
    rays=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
    focal=size/(2*np.tan(np.deg2rad(fov)/2));best=np.full((height,width),-np.inf)
    image=np.zeros((height,width,3),np.float32);alpha=np.zeros((height,width),np.float32)
    for name,r in rotations():
        q=rays@r;z=q[...,2];eligible=z>best
        mx=(q[...,0]/np.maximum(z,1e-8)*focal+size/2-.5).astype('f4')
        my=(q[...,1]/np.maximum(z,1e-8)*focal+size/2-.5).astype('f4')
        eligible &= (z>0)&(mx>=0)&(mx<=size-1)&(my>=0)&(my<=size-1)
        rgb=cv2.remap(faces[name],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        a=cv2.remap(alphas[name],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        image[eligible]=rgb[eligible];alpha[eligible]=a[eligible];best[eligible]=z[eligible]
    if not np.isfinite(best).all():raise ValueError('Cube-to-ERP stitching left uncovered directions')
    return image,alpha

def render(job):
    c,out,deps,prefix=context(job)
    training=deps[prefix+'.gs_train'];planning=deps[prefix+'.plan'];dataset=Path(deps[prefix+'.gs_prepare']['dataset'])
    request=read(planning['requests']);ck=torch.load(training['checkpoint'],map_location='cuda',weights_only=False)
    splats=ck['splats'];poses=np.load(dataset/'panorama_poses.npy');records=[]
    torch.set_num_threads(4)
    for view in request['requests']:
        width,height=view['width'],view['height'];size=max(128,int(np.ceil(width/4*1.2)))
        focal=size/(2*np.tan(np.deg2rad(100)/2));K=torch.tensor([[focal,0,size/2],[0,focal,size/2],[0,0,1]],device='cuda',dtype=torch.float32)
        pose=np.array(view['reference_from_camera']);faces={};alphas={}
        if view['reference_frame_id']!=ck['frame_id']:
            raise ValueError('3DGS/request coordinate-frame mismatch')
        with torch.no_grad():
            for name,r in rotations():
                camera=pose.copy();camera[:3,:3]=pose[:3,:3]@r
                rgb,alpha,_=draw(splats,torch.tensor(camera,device='cuda',dtype=torch.float32),K,size,min(ck['step']//1000,3))
                faces[name]=rgb.clamp(0,1).cpu().numpy();alphas[name]=alpha[...,0].cpu().numpy()
        image,alpha=stitch(faces,alphas,width,height,size)
        name=view['viewpoint_id'];target=out/(name+'.jpg')
        cv2.imwrite(str(target),cv2.cvtColor(np.rint(image*255).astype('u1'),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(out/(name+'_alpha.png')),np.rint(alpha.clip(0,1)*65535).astype('u2'))
        records.append(dict(viewpoint_id=name,reference_from_camera=pose.tolist(),image_ref=str(target),
                            render_status='succeeded',quality_accepted=None,media_origin='3DGS_novel_view',
                            alpha_coverage=float((alpha>.5).mean()),
                            nearest_training_camera_distance_m=float(np.linalg.norm(poses[:,:3,3]-pose[:3,3],axis=1).min())))
        print('GS_RENDER',name,records[-1]['alpha_coverage'],flush=True)
    save(out/'renders.json',dict(schema_name='spacewatch_pipeline.3dgs_renders',capture_id=prefix,
         frame_id=ck['frame_id'],results=records,planning_status=planning['planning_status'],
         quality_accepted=None,method='gsplat SH Gaussian rasterization; six 100-degree pinhole views stitched to ERP',
         training=training,limitations=['Novel views outside training coverage may blur or contain holes',
                                       'Alpha coverage is not an image correctness score']))
    result(out,renders=str(out/'renders.json'),images=len(records),method='3DGS',quality_validated=False)
