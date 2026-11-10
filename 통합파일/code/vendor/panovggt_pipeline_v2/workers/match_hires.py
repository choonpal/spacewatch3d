"""Use original RGB only for weak correspondence bridges; PanoVGGT stays 1036x518."""
from pathlib import Path
import os,json,time,functools,subprocess
from runtime import R,C,gpu_guard,replay_files
os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']='2'
import numpy as np,cv2,torch
from kornia.feature import LoFTR
from infer import save
M=json.loads((R/'input/manifest.json').read_text());S=C['matching']['high_resolution']['face_size'];H,W=None,None
cv2.setNumThreads(2);torch.set_num_threads(2)
def erp(xy,yaw):
    x=(xy[...,0]+.5-S/2)/(S/2);y=(xy[...,1]+.5-S/2)/(S/2)
    lon=np.arctan2(x,np.ones_like(x))+yaw;lat=np.arctan2(y,np.sqrt(x*x+1))
    return np.stack((((lon/(2*np.pi)+.5)*W-.5)%W,(lat/np.pi+.5)*H-.5),-1)
@functools.lru_cache(maxsize=12)
def views(f):
    im=cv2.imread(str(R/'registration_rgb4k'/f'{f:03d}.png'),0)
    name=Path(M['frames'][f]['file']).stem+'.png';mask=cv2.imread(str(R/'masks'/name),0)
    yy,xx=np.mgrid[:S,:S];xy=np.stack((xx,yy),-1);out=[]
    for face in range(4):
        uv=erp(xy,face*np.pi/2).astype(np.float32)
        mu=(uv+.5)*np.array([C['width']/W,C['height']/H],np.float32)-.5
        out.append((cv2.remap(im,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP),
                    cv2.remap(mask,mu[...,0],mu[...,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_WRAP)==0))
    return out
def main():
    global H,W
    started=time.perf_counter();pairs=json.loads((R/'hires_pairs.json').read_text());folder=R/'hires_matches';folder.mkdir(exist_ok=True)
    if not pairs or replay_files('hires_matches',[f'{a:03d}_{b:03d}.npz' for a,b in pairs]):
        save(R/'hires_matching_summary.json',dict(status='complete',pairs=pairs,mode='verified_cache_replay_or_empty',gpu_used=False));return
    gpu_guard()
    rgb=R/'registration_rgb4k';rgb.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(R/'source/video.mp4'));W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W!=2*H or H<=0:raise ValueError('Original video must be 2:1 ERP')
    for f in sorted(set(sum(pairs,[]))):
        p=rgb/f'{f:03d}.png'
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES,M['frames'][f]['source_frame_index']);ok,im=cap.read();assert ok and im.shape[:2]==(H,W)
            assert cv2.imwrite(str(p),im)
    cap.release()
    torch.manual_seed(C['seed'])
    model=LoFTR(pretrained=C['matching']['model']).eval().cuda();gpu=0
    for k,(a,b) in enumerate(pairs):
        dest=folder/f'{a:03d}_{b:03d}.npz'
        if dest.exists():raise RuntimeError('Unreceipted high-resolution output: '+str(dest))
        va,vb=views(a),views(b);xa=[];xb=[];conf=[]
        for fa in range(4):
            for fb in range(4):
                ia,ma=va[fa];ib,mb=vb[fb]
                x=torch.from_numpy(ia.copy()).float()[None,None].cuda()/255;y=torch.from_numpy(ib.copy()).float()[None,None].cuda()/255
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode():r=model({'image0':x,'image1':y})
                torch.cuda.synchronize();gpu+=time.perf_counter()-t
                pa=r['keypoints0'].cpu().numpy();pb=r['keypoints1'].cpu().numpy();c=r['confidence'].cpu().numpy()
                ai=np.rint(pa).astype(int).clip(0,S-1);bi=np.rint(pb).astype(int).clip(0,S-1)
                keep=(c>=C['matching']['confidence'])&ma[ai[:,1],ai[:,0]]&mb[bi[:,1],bi[:,0]]
                keep&=(pa.min(1)>C['matching']['border_pixels'])&(pa.max(1)<S-C['matching']['border_pixels']-1)&(pb.min(1)>C['matching']['border_pixels'])&(pb.max(1)<S-C['matching']['border_pixels']-1)
                xa.append((erp(pa[keep],fa*np.pi/2)+.5)*[C['width']/W,C['height']/H]-.5)
                xb.append((erp(pb[keep],fb*np.pi/2)+.5)*[C['width']/W,C['height']/H]-.5);conf.append(c[keep])
        xa,xb,conf=np.concatenate(xa),np.concatenate(xb),np.concatenate(conf);ids=[];ua=set();ub=set()
        for j in np.argsort(-conf):
            ka=tuple((xa[j]/C['matching']['unique_erp_cell']).astype(int));kb=tuple((xb[j]/C['matching']['unique_erp_cell']).astype(int))
            if ka in ua or kb in ub:continue
            ua.add(ka);ub.add(kb);ids.append(j)
        tmp=dest.with_suffix('.partial.npz');np.savez(tmp,a=a,b=b,pixels_a=xa[ids],pixels_b=xb[ids],confidence=conf[ids]);tmp.replace(dest)
        save(R/'hires_matching_status.json',dict(status='matching',completed=k+1,total=len(pairs),physical_gpu=C['physical_gpu']))
        print('HIRES_MATCH',a,b,len(ids),flush=True)
    save(R/'hires_matching_summary.json',dict(status='complete',seconds=time.perf_counter()-started,gpu_forward_seconds=gpu,
         pairs=pairs,source_rgb=[W,H],perspective_size=S,panovggt_input_unchanged=[C['width'],C['height']],confidence=C['matching']['confidence'],unique_grid=C['matching']['unique_erp_cell'],physical_gpu=C['physical_gpu']))
if __name__=='__main__':main()
