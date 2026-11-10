"""Temporal visual matches on the sole configured GPU, with optional cross-face bridges."""
from pathlib import Path
import os, json, time, argparse, functools, subprocess
from runtime import R,C,gpu_guard,replay_files
os.environ['CUDA_VISIBLE_DEVICES'] = str(C['physical_gpu'])
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import cv2
from infer import save
M = json.loads((R / 'input/manifest.json').read_text())
S = C['matching']['face_size']; H, W = C['height'], C['width']

def erp_coords(xy, yaw):
    x = (xy[..., 0]+.5-S/2)/(S/2); y = (xy[..., 1]+.5-S/2)/(S/2)
    lon = np.arctan2(x, np.ones_like(x)) + yaw; lat = np.arctan2(y, np.sqrt(x*x+1))
    return np.stack((((lon/(2*np.pi)+.5)*W-.5) % W, (lat/np.pi+.5)*H-.5), -1)

@functools.lru_cache(maxsize=12)
def views(f):
    p = R / 'input' / M['frames'][f]['file']
    im = cv2.imread(str(p), 0); mask = cv2.imread(str(R / 'masks' / (p.stem+'.png')), 0)
    yy, xx = np.mgrid[:S, :S]; xy = np.stack((xx, yy), -1); out = []
    for face in range(4):
        uv = erp_coords(xy, face*np.pi/2).astype(np.float32)
        out.append((cv2.remap(im, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP),
                    cv2.remap(mask, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) == 0))
    return out

def run(bridge=False):
    n=len(M['frames'])
    pairs=json.loads((R/'bridge_pairs.json').read_text()) if bridge else [(a,a+d) for a in range(n) for d in C['matching']['gaps'] if a+d<n]
    folder_name='bridge_matches' if bridge else 'matches'
    if not pairs or replay_files(folder_name,[f'{a:03d}_{b:03d}.npz' for a,b in pairs]):
        (R/folder_name).mkdir(exist_ok=True)
        save(R/('bridge_matching_summary.json' if bridge else 'matching_summary.json'),dict(status='complete',pairs=len(pairs),mode='verified_cache_replay_or_empty',gpu_used=False));return
    gpu_guard()
    import torch, kornia
    from kornia.feature import LoFTR
    cv2.setNumThreads(2); torch.set_num_threads(2)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(C['physical_gpu']) and torch.cuda.device_count() == 1
    torch.manual_seed(C['seed'])
    n = len(M['frames']); start = time.perf_counter(); gpu_seconds = 0
    pairs = json.loads((R/'bridge_pairs.json').read_text()) if bridge else [(a, a+d) for a in range(n) for d in C['matching']['gaps'] if a+d < n]
    folder = R / ('bridge_matches' if bridge else 'matches'); folder.mkdir(exist_ok=True)
    matcher = LoFTR(pretrained=C['matching']['model']).eval().cuda()
    for k, (a, b) in enumerate(pairs):
        dest = folder / f'{a:03d}_{b:03d}.npz'
        if dest.exists(): raise RuntimeError('Unreceipted matching output: '+str(dest))
        va, vb = views(a), views(b); xa=[]; xb=[]; scores=[]
        for fa in range(4):
            for fb in (range(4) if bridge else [fa]):
                ia, ma = va[fa]; ib, mb = vb[fb]
                ta = torch.from_numpy(ia.copy()).float()[None,None].cuda()/255
                tb = torch.from_numpy(ib.copy()).float()[None,None].cuda()/255
                torch.cuda.synchronize(); t=time.perf_counter()
                with torch.inference_mode(): out=matcher({'image0':ta,'image1':tb})
                torch.cuda.synchronize(); gpu_seconds += time.perf_counter()-t
                pa=out['keypoints0'].cpu().numpy(); pb=out['keypoints1'].cpu().numpy(); score=out['confidence'].cpu().numpy()
                ai=np.rint(pa).astype(int).clip(0,S-1); bi=np.rint(pb).astype(int).clip(0,S-1)
                keep=(score >= C['matching']['confidence']) & ma[ai[:,1],ai[:,0]] & mb[bi[:,1],bi[:,0]]
                keep &= (pa.min(1)>C['matching']['border_pixels'])&(pa.max(1)<S-C['matching']['border_pixels']-1)&(pb.min(1)>C['matching']['border_pixels'])&(pb.max(1)<S-C['matching']['border_pixels']-1)
                xa.append(erp_coords(pa[keep],fa*np.pi/2)); xb.append(erp_coords(pb[keep],fb*np.pi/2)); scores.append(score[keep])
        xa,xb,scores=np.concatenate(xa),np.concatenate(xb),np.concatenate(scores)
        ua=set(); ub=set(); ids=[]
        for j in np.argsort(-scores):
            ka=tuple((xa[j]/C['matching']['unique_erp_cell']).astype(int)); kb=tuple((xb[j]/C['matching']['unique_erp_cell']).astype(int))
            if ka in ua or kb in ub: continue
            ua.add(ka); ub.add(kb); ids.append(j)
        tmp=dest.with_suffix('.partial.npz'); np.savez(tmp,a=a,b=b,pixels_a=xa[ids],pixels_b=xb[ids],confidence=scores[ids]); tmp.replace(dest)
        save(R/'matching_status.json',dict(status='matching',bridge=bridge,completed=k+1,total=len(pairs),physical_gpu=C['physical_gpu']))
        if k%20==0: print('MATCH',bridge,k+1,len(pairs),len(ids),flush=True)
    del matcher; torch.cuda.empty_cache()
    report=dict(status='complete',pairs=len(pairs),wall_seconds=time.perf_counter()-start,gpu_forward_seconds=gpu_seconds,
                physical_gpu=C['physical_gpu'],kornia=kornia.__version__,bridge=bridge)
    save(R/('bridge_matching_summary.json' if bridge else 'matching_summary.json'),report)
    save(R/'matching_status.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bridge',action='store_true'); run(p.parse_args().bridge)
