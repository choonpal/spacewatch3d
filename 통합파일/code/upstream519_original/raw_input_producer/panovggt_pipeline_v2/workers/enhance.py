"""Supplement weak temporal bridges with mutual SIFT; keep the same 3D gates."""
from pathlib import Path
import os,json,time,functools
os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']='2'
import numpy as np,cv2
from infer import save
from runtime import R,C,replay_files
M=json.loads((R/'input/manifest.json').read_text())
SC=C['matching']['sift']
from refine import components
cv2.setNumThreads(2)
valid=np.load(R/'valid.npy',mmap_mode='r');H,W=valid.shape[1:]
sift=cv2.SIFT_create(nfeatures=SC['features'],contrastThreshold=SC['contrast'],edgeThreshold=SC['edge_threshold'])
@functools.lru_cache(maxsize=40)
def features(f):
    im=cv2.imread(str(R/'input'/M['frames'][f]['file']),0);mask=valid[f].astype(np.uint8)*255
    xy=[];desc=[]
    for shift in [0,W//2]:
        kp,d=sift.detectAndCompute(np.roll(im,shift,axis=1),np.roll(mask,shift,axis=1))
        if d is not None:
            p=np.array([k.pt for k in kp]);p[:,0]=(p[:,0]-shift)%W;xy.append(p);desc.append(d)
    return (np.concatenate(xy),np.concatenate(desc)) if xy else (np.empty((0,2)),np.empty((0,128),np.float32))

def main():
    start=time.perf_counter();edges=json.loads((R/'initial_pair_transforms.json').read_text());labels,groups=components(edges)
    pairs=[(a,a+d) for a in range(len(M['frames'])) for d in range(1,SC['max_gap']+1) if a+d<len(labels) and labels[a]!=labels[a+d]]
    out=R/'enhanced_matches';out.mkdir(exist_ok=True)
    if not pairs or replay_files('enhanced_matches',[f'{a:03d}_{b:03d}.npz' for a,b in pairs]):
        save(R/'enhanced_matching_summary.json',dict(status='complete',pairs=len(pairs),mode='verified_cache_replay_or_empty'));return
    matcher=cv2.BFMatcher(cv2.NORM_L2);audit=[]
    for a,b in pairs:
        pa,da=features(a);pb,db=features(b)
        fw=matcher.knnMatch(da,db,k=2) if len(da)>=2 and len(db)>=2 else [];rev=matcher.knnMatch(db,da,k=2) if len(da)>=2 and len(db)>=2 else []
        reverse={m.queryIdx:m.trainIdx for m,n in rev if m.distance<SC['ratio']*n.distance}
        keep=[m for m,n in fw if m.distance<SC['ratio']*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
        keep.sort(key=lambda m:m.distance)
        xa=[];xb=[];conf=[]
        old=R/'bridge_matches'/f'{a:03d}_{b:03d}.npz'
        if not old.exists():old=R/'matches'/old.name
        if old.exists():
            with np.load(old) as z:xa.extend(z['pixels_a']);xb.extend(z['pixels_b']);conf.extend(z['confidence'])
        original=len(xa)
        for m in keep:xa.append(pa[m.queryIdx]);xb.append(pb[m.trainIdx]);conf.append(.5)
        xa=np.array(xa).reshape(-1,2);xb=np.array(xb).reshape(-1,2);conf=np.array(conf)
        ua=set();ub=set();ids=[]
        for k in range(len(xa)):
            ka=tuple((xa[k]/C['matching']['unique_erp_cell']).astype(int));kb=tuple((xb[k]/C['matching']['unique_erp_cell']).astype(int))
            if ka in ua or kb in ub:continue
            ua.add(ka);ub.add(kb);ids.append(k)
        np.savez(out/f'{a:03d}_{b:03d}.npz',a=a,b=b,pixels_a=xa[ids],pixels_b=xb[ids],confidence=conf[ids])
        audit.append(dict(a=a,b=b,previous_matches=original,sift_mutual=len(keep),merged_matches=len(ids)))
        print('ENHANCE',a,b,original,len(keep),len(ids),flush=True)
    save(R/'enhanced_matching_summary.json',dict(seconds=time.perf_counter()-start,pairs=len(pairs),audit=audit,
         method='LoFTR plus configured mutual SIFT; unchanged 3D acceptance'))
if __name__=='__main__':main()
