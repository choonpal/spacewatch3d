"""Visual frame registration; fixed first camera gauge, no motion prior."""
from pathlib import Path
import os,sys,json,time
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from infer import save,digest
import panovggt_global as pg
from runtime import R as ROOT,C
G=C['registration'];O=C['optimization']
M=json.loads((ROOT/'input/manifest.json').read_text())
N=len(M['frames']);H,W=C['height'],C['width']
def proper(poses):
    poses=poses.copy();poses[:,:3,:3]=Rotation.from_matrix(poses[:,:3,:3]).as_matrix();return poses

def transform(x,r,t):return np.einsum('nij,nj->ni',r,x)+t

class Problem:
    def __init__(self,obs,initial):
        self.initial=initial;self.a=obs[:,0].astype(int);self.b=obs[:,1].astype(int)
        self.xa=obs[:,2:5];self.xb=obs[:,5:8];self.depth=obs[:,-1]
        _,inv,counts=np.unique(self.a*1000+self.b,return_inverse=True,return_counts=True)
        self.weights=1/np.sqrt(counts[inv])/(O['depth_weight_fraction']*self.depth)
        self.x0=np.c_[initial[1:,:3,3],Rotation.from_matrix(initial[1:,:3,:3]).as_rotvec()].ravel()
    def unpack(self,x):
        p=self.initial.copy();v=x.reshape(-1,6);p[1:,:3,3]=v[:,:3];p[1:,:3,:3]=Rotation.from_rotvec(v[:,3:]).as_matrix();return p
    def residual(self,x):
        p=self.unpack(x);r=p[:,:3,:3];t=p[:,:3,3]
        return ((transform(self.xa,r[self.a],t[self.a])-transform(self.xb,r[self.b],t[self.b]))*self.weights[:,None]).ravel()
    def solve(self,max_nfev=None,linear_solver=None):
        if max_nfev is None:max_nfev=O['attempts'][0]['max_nfev']
        if linear_solver is None:linear_solver=O['attempts'][0]['linear_solver']
        analytic = O.get('analytic_jacobian', True)
        if analytic:
            from registration_jacobian import jacobian
            kwargs = dict(jac=lambda x: jacobian(x,self.initial,self.a,self.b,self.xa,self.xb,self.weights))
        else:
            s=lil_matrix((3*len(self.a),len(self.x0)),dtype=int)
            for k,(a,b) in enumerate(zip(self.a,self.b)):
                for f in [a,b]:
                    if f:s[3*k:3*k+3,6*(f-1):6*f]=1
            kwargs = dict(jac_sparsity=s.tocsr())
        tick=time.perf_counter()
        result=least_squares(self.residual,self.x0,**kwargs,method='trf',loss=O['loss'],f_scale=O['f_scale'],
            max_nfev=max_nfev,x_scale='jac',ftol=O['ftol'],xtol=O['xtol'],gtol=O['gtol'],tr_options=linear_solver)
        return self.unpack(result.x),dict(success=bool(result.success),message=result.message,nfev=int(result.nfev),
            cost=float(result.cost),initial_cost=float(np.sum(O['f_scale']**2*(np.sqrt(1+(self.residual(self.x0)/O['f_scale'])**2)-1))),
            seconds=time.perf_counter()-tick,optimality=float(result.optimality),
            jacobian='analytic_sparse_so3' if analytic else 'sparse_finite_difference')

def initialize(original,edges):
    p=original.copy();known={0};remaining=edges.copy()
    while len(known)<len(original):
        options=[e for e in remaining if (e['a'] in known)!=(e['b'] in known)]
        if not options:
            raise ValueError('Disconnected visual graph; unreachable frames: '+str(sorted(set(range(len(original)))-known)))
        e=min(options,key=lambda e:(abs(e['b']-e['a']),-e['train_inliers']));a,b=e['a'],e['b']
        tf=np.eye(4);tf[:3,:3]=e['rotation'];tf[:3,3]=e['translation']
        if a in known:p[b]=p[a]@tf;known.add(b)
        else:p[a]=p[b]@np.linalg.inv(tf);known.add(a)
        remaining.remove(e)
    return proper(p)

def preflight():
    rng=np.random.default_rng(482);n=8;truth=np.tile(np.eye(4),(n,1,1));truth[:,:3,3]=np.arange(n)[:,None]*[.08,.01,.1]
    obs=[]
    for a in range(n):
        for b in range(a+1,min(n,a+3)):
            for x in rng.normal(size=(20,3))+[0,0,4]:obs.append([a,b,*(x-truth[a,:3,3]),*(x-truth[b,:3,3]),0,0,0,0,4])
    initial=truth.copy();initial[1:,:3,3]+=rng.normal(0,.03,(n-1,3));initial[1:,:3,:3]=Rotation.from_rotvec(rng.normal(0,.01,(n-1,3))).as_matrix()
    fitted,report=Problem(np.array(obs),initial).solve();error=float(np.max(abs(fitted-truth)))
    assert error<1e-4,(error,report)
    save(ROOT/'optimizer_preflight.json',dict(passed=True,max_pose_error=error,solver=report,motion_prior=False))

def lift(depth,valid,uv):
    x0=np.floor(uv[:,0]).astype(int)%W;x1=(x0+1)%W
    y0=np.floor(uv[:,1]).astype(int).clip(0,H-1);y1=(y0+1).clip(0,H-1)
    good=valid[y0,x0]&valid[y0,x1]&valid[y1,x0]&valid[y1,x1]
    ds=np.stack([depth[y0,x0],depth[y0,x1],depth[y1,x0],depth[y1,x1]])
    good&=ds.max(0)-ds.min(0)<G['depth_discontinuity_fraction']*np.median(ds,axis=0)
    dx=uv[:,0]-np.floor(uv[:,0]);dy=uv[:,1]-np.floor(uv[:,1])
    d=(1-dx)*(1-dy)*ds[0]+dx*(1-dy)*ds[1]+(1-dx)*dy*ds[2]+dx*dy*ds[3]
    lon=((uv[:,0]+.5)/W-.5)*2*np.pi;lat=((uv[:,1]+.5)/H-.5)*np.pi
    rays=np.c_[np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)]
    return rays*d[:,None],good&np.isfinite(d)&(d>0)

def prepare():
    start=time.perf_counter();windows=C['window_indices'];owners=pg.owners(windows)
    # Every frame chooses the window where it is closest to the temporal center.
    assert set(owners)==set(range(N))
    scales=[];audit=[];previous=None;hashes={};rawposes=np.empty((N,4,4));stride=G['depth_calibration_stride']
    depths=np.lib.format.open_memmap(ROOT/'depths.npy',mode='w+',dtype=np.float32,shape=(N,H,W))
    valid=np.lib.format.open_memmap(ROOT/'valid.npy',mode='w+',dtype=bool,shape=(N,H,W))
    for wi,ids in enumerate(windows):
        path=ROOT/f'windows/window_{wi:03d}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['indices']==ids and digest(path)==meta['sha256'];hashes[str(path)]=meta['sha256']
        with np.load(path) as z:
            d=z['depths'];v=z['valid'];p=z['poses'];indices=z['indices']
            current=dict(indices=list(indices),depth=d[:,::stride,::stride].copy(),valid=v[:,::stride,::stride].copy())
            if previous is None:
                first_depth=float(np.median(d[0][v[0]]));scales.append(1/first_depth)
            else:
                ratios=[]
                for f in sorted(set(previous['indices'])&set(indices)):
                    a=previous['indices'].index(f);b=list(indices).index(f)
                    mask=previous['valid'][a]&current['valid'][b]
                    ratios.extend((previous['depth'][a][mask]/current['depth'][b][mask]).tolist())
                ratios=np.array(ratios);ratio=float(np.median(ratios))
                assert len(ratios)>G['min_depth_scale_samples'] and G['window_scale_ratio_bounds'][0]<ratio<G['window_scale_ratio_bounds'][1],(wi,len(ratios),ratio)
                scales.append(scales[-1]*ratio)
                audit.append(dict(a=wi-1,b=wi,samples=len(ratios),ratio=ratio,mad=float(np.median(abs(ratios-ratio)))))
            for f,(_,owner,li) in owners.items():
                if owner==wi:depths[f]=d[li]*scales[wi];valid[f]=v[li];rawposes[f]=p[li]
        previous=current
        if wi%10==0:print('PREPARE_DEPTH',wi,len(windows),flush=True)
    depths.flush();valid.flush()
    np.savez(ROOT/'prepared.npz',scales=np.array(scales),owners=np.array([owners[f][1:] for f in range(N)]),rawposes=rawposes)
    save(ROOT/'depth_calibration.json',dict(window_scales=scales,edges=audit,first_frame_median_depth=first_depth,
         method='Same-frame same-pixel radial depth ratios. Each frame inherits its selected window scale; per-pixel depth shape is unchanged.',seconds=time.perf_counter()-start))
    save(ROOT/'source_hashes.json',hashes)

def components(edges):
    adj=[set() for _ in range(N)]
    for e in edges:adj[e['a']].add(e['b']);adj[e['b']].add(e['a'])
    labels=np.full(N,-1,int);groups=[]
    for start in range(N):
        if labels[start]>=0:continue
        label=len(groups);todo=[start];labels[start]=label;group=[]
        while todo:
            a=todo.pop();group.append(a)
            for b in adj[a]:
                if labels[b]<0:labels[b]=label;todo.append(b)
        groups.append(sorted(group))
    return labels,groups

def pair_observations(p,depths,valid,cfg,source_sha256):
    train=[];evaluation=[];accepted_eval=[];edges=[]
    with np.load(p) as z:a=int(z['a']);b=int(z['b']);ua=z['pixels_a'];ub=z['pixels_b']
    rec=dict(a=a,b=b,matches=len(ua),accepted=False,cross_face=p.parent.name=='bridge_matches',match_source=p.parent.name,source_sha256=source_sha256)
    if len(ua)<G['min_image_matches']:rec['reason']='Too few image matches';return dict(train=train,evaluation=evaluation,accepted_eval=accepted_eval,edges=edges,audit=rec)
    xa,va=lift(depths[a],valid[a],ua);xb,vb=lift(depths[b],valid[b],ub)
    indices=np.flatnonzero(va&vb);rng=np.random.default_rng(G['split_seed']+a*131+b)
    order=rng.permutation(indices);cut=int(G['train_fraction']*len(order));tr,te=order[:cut],order[cut:]
    rec.update(valid_matches=len(indices),train=len(tr),heldout=len(te))
    if len(indices)<G['min_valid_depth_matches']:rec['reason']='Too few valid depth matches';return dict(train=train,evaluation=evaluation,accepted_eval=accepted_eval,edges=edges,audit=rec)
    depth=float(np.median(np.r_[np.linalg.norm(xa[indices],axis=1),np.linalg.norm(xb[indices],axis=1)]))
    data=np.c_[np.full(len(ua),a),np.full(len(ua),b),xa,xb,ua,ub,np.full(len(ua),depth)]
    evaluation.extend(data[te].tolist());rec.update(valid_matches=len(indices),train=len(tr),heldout=len(te))
    try:
        tf,_,quality=pg.robust_fit(xb[tr],xa[tr],G['residual_depth_fraction']*depth,cfg,seed_offset=a*131+b)
        used=tr[np.linalg.norm(pg.apply(xb[tr],tf)-xa[tr],axis=1)<G['residual_depth_fraction']*depth]
        if not G['relative_scale_bounds'][0]<tf[0]<G['relative_scale_bounds'][1]:raise ValueError('relative scale outside [0.8,1.25]')
        if len(used)<G['min_inliers']:raise ValueError('insufficient training inliers')
        source,target=xb[used],xa[used];cx=source.mean(0);cy=target.mean(0)
        u,_,vt=np.linalg.svd((target-cy).T@(source-cx));sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
        rot=(u*sign)@vt;trans=cy-rot@cx
        if len(used)>G['max_observations_per_pair']:used=rng.choice(used,G['max_observations_per_pair'],replace=False)
        train.extend(data[used].tolist());accepted_eval.extend(data[te].tolist())
        edges.append(dict(a=a,b=b,rotation=rot.tolist(),translation=trans.tolist(),train_inliers=len(used)))
        rec.update(accepted=True,training_inliers=len(used),relative_scale=float(tf[0]),quality=quality)
    except ValueError as e:rec['reason']=str(e)
    return dict(train=train,evaluation=evaluation,accepted_eval=accepted_eval,edges=edges,audit=rec)

def observations():
    start=time.perf_counter();depths=np.load(ROOT/'depths.npy',mmap_mode='r');valid=np.load(ROOT/'valid.npy',mmap_mode='r')
    from common import select_match_files,record,verify
    ready=json.loads((ROOT/'matching_sources_ready.json').read_text())
    files=select_match_files(ROOT,ready['sources'])
    inputs={name:record(p) for name,p in files.items()}
    save(ROOT/'observation_inputs.json',dict(source_stages=ready,files=inputs))
    train=[];evaluation=[];accepted_eval=[];edges=[];audit=[]
    cfg=pg.Config(min_inlier_fraction=G['min_inlier_fraction'],min_matches=G['min_fitting_matches'],min_inliers=G['min_inliers'],ransac_trials=G['ransac_trials'],seed=C['seed'],inner_train_fraction=G['inner_train_fraction'],min_inner_holdout=G['min_inner_holdout'],inner_inlier_fraction=G['inner_inlier_fraction'])
    cache=None
    if C.get('execution',{}).get('observation_pair_cache',True):
        import scipy
        from observation_cache import PairObservationCache
        context=dict(schema=1,registration=G,seed=C['seed'],shape=[N,H,W],
            cpu_threads=C['execution']['cpu_threads'],numpy=np.__version__,scipy=scipy.__version__,python=sys.version,
            geometry={name:digest(ROOT/name) for name in ['depths.npy','valid.npy']},
            code={name:digest(Path(__file__).with_name(name)) for name in ['refine.py','panovggt_global.py','observation_cache.py']})
        cache=PairObservationCache(ROOT/'observation_pair_cache',context)
    for k,p in enumerate(sorted(files.values(),key=lambda x:x.name)):
        key=dict(file=p.name,source=p.parent.name,sha256=inputs[p.name]['sha256'])
        result=cache.get(key) if cache is not None else None
        if result is None:
            result=pair_observations(p,depths,valid,cfg,inputs[p.name]['sha256'])
            if cache is not None:cache.put(key,result)
        train.extend(result['train']);evaluation.extend(result['evaluation']);accepted_eval.extend(result['accepted_eval'])
        edges.extend(result['edges']);audit.append(result['audit'])
        if k%50==0:print('GEOMETRY',k,len(files),len(edges),flush=True)
    for item in inputs.values():verify(item)
    np.savez(ROOT/'observations.npz',train=np.array(train).reshape(-1,13),evaluation=np.array(evaluation).reshape(-1,13),accepted_eval=np.array(accepted_eval).reshape(-1,13))
    save(ROOT/'edge_audit.json',audit);save(ROOT/'initial_pair_transforms.json',edges)
    labels,groups=components(edges)
    bridges=[(a,a+d) for a in range(N) for d in C['matching']['gaps'] if a+d<N and labels[a]!=labels[a+d]]
    save(ROOT/'bridge_pairs.json',bridges)
    report=dict(accepted_edges=len(edges),tested_pairs=len(files),train=len(train),evaluation=len(evaluation),components=groups,
                connected=len(groups)==1,bridge_pairs=bridges,seconds=time.perf_counter()-start,
                evaluation_scope='Held-out 30% split before geometric fitting, including rejected pairs. Not external ground truth.')
    if cache is not None:report['pair_cache']=cache.statistics()
    save(ROOT/'observation_summary.json',report);return report

def metrics(poses,obs,frame_ids=None):
    a=obs[:,0].astype(int);b=obs[:,1].astype(int);xa=obs[:,2:5];xb=obs[:,5:8];depth=obs[:,-1]
    r=poses[:,:3,:3];t=poses[:,:3,3]
    wa=transform(xa,r[a],t[a]);wb=transform(xb,r[b],t[b]);err=np.linalg.norm(wa-wb,axis=1)/depth
    pa=np.einsum('nji,nj->ni',r[a],wb-t[a]);pb=np.einsum('nji,nj->ni',r[b],wa-t[b])
    def angle(x,y):return np.degrees(np.arccos(np.clip(np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)),-1,1)))
    angles=(angle(pa,xa)+angle(pb,xb))/2;out={}
    ids=np.arange(N) if frame_ids is None else np.asarray(frame_ids)
    for label,mask in [('full',np.ones(len(a),bool)),('40_50_seconds',(np.array([M['frames'][int(ids[i])]['timestamp_s'] for i in a])>=40)&(np.array([M['frames'][int(ids[i])]['timestamp_s'] for i in b])<=50))]:
        if mask.any():out[label]=dict(observations=int(mask.sum()),median_3d_over_depth=float(np.median(err[mask])),p90_3d_over_depth=float(np.quantile(err[mask],.9)),median_angle_deg=float(np.median(angles[mask])),p90_angle_deg=float(np.quantile(angles[mask],.9)),fraction_under_4pct=float(np.mean(err[mask]<.04)))
    out['per_pair']=[dict(a=int(i),b=int(j),median_3d_over_depth=float(np.median(err[(a==i)&(b==j)])),median_angle_deg=float(np.median(angles[(a==i)&(b==j)]))) for i,j in sorted(set(zip(a,b)))]
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','observations','preflight']);a=p.parse_args()
    {'prepare':prepare,'observations':observations,'preflight':preflight}[a.stage]()
