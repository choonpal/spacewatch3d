"""CPU loop closure and Sim(3) pose-graph optimization for PanoVGGT windows.

This is a standalone reconstruction helper, not the task 1b frame-manifest backend.
Requires NumPy, SciPy, and OpenCV (SIFT). No torch/model import on this path.
Local-to-world transforms use p_world = scale * (p_local @ rotation.T) + translation.
Edges map node j into node i. Node zero fixes the monocular gauge.
This optimizes window transforms, NOT per-pixel depth or full bundle adjustment.

CLI: python panovggt_global.py --run precise_step1 --input input --masks masks
     --output precise_step1_global [--detect-only]
Dense PLY regeneration uses export_dense() after optimization. A fused
PLY alone is insufficient: raw windows retain the frame/pixel correspondences.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import tempfile

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Config:
    keyframe_stride: int = 3
    min_frame_gap: int = 15
    min_matches: int = 30
    min_inliers: int = 20
    min_inlier_fraction: float = 0.45
    ratio: float = 0.70
    ransac_trials: int = 700
    residual_depth_fraction: float = 0.04
    min_image_bins: int = 4
    max_loops: int = 16
    max_candidates: int = 80
    edge_sample_limit: int = 1600
    rotation_sigma_degrees: float = 2.0
    log_scale_sigma: float = 0.04
    translation_sigma_depth_fraction: float = 0.03
    max_nfev: int = 300
    seed: int = 42


def identity():
    return (1.0, np.eye(3), np.zeros(3))


def apply(points, transform):
    s, r, t = transform
    return s * (np.asarray(points) @ r.T) + t


def compose(a, b):
    sa, ra, ta = a
    sb, rb, tb = b
    return sa * sb, ra @ rb, sa * (ra @ tb) + ta


def inverse(a):
    s, r, t = a
    return 1 / s, r.T, -(r.T @ t) / s


def fit_similarity(source, target):
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 4:
        raise ValueError('At least four paired 3D points are required.')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite correspondences.')
    xc, yc = x.mean(0), y.mean(0)
    xx, yy = x - xc, y - yc
    # Planar scenes are valid; collinear correspondences do not constrain rotation.
    for centered in (xx, yy):
        eig = np.linalg.svd(centered, compute_uv=False)
        if eig[0] < 1e-9 or eig[1] < 1e-3 * eig[0]:
            raise ValueError('Degenerate, nearly collinear correspondences.')
    u, singular, vt = np.linalg.svd(yy.T @ xx / len(x))
    correction = np.ones(3)
    correction[-1] = np.sign(np.linalg.det(u @ vt))
    r = (u * correction) @ vt
    s = float(np.sum(singular * correction) / np.mean(np.sum(xx**2, axis=1)))
    if not np.isfinite(s) or not .25 < s < 4:
        raise ValueError('Implausible relative scale.')
    return s, r, yc - s * (r @ xc)


def robust_fit(source, target, threshold, cfg, seed_offset=0):
    """RANSAC with an independent holdout; fit never uses holdout points."""
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('Invalid residual threshold.')
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < cfg.min_matches:
        raise ValueError('Too few 3D matches.')
    rng = np.random.default_rng(cfg.seed + seed_offset)
    order = rng.permutation(len(source)); split = max(4, int(.75 * len(order)))
    train, held = order[:split], order[split:]
    x, y = source[train], target[train]
    best = None; best_count = 0
    for _ in range(cfg.ransac_trials):
        ids = rng.choice(len(x), 4, replace=False)
        try:
            transform = fit_similarity(x[ids], y[ids])
        except ValueError:
            continue
        keep = np.linalg.norm(apply(x, transform)-y, axis=1) < threshold
        if int(keep.sum()) > best_count:
            best, best_count = keep, int(keep.sum())
    if best is None or best_count < max(8, int(cfg.min_inlier_fraction * len(train))):
        raise ValueError('RANSAC has insufficient consensus.')
    for _ in range(3):
        transform = fit_similarity(x[best], y[best])
        best = np.linalg.norm(apply(x, transform)-y, axis=1) < threshold
        if best.sum() < 4:
            raise ValueError('Consensus collapsed during refit.')
    transform = fit_similarity(x[best], y[best])
    errors = np.linalg.norm(apply(source, transform)-target, axis=1)
    inliers = errors < threshold
    if inliers.sum() < cfg.min_inliers or np.mean(inliers) < cfg.min_inlier_fraction:
        raise ValueError('Insufficient 3D inlier coverage.')
    if len(held) < 4 or np.mean(inliers[held]) < .5 or np.median(errors[held]) > threshold:
        raise ValueError('Independent 3D holdout validation failed.')
    return transform, inliers, dict(matches=len(source), inliers=int(inliers.sum()),
        inlier_fraction=float(inliers.mean()), median_inlier_residual=float(np.median(errors[inliers])),
        holdout_count=len(held), holdout_inlier_fraction=float(inliers[held].mean()),
        holdout_median_residual=float(np.median(errors[held])), holdout_indices=held.tolist(),
        threshold=float(threshold))


def transform_json(t):
    return dict(scale=float(t[0]), rotation=t[1].tolist(), translation=t[2].tolist())


def make_edge(i, j, measurement, depth, kind, quality=None):
    return dict(i=int(i), j=int(j), kind=kind, measurement=transform_json(measurement),
                depth=float(depth), quality=quality or {})


def edge_transform(edge):
    m = edge['measurement']
    return float(m['scale']), np.asarray(m['rotation']), np.asarray(m['translation'])


def _edge_residual(transforms, edge, cfg):
    actual = compose(inverse(transforms[edge['i']]), transforms[edge['j']])
    expected = edge_transform(edge)
    sigma_t = max(edge['depth'] * cfg.translation_sigma_depth_fraction, 1e-6)
    # Similarity relative pose residual; translation is in the target node's units.
    return np.r_[Rotation.from_matrix(expected[1].T @ actual[1]).as_rotvec()
                 / np.radians(cfg.rotation_sigma_degrees),
                 (actual[2]-expected[2]) / sigma_t,
                 np.log(actual[0]/expected[0]) / cfg.log_scale_sigma]


def optimize_graph(initial, edges, cfg=Config()):
    """Robust sparse Sim(3) pose graph, anchored at node 0; reject no-loop runs."""
    n = len(initial)
    if n < 2 or not any(e['kind'] == 'loop' for e in edges):
        raise ValueError('No verified loop: global loop optimization was not performed.')
    reached = {0}
    while True:
        old = len(reached)
        for e in edges:
            if not (0 <= e['i'] < n and 0 <= e['j'] < n and e['i'] != e['j']):
                raise ValueError('Invalid graph node index.')
            if e['i'] in reached or e['j'] in reached:
                reached.update((e['i'], e['j']))
        if len(reached) == old:
            break
    if len(reached) != n:
        raise ValueError('Disconnected graph.')
    def pack(ts):
        return np.concatenate([np.r_[np.log(s), Rotation.from_matrix(r).as_rotvec(), t] for s,r,t in ts[1:]])
    def unpack(v):
        return [initial[0]] + [(float(np.exp(row[0])), Rotation.from_rotvec(row[1:4]).as_matrix(), row[4:7]) for row in v.reshape(-1,7)]
    def residual(v):
        ts = unpack(v)
        return np.concatenate([_edge_residual(ts,e,cfg) for e in edges])
    sparsity = lil_matrix((7*len(edges),7*(n-1)), dtype=int)
    for k,e in enumerate(edges):
        for node in (e['i'],e['j']):
            if node:
                sparsity[7*k:7*k+7,7*(node-1):7*node] = 1
    x0 = pack(initial); lower = np.full_like(x0,-np.inf); upper = -lower
    # Positive scales by log parameterization, with finite bounds against collapse.
    lower[::7] = x0[::7]-np.log(4); upper[::7] = x0[::7]+np.log(4)
    result = least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),bounds=(lower,upper),
        loss='soft_l1',f_scale=1,method='trf',x_scale='jac',max_nfev=cfg.max_nfev,
        ftol=1e-7,xtol=1e-7,gtol=1e-7)
    ts = unpack(result.x)
    before,after = residual(x0).reshape(-1,7),residual(result.x).reshape(-1,7)
    cost = lambda x: float(np.sum(np.sqrt(1+x*x)-1))
    if not result.success or not np.isfinite(result.x).all() or cost(after) > cost(before)+1e-8:
        raise RuntimeError(f'Global optimizer did not converge safely: {result.message}')
    rows=[]
    for e,b,a in zip(edges,before,after):
        rows.append(dict(i=e['i'],j=e['j'],kind=e['kind'],before_normalized_norm=float(np.linalg.norm(b)),
                         after_normalized_norm=float(np.linalg.norm(a))))
    loops = [k for k,e in enumerate(edges) if e['kind']=='loop']
    # Do not publish a result that simply ignores all long-range constraints.
    loop_before=float(np.median(np.linalg.norm(before[loops],axis=1)))
    loop_after=float(np.median(np.linalg.norm(after[loops],axis=1)))
    if loop_before > 1 and loop_after >= .95*loop_before:
        raise RuntimeError('Verified loop residual did not improve; inspect graph conflicts.')
    report=dict(success=True,message=str(result.message),evaluations=int(result.nfev),
        before_robust_cost=cost(before),after_robust_cost=cost(after),
        loop_before_median_normalized_norm=loop_before,loop_after_median_normalized_norm=loop_after,
        anchor_node=0,nodes=n,edges=len(edges),loop_edges=len(loops),edge_residuals=rows,
        scope='Window Sim(3) optimization; not full bundle adjustment or depth refinement',
        residuals_are_not_ground_truth_accuracy=True)
    return ts,report


def owners(window_indices):
    result={}
    for wi,ids in enumerate(window_indices):
        for local,frame in enumerate(ids):
            score=abs(local-(len(ids)-1)/2)
            if frame not in result or score < result[frame][0]:
                result[frame]=(score,wi,local)
    return result


def detect_candidates(frame_paths,mask_paths,cfg=Config()):
    """Image-only retrieval. Candidates are NOT accepted 3D loop closures."""
    import cv2
    cv2.setRNGSeed(cfg.seed)
    ids=sorted(set(range(0,len(frame_paths),cfg.keyframe_stride)) | {len(frame_paths)-1})
    detector=cv2.SIFT_create(nfeatures=4000);features={};shapes={}
    for i in ids:
        image=cv2.imread(str(frame_paths[i]),cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(str(mask_paths[i]),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != mask.shape:
            raise ValueError(f'Missing or incompatible frame/mask: {i}')
        valid=np.where(mask==0,255,0).astype(np.uint8);valid[:int(.03*len(valid))]=0
        kp,desc=detector.detectAndCompute(image,valid)
        features[i]=(np.array([k.pt for k in kp]).reshape(-1,2),desc);shapes[i]=image.shape
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=96))
    candidates=[]
    for ia,a in enumerate(ids):
        for b in ids[ia+1:]:
            if b-a < cfg.min_frame_gap: continue
            pa,da=features[a];pb,db=features[b]
            if da is None or db is None or min(len(da),len(db))<2: continue
            forward=matcher.knnMatch(da,db,k=2);reverse=matcher.knnMatch(db,da,k=2)
            rev={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2 for m,n in [pair] if m.distance<cfg.ratio*n.distance}
            matches=[m for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<cfg.ratio*n.distance and rev.get(m.trainIdx)==m.queryIdx]
            if len(matches)<cfg.min_matches: continue
            xa=pa[[m.queryIdx for m in matches]];xb=pb[[m.trainIdx for m in matches]]
            if min(image_bins(xa,shapes[a]),image_bins(xb,shapes[b]))<cfg.min_image_bins: continue
            candidates.append(dict(a=a,b=b,matches=len(matches),pixels_a=xa,pixels_b=xb,
                                   shape_a=shapes[a],shape_b=shapes[b]))
    return sorted(candidates,key=lambda c:(-c['matches'],c['a'],c['b']))[:cfg.max_candidates]


def image_bins(points,shape):
    h,w=shape
    bins=np.floor(np.asarray(points)/[w,h]*[8,4]).astype(int)
    return len(np.unique(bins,axis=0))


def candidate_json(c):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()}


class WindowCache:
    """A bounded two-window cache; full-resolution arrays can be very large."""
    def __init__(self,run,indices):
        self.run,self.indices=Path(run),indices;self.cache={}
    def get(self,i):
        if i not in self.cache:
            if len(self.cache)>=2: self.cache.pop(next(iter(self.cache)))
            with np.load(self.run/f'windows/window_{i:03d}.npz',allow_pickle=False) as z:
                data={k:z[k] for k in ['indices','points','valid','depths']}
            if not np.array_equal(data['indices'],self.indices[i]):
                raise ValueError(f'Window {i} frame indices differ from settings.')
            if data['points'].shape[:-1] != data['valid'].shape or data['valid'].shape != data['depths'].shape:
                raise ValueError(f'Window {i} has incompatible array shapes.')
            self.cache[i]=data
        return self.cache[i]


def lift(data,local,pixels,pixel_step,image_shape):
    # points[:, ::pixel_step, ::pixel_step] stores original pixels 0,step,2*step...
    coords=np.rint(pixels/pixel_step).astype(int)
    h,w=data['valid'].shape[1:]
    inside=(coords[:,0]>=0)&(coords[:,0]<w)&(coords[:,1]>=0)&(coords[:,1]<h)
    coords[:,0]=coords[:,0].clip(0,w-1);coords[:,1]=coords[:,1].clip(0,h-1)
    x,y=coords.T;points=data['points'][local,y,x];depth=data['depths'][local,y,x]
    valid=inside & data['valid'][local,y,x] & np.isfinite(points).all(1) & np.isfinite(depth) & (depth>0)
    expected=((image_shape[0]+pixel_step-1)//pixel_step,(image_shape[1]+pixel_step-1)//pixel_step)
    if (h,w)!=expected: raise ValueError('Image dimensions/pixel_step do not match prediction cache.')
    return points,depth,valid


def build_graph(run,settings,candidates,cfg):
    indices=settings['window_indices'];cache=WindowCache(run,indices);owner=owners(indices)
    edges=[];audits=[];rng=np.random.default_rng(cfg.seed)
    # Re-estimate all shared-frame constraints, including nonconsecutive overlaps.
    for i in range(len(indices)):
        for j in range(i+1,len(indices)):
            common=sorted(set(indices[i])&set(indices[j]))
            if not common: continue
            a,b=cache.get(i),cache.get(j);xs=[];ys=[];depths=[]
            for f in common:
                ia=indices[i].index(f);ib=indices[j].index(f)
                valid=a['valid'][ia]&b['valid'][ib]
                y=a['points'][ia][valid];x=b['points'][ib][valid];d=a['depths'][ia][valid]
                ok=np.isfinite(x).all(1)&np.isfinite(y).all(1)&np.isfinite(d)&(d>0)
                take=np.flatnonzero(ok)
                if len(take)>cfg.edge_sample_limit:take=rng.choice(take,cfg.edge_sample_limit,replace=False)
                xs.append(x[take]);ys.append(y[take]);depths.append(d[take])
            x,y=np.concatenate(xs),np.concatenate(ys);depth=float(np.median(np.concatenate(depths)))
            try:
                transform,keep,quality=robust_fit(x,y,depth*cfg.residual_depth_fraction,cfg,i*1000+j)
            except ValueError as exc:
                if j==i+1:raise ValueError(f'Adjacent constraint {i}->{j} invalid: {exc}') from exc
                audits.append(dict(kind='overlap',i=i,j=j,accepted=False,reason=str(exc)));continue
            quality['shared_frames']=common
            edges.append(make_edge(i,j,transform,depth,'overlap',quality))
    loop_nodes=set()
    for k,c in enumerate(candidates):
        _,i,ia=owner[c['a']];_,j,ib=owner[c['b']]
        if i==j or (i,j) in loop_nodes: continue
        if len(loop_nodes)>=cfg.max_loops: break
        a,b=cache.get(i),cache.get(j)
        ya,da,va=lift(a,ia,c['pixels_a'],settings['pixel_step'],c['shape_a'])
        xb,db,vb=lift(b,ib,c['pixels_b'],settings['pixel_step'],c['shape_b'])
        valid=va&vb
        audit=dict(kind='loop',a=c['a'],b=c['b'],i=i,j=j,image_matches=c['matches'],valid_3d_matches=int(valid.sum()))
        try:
            if valid.sum()<cfg.min_matches: raise ValueError('Too few valid depth matches.')
            depth=float(np.median(da[valid]))
            transform,keep,quality=robust_fit(xb[valid],ya[valid],depth*cfg.residual_depth_fraction,cfg,20000+k)
            bins_a=image_bins(c['pixels_a'][valid][keep],c['shape_a'])
            bins_b=image_bins(c['pixels_b'][valid][keep],c['shape_b'])
            if min(bins_a,bins_b)<cfg.min_image_bins:raise ValueError('3D inliers occupy too little image area.')
            quality.update(frame_a=c['a'],frame_b=c['b'],image_bins_a=bins_a,image_bins_b=bins_b)
            held=quality['holdout_indices']
            quality['holdout_points_i']=ya[valid][held].tolist()
            quality['holdout_points_j']=xb[valid][held].tolist()
            edges.append(make_edge(i,j,transform,depth,'loop',quality));loop_nodes.add((i,j))
            audit.update(accepted=True,quality=quality)
        except ValueError as exc:
            audit.update(accepted=False,reason=str(exc))
        audits.append(audit)
    return edges,audits


def optimize_run(run,input_dir,mask_dir,output,cfg=Config(),detect_only=False):
    run,input_dir,mask_dir,output=map(Path,(run,input_dir,mask_dir,output))
    if output.resolve()==run.resolve() or run.resolve() in output.resolve().parents:
        raise ValueError('Use a separate output directory; source reconstruction is read-only.')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be new or empty: {output}')
    settings=json.loads((run/'settings.json').read_text())
    manifest=json.loads((run/'input_manifest.json').read_text())
    paths=[input_dir/f['file'] for f in manifest['frames']]
    masks=[mask_dir/(p.stem+'.png') for p in paths]
    if not detect_only:
        missing=[i for i in range(len(settings['window_indices'])) if not (run/f'windows/window_{i:03d}.npz').is_file()]
        if missing:
            raise FileNotFoundError(f'Missing raw prediction windows {missing}. Re-run precise inference or restore the raw cache. The merged PLY cannot replace frame/pixel provenance.')
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    candidates=detect_candidates(paths,masks,cfg)
    (output/'loop_candidates.json').write_text(json.dumps([candidate_json(c) for c in candidates],indent=2))
    (output/'global_config.json').write_text(json.dumps(asdict(cfg),indent=2))
    if detect_only:
        report=dict(status='image_candidates_only',candidates=len(candidates),verified_3d_loops=0,seconds=time.perf_counter()-start)
        (output/'detection_report.json').write_text(json.dumps(report,indent=2));return None,report
    edges,audits=build_graph(run,settings,candidates,cfg)
    (output/'pose_graph.json').write_text(json.dumps(edges,indent=2))
    (output/'loop_validation.json').write_text(json.dumps(audits,indent=2))
    with np.load(run/'window_transforms.npz',allow_pickle=False) as z:
        initial=[(float(s),r,t) for s,r,t in zip(z['scale'],z['rotation'],z['translation'])]
    if len(initial)!=len(settings['window_indices']):raise ValueError('Transform count differs from window count.')
    optimized,report=optimize_graph(initial,edges,cfg)
    report['loop_holdout_3d_residuals']=[]
    for edge in edges:
        if edge['kind']!='loop':continue
        quality=edge['quality'];i,j=edge['i'],edge['j']
        pi=np.asarray(quality['holdout_points_i']);pj=np.asarray(quality['holdout_points_j'])
        before=np.linalg.norm(apply(pi,initial[i])-apply(pj,initial[j]),axis=1)
        after=np.linalg.norm(apply(pi,optimized[i])-apply(pj,optimized[j]),axis=1)
        report['loop_holdout_3d_residuals'].append(dict(frame_a=quality['frame_a'],frame_b=quality['frame_b'],
            before_median=float(np.median(before)),after_median=float(np.median(after)),
            units='first-window monocular units',not_ground_truth=True))
    report.update(seconds=time.perf_counter()-start,source_run=str(run.resolve()),config=asdict(cfg))
    np.savez(output/'window_transforms.npz',scale=np.array([t[0] for t in optimized]),
             rotation=np.stack([t[1] for t in optimized]),translation=np.stack([t[2] for t in optimized]))
    (output/'global_optimization.json').write_text(json.dumps(report,indent=2))
    return optimized,report


def export_dense(run,output,transforms):
    """Re-fuse every original frame using optimized transforms; bounded RAM staging.

    Source colors/validity/ownership match precise_engine. Old per-point semantic
    labels are intentionally not copied because voxel membership/order changes.
    """
    run,output=Path(run),Path(output)
    if run.resolve()==output.resolve():raise ValueError('Source reconstruction is read-only.')
    if (output/'panovggt_merged.ply').exists():raise FileExistsError('Dense output already exists.')
    settings=json.loads((run/'settings.json').read_text())
    manifest=json.loads((run/'input_manifest.json').read_text())
    indices=settings['window_indices'];owner=owners(indices)
    if len(transforms)!=len(indices):raise ValueError('Wrong number of optimized transforms.')
    for s,r,t in transforms:
        if not np.isfinite(s) or s<=0 or not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError('Invalid optimized transform.')
    output.mkdir(parents=True,exist_ok=True)
    voxel=float(settings['voxel_size'])
    if not np.isfinite(voxel) or voxel<=0:raise ValueError('Invalid voxel size.')
    owned={wi:sorted((f,li) for f,(_,w,li) in owner.items() if w==wi) for wi in range(len(indices))}
    raw=0
    for wi in range(len(indices)):
        with np.load(run/f'windows/window_{wi:03d}.npz',allow_pickle=False) as z:
            if not np.array_equal(z['indices'],indices[wi]):raise ValueError('Cache frame indices differ.')
            valid=z['valid']
            raw+=sum(int(valid[li].sum()) for _,li in owned[wi])
    if raw==0:raise ValueError('No valid source points.')
    start=time.perf_counter();trajectory=[]
    with tempfile.TemporaryDirectory(prefix='fusion_',dir=output) as work:
        work=Path(work)
        xyz=np.lib.format.open_memmap(work/'xyz.npy',mode='w+',dtype=np.float32,shape=(raw,3))
        rgb=np.lib.format.open_memmap(work/'rgb.npy',mode='w+',dtype=np.uint8,shape=(raw,3))
        offset=0
        for wi in range(len(indices)):
            with np.load(run/f'windows/window_{wi:03d}.npz',allow_pickle=False) as z:
                points,colors,valid,poses=z['points'],z['colors'],z['valid'],z['poses']
            for frame,local in owned[wi]:
                keep=valid[local];n=int(keep.sum())
                transformed=apply(points[local][keep],transforms[wi])
                if not np.isfinite(transformed).all():raise ValueError('Non-finite fused points.')
                xyz[offset:offset+n]=transformed;rgb[offset:offset+n]=colors[local][keep];offset+=n
                pose=np.eye(4);pose[:3,:3]=transforms[wi][1]@poses[local,:3,:3]
                pose[:3,3]=apply(poses[local,:3,3],transforms[wi])
                trajectory.append(dict(frame=frame,timestamp_s=manifest['frames'][frame]['timestamp_s'],
                    camera_to_world=pose.tolist(),source_window=wi))
            del points,colors,valid,poses
        if offset!=raw:raise RuntimeError('Source point count changed during export.')
        keys=np.lib.format.open_memmap(work/'keys.npy',mode='w+',dtype=np.int32,shape=(raw,3))
        for lo in range(0,raw,1000000):
            block=np.floor(xyz[lo:lo+1000000].astype(np.float64)/voxel)
            if block.min()<np.iinfo(np.int32).min or block.max()>np.iinfo(np.int32).max:
                raise ValueError('Voxel grid integer overflow.')
            keys[lo:lo+len(block)]=block.astype(np.int32)
        unique,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        n=len(unique);del unique,keys
        dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
        vertex=np.lib.format.open_memmap(work/'vertex.npy',mode='w+',dtype=dtype,shape=(n,))
        for k,name in enumerate(('x','y','z')):
            vertex[name]=(np.bincount(inv,weights=xyz[:,k],minlength=n)/counts).astype(np.float32)
        for k,name in enumerate(('red','green','blue')):
            vertex[name]=np.rint(np.bincount(inv,weights=rgb[:,k],minlength=n)/counts).clip(0,255).astype(np.uint8)
        temporary=output/'panovggt_merged.partial.ply'
        with temporary.open('wb') as f:
            header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'
            header+=''.join('property float '+name+'\n' for name in ('x','y','z'))
            header+=''.join('property uchar '+name+'\n' for name in ('red','green','blue'))+'end_header\n'
            f.write(header.encode('ascii'));vertex.tofile(f)
        temporary.replace(output/'panovggt_merged.ply')
        del vertex,xyz,rgb,inv,counts
    trajectory.sort(key=lambda row:row['frame'])
    settings.update(fusion='Verified image/3D loop closures + robust global Sim(3) pose graph',
                    source_run=str(run.resolve()),global_optimization=True)
    report=dict(raw_points=raw,merged_points=n,voxel_size=voxel,input_frames=len(owner),windows=len(indices),
        pixel_step=settings['pixel_step'],export_seconds=time.perf_counter()-start,
        coordinate_units='arbitrary monocular scale',full_bundle_adjustment=False,
        depth_refinement=False,semantic_labels_must_be_recomputed=True)
    for name,value in [('camera_poses.json',trajectory),('settings.json',settings),
                       ('input_manifest.json',manifest),('summary.json',report)]:
        (output/name).write_text(json.dumps(value,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','input','masks','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--detect-only',action='store_true')
    args=parser.parse_args()
    transforms,report=optimize_run(args.run,args.input,args.masks,args.output,detect_only=args.detect_only)
    if not args.detect_only:
        report['dense_export']=export_dense(args.run,args.output,transforms)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
