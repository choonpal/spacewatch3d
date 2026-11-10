"""RGB-primary appearance areas; mesh is a shared-coordinate support only.

No class, fixed ID, coordinate or desired candidate count is an admission rule.
Physical added/moved/removed classification is not performed in this module.
"""
import argparse,json,hashlib,time
from pathlib import Path
from functools import lru_cache
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial import cKDTree

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def components(points,radius):
    parent=np.arange(len(points))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    if len(points):
        for a,b in sorted(cKDTree(points).query_pairs(radius)):
            a,b=find(a),find(b)
            if a!=b:parent[max(a,b)]=min(a,b)
    return [find(i) for i in range(len(points))]

def repeated_frames(rows,minimum_pixels=16,minimum_frames=2):
    """Different cube faces in one source frame cannot invent repetition."""
    counts=defaultdict(int);unique={}
    for i,r in enumerate(rows):
        key=(r['epoch'],str(r['frame_id']),r.get('view',str(i)))
        unique[key]=max(unique.get(key,0),int(r['pixels']))
    for (epoch,frame,view),pixels in unique.items():counts[(epoch,frame)]+=pixels
    result={str(e):sorted(f for (epoch,f),n in counts.items() if epoch==e and n>=minimum_pixels) for e in [0,1]}
    return result,any(len(fs)>=minimum_frames for fs in result.values())

def sample(a,n=256):return a[np.linspace(0,len(a)-1,min(n,len(a)),dtype=int)] if len(a) else a

def build(prepared_root,evidence_root,mask_root,local_root,output,threshold=.7):
    # Imports resolve only pinned upstream math, never a proposal-label fixture.
    from tools.mesh_simsac.core import project_in_view
    from tools.mesh_simsac.rgb_visibility import MeshVisibility
    root=Path(prepared_root);evidence=Path(evidence_root);masks=Path(mask_root);local=Path(local_root);out=Path(output)
    out.mkdir(exist_ok=False);(out/'samples').mkdir();(out/'observations').mkdir();started=time.time()
    prep=read(root/'manifest.json');settings=prep['settings'];replay=read(evidence/'evidence.json');pairs={r['query']:r for r in read(root/'pairs.json')};metadata={r['id']:r for r in prep['views']}
    probability_threshold=float(threshold);minimum_pixels=settings['min_object_pixels'];minimum_frames=settings['min_frames'];voxel=settings['voxel_size_u']
    if not .5<=probability_threshold<=1:raise ValueError('RGB threshold must not be lower than baseline0.5')
    visibility=MeshVisibility(root,prep,settings['visibility_tolerance_u']);gate_audit=Counter()
    error=read(root/'results/events.json')['policy']['registration_error_u'];radius=max(voxel,2*error)
    rows=replay['voxels'];points=np.array([(np.array(r['key'])+.5)*voxel for r in rows]);labels=components(points,radius)
    cell_to_group={(r['epoch'],*r['key']):int(g) for r,g in zip(rows,labels)}
    groups=defaultdict(lambda:dict(observations=[],points=[],regions=[],routes=set(),hints=Counter()))
    active={(o['epoch'],o['instance_id']) for o in replay['objects'] if o['state']=='change_candidate'}
    pooled=defaultdict(lambda:dict(observations=[],points=[],regions=[],routes={'surface_group_rgb'},hints=Counter()))
    @lru_cache(maxsize=24)
    def view(vid):
        with np.load(root/'views'/(vid+'.npz')) as z:q={k:z[k] for k in z.files}
        with np.load(masks/(vid+'.npz')) as z:q['person']=z['person']
        return q
    def observation(vid,mask,reference=None):
        q=view(vid);meta=metadata[vid];y,x=np.nonzero(mask)
        if not len(x):return None
        pts=q['points'][mask];reg=q['point_region'][mask];choices=[]
        for r in ([dict(id=reference)] if reference else pairs[vid]['refs']):
            ref=view(r['id']);uv,inside=visibility.test(pts,ref,reg,metadata[r['id']]['epoch'])
            if reference is None:
                with np.load(root/'results/pairs'/(vid+'--'+r['id']+'.npz')) as z:inside&=z['observable'][mask]
            if inside.sum()>=minimum_pixels:
                choices.append((int(inside.sum()),r['id'],inside,uv))
        if not choices:return None
        # Display reference is selected on geometric visibility, not maximum RGB.
        _,rid,inside,uv=max(choices,key=lambda r:(r[0],r[1]))
        saved=local/'pairs'/(vid+'--'+rid+'.npz')
        if not saved.exists():saved=root/'results/pairs'/(vid+'--'+rid+'.npz')
        with np.load(saved) as z:probability=z['probability'][mask]
        keep=inside&(probability>probability_threshold)
        gate_audit.update(source_pixel_observations=len(x),rejected_visibility=int((~inside).sum()),rejected_threshold=int((inside&~keep).sum()),retained=int(keep.sum()))
        if keep.sum()<minimum_pixels:return None
        # Prune pixels BEFORE samples, hints, bounds, frame votes and UI boxes.
        mask=np.zeros_like(mask);mask[y[keep],x[keep]]=True;y,x=np.nonzero(mask)
        pts,reg=pts[keep],reg[keep];a=uv[keep]
        refbbox=[int(a[:,0].min()),int(a[:,1].min()),int(a[:,0].max()+1),int(a[:,1].max()+1)]
        bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        x0,y0,x1,y1=bbox;rq=view(rid);a,b,c,d=refbbox
        contamination=max(float(q['person'][y0:y1,x0:x1].mean()),float(rq['person'][b:d,a:c].mean()))
        hints=Counter(q['segmentation_class'][mask].tolist());names=meta.get('segmentation_labels',[])
        hints={names[k-1] if 0<k<=len(names) else 'unknown':int(n) for k,n in hints.items()}
        index=sample(np.arange(len(pts)),256)
        support='observations/'+vid+'--'+rid+'--'+hashlib.sha256(mask.tobytes()).hexdigest()[:12]+'.npz'
        np.savez_compressed(out/support,mask=mask)
        return dict(view=vid,epoch=meta['epoch'],frame_id=meta['frame_id'],timestamp_s=meta['timestamp_s'],reference=rid,
            pixels=len(x),bbox=bbox,reference_bbox=refbbox,photographer_bbox_fraction=contamination,hints=hints,
            visible_support_mask=support,first_hit_checked=True,mean_probability=float(probability[keep].mean()),minimum_probability=float(probability[keep].min()),
            _points=pts[index],_regions=reg[index])
    # Geometry-supported RGB pixels, including unlabelled surface cells. Labels
    # are merely recorded after RGB/multiframe admission, not used as a gate.
    for n,meta in enumerate(prep['views']):
        vid=meta['id'];q=view(vid)
        with np.load(evidence/'evidence-views'/(vid+'.npz')) as z:changed=z['candidate']&~q['person']
        y,x=np.nonzero(changed);keys=np.floor(q['points'][changed]/voxel).astype(int)
        ids=np.array([cell_to_group.get((meta['epoch'],*map(int,k)),-1) for k in keys])
        for g in np.unique(ids[ids>=0]):
            mask=np.zeros(changed.shape,bool);selected=ids==g;mask[y[selected],x[selected]]=True
            row=observation(vid,mask)
            if row:
                groups[int(g)]['observations'].append(row);groups[int(g)]['routes'].add('registered_rgb_pixels')
        for iid in np.unique(q['instance'][changed]):
            key=(meta['epoch'],int(iid))
            if key not in active:continue
            mask=changed&(q['instance']==iid);row=observation(vid,mask)
            if row:pooled[key]['observations'].append(row)
        if n%120==0:print('PRIMARY_SPATIAL_RGB',n,flush=True)
    # Cache samples and discard spatial components without repeated image support.
    admitted=[]
    def finish_group(g):
        for r in g['observations']:
            g['points'].append(r['_points']);g['regions'].append(r['_regions']);g['hints'].update(r['hints'])
        frames,passed=repeated_frames(g['observations'],minimum_pixels,minimum_frames)
        if not passed:return None
        g['frames']=frames;g['_samples']=np.concatenate(g['points']);g['_regions']=np.concatenate(g['regions'])
        return g
    for key in sorted(groups):
        item=finish_group(groups[key])
        if item:admitted.append(item)
    # Fine voxel identities can drift within registration error. Pool repeatable
    # RGB support on an existing spatial surface group as an additional route.
    # This route never blocks the label-free fallback when no object exists.
    for key in sorted(pooled):
        g=finish_group(pooled[key])
        if not g:continue
        union=np.concatenate([a['_samples'] for a in admitted]) if admitted else np.empty((0,3))
        overlap=float((cKDTree(union).query(g['_samples'])[0]<=radius).mean()) if len(union) else 0.
        if overlap>=.35:continue
        admitted.append(g)
    # Independent RGB-region fallback: no foreground shape or class-label veto.
    # These are explicitly weaker position proxies, not reconstructed objects.
    fallback=read(evidence/'evaluation-threshold-0.5.json')['proposals']
    for proposal in fallback:
        g=dict(observations=[],points=[],regions=[],routes={'rgb_background_proxy'},hints=Counter())
        for source in proposal['observations']:
            vid=source['view'];rid=source['reference'];q=view(vid)
            saved=local/'pairs'/(vid+'--'+rid+'.npz')
            if not saved.exists():saved=root/'results/pairs'/(vid+'--'+rid+'.npz')
            with np.load(saved) as z:prob=z['probability']
            mask=(prob>probability_threshold)&q['mesh_hit']&~q['person'];x0,y0,x1,y1=source['bbox'];roi=np.zeros_like(mask);roi[y0:y1,x0:x1]=True;mask&=roi
            row=observation(vid,mask,rid)
            if row:g['observations'].append(row)
        g=finish_group(g)
        if not g:continue
        overlap=[float((cKDTree(a['_samples']).query(g['_samples'])[0]<=radius).mean()) for a in admitted]
        if overlap and max(overlap)>=.35:
            a=admitted[int(np.argmax(overlap))];a['observations'].extend(g['observations']);a['routes']|=g['routes'];a['hints'].update(g['hints'])
            a['_samples']=np.r_[a['_samples'],g['_samples']];a['_regions']=np.r_[a['_regions'],g['_regions']]
            a['frames'],_=repeated_frames(a['observations'],minimum_pixels,minimum_frames)
        else:admitted.append(g)
    candidates=[]
    for n,g in enumerate(admitted):
        key=f'RGB-{n+1:03d}';samples=g['_samples'];regions=g['_regions'];frames=g['frames'];clean=[r for r in g['observations'] if r['photographer_bbox_fraction']<=1-.6]
        _,clean_repetition=repeated_frames(clean,minimum_pixels,minimum_frames)
        geometry='registered_rgb_pixels' in g['routes'];quality='supported_rgb' if geometry and clean_repetition else 'rgb_proxy_candidate'
        filename=f'samples/{key}.npz';index=sample(np.arange(len(samples)),1024);np.savez_compressed(out/filename,points=samples[index],regions=regions[index])
        # Deduplicate display records; duplicate routes do not add source frames.
        observations={}
        for r in g['observations']:
            old=observations.get(r['view'])
            if old is None or old['pixels']<r['pixels']:observations[r['view']]=r
        observations=sorted(observations.values(),key=lambda r:(-r['pixels'],r['view']))
        candidates.append(dict(id=key,label='RGB 변화영역',type='appearance_candidate',physical_type='unresolved',
            quality=quality,localization='registered background/surface proxy, not reconstructed foreground shape',
            localization_approximate=not geometry,routes=sorted(g['routes']),frames=frames,
            observations=[{k:v for k,v in r.items() if not k.startswith('_')} for r in observations],
            bounds=np.stack([samples.min(0),samples.max(0)]).tolist(),samples=filename,
            semantic_hints=[dict(label=k,pixels=int(v)) for k,v in g['hints'].most_common()],
            labels_used_for_admission=False,physical_identity_verified=False))
    data=dict(schema='rgb-primary-appearance-v1',candidates=candidates,settings=dict(probability_threshold=probability_threshold,min_pixels=minimum_pixels,min_frames=minimum_frames,
        cluster_radius_u=radius,radius_origin='2timesexistinguncensoredregistrationerror; not fitted to requested examples'),
        source_root=str(root),evidence_root=str(evidence),mask_root=str(masks),local_pair_root=str(local),
        visibility_audit=dict(visibility.stats),pixel_gate_audit=dict(gate_audit),visibility_tolerance_u=settings['visibility_tolerance_u'],
        visibility_policy='Both reference image first-hit depth and full original mesh ray; hidden/unknown pixels are NOT compared and never vote no-change. Filter before RGB area/frame votes and projected boxes.',
        input_rgb_changed=False,geometry_changed=False,keypoint_plan_used_for_detection=False,physical_event_decisions_changed=False,
        policy='RGB pixel changes first, independent frames in at least one capture; geometry supports shared positions/occlusion only; labels optional hints. No physical change-type assertion.',
        seconds=time.time()-started,quality_status='UNVERIFIED')
    write(out/'candidates.json',data);write(out/'state.json',dict(state='complete',seconds=data['seconds'],candidates=len(candidates),
        supported=sum(c['quality']=='supported_rgb' for c in candidates),proxy=sum(c['quality']!='supported_rgb' for c in candidates)))
    print(json.dumps(read(out/'state.json')),flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--evidence',required=True);p.add_argument('--masks',required=True);p.add_argument('--local-pairs',required=True);p.add_argument('--out',required=True);p.add_argument('--threshold',type=float,default=.7);a=p.parse_args();build(a.prepared,a.evidence,a.masks,a.local_pairs,a.out,a.threshold)
