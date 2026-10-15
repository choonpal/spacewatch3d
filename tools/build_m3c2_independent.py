"""Fuse label-free M3C2 candidates, RGB-D visibility and geometric patch motion.

Existing events supply EXCLUSION boxes/frame IDs only. No object IDs, semantic
classes, or previous change types enter discovery, matching, or classification.
Output types are conditional geometric estimates; ambiguous evidence is retained
as 'suspected', never converted to a disappearance from a reconstruction hole.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from mesh_change_noise import (screen_surface_changes, patch_noise_decision, RADII,
    REGISTRATION_SIGMA_M, PATCH_FRACTION, MIN_SCALES, MIN_DISPLACEMENT_M)

ROTATION=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
MIN_POINTS=9
SUPPORTED_RGB_POSES=('estimated_rgb_checked','estimated_rgb_tracked','dataset_pose')


def exclusions(existing, padding=.15):
    boxes=[];frames=set()
    for event in existing['regions']:
        for box in (event.get('before_bounds'),event.get('after_bounds')):
            if box:boxes.append((np.array(box['min'])-padding,np.array(box['max'])+padding))
        for group in event.get('keyframes',[]):
            frames.update(item['frame_id'] for item in group)
    return boxes,frames


def outside_boxes(points, boxes):
    mask=np.ones(len(points),bool)
    for low,high in boxes:mask &= ~((points>=low)&(points<=high)).all(1)
    return mask


def box_of(points, padding=0.):
    return {'min':(points.min(0)-padding).tolist(),'max':(points.max(0)+padding).tolist()}


def intersects(box, excluded):
    low,high=np.array(box['min']),np.array(box['max'])
    return any(np.all(np.minimum(high,b)-np.maximum(low,a)>0) for a,b in excluded)


def split_components(points, indices, cell=.13, tile=1.10):
    if not len(indices):return []
    grid=np.floor(points[indices]/cell).astype(int);grid-=grid.min(0)
    shape=grid.max(0)+1
    if np.prod(shape)>8_000_000:raise ValueError('Candidate extent too large')
    occupied=np.zeros(shape,bool);occupied[tuple(grid.T)]=True
    components,_=label(occupied,np.ones((3,3,3)));membership=components[tuple(grid.T)]
    result=[]
    for component in range(1,int(membership.max())+1):
        ids=indices[membership==component]
        if len(ids)<MIN_POINTS:continue
        p=points[ids]
        if np.max(np.ptp(p,axis=0))>1.5:
            _,parts=np.unique(np.floor(p/tile).astype(int),axis=0,return_inverse=True)
            result.extend(ids[parts==i] for i in range(int(parts.max())+1) if np.sum(parts==i)>=MIN_POINTS)
        else:result.append(ids)
    return result


def ray_votes(points_world, frames):
    """Measured raw depth only: occupied/free/occluded/unknown, per point/view.

    Invalid depth, out-of-image points and surface edges are UNKNOWN. A valid
    return behind a query is required for free space. No mesh hole is evidence.
    """
    accepted=[];occupied=[];free=[];occluded=[]
    for frame in frames:
        pose=frame.get('depth_camera_to_world');path=frame.get('depth_path')
        if pose is None or not path:continue
        if frame.get('status')!='dataset_pose' and (frame.get('depth_fitness',0)<.65 or frame.get('depth_rmse_m',1)>.035):continue
        path=Path(path)
        if not path.is_file():continue
        depth=np.fromfile(path,dtype='<u2')
        width,height=frame.get('depth_size',[640,480])
        if depth.size!=width*height:continue
        depth=depth.reshape(height,width).astype(float)/frame.get('depth_units_per_meter',1000)
        pose=np.asarray(pose);camera=(points_world-pose[:3,3])@pose[:3,:3];z=camera[:,2]
        fx,fy,cx,cy=frame.get('depth_intrinsics',[570.5,570.5,319.5,239.5])
        pixels=camera[:,:2]/np.maximum(z[:,None],1e-9)*[fx,fy]+[cx,cy]
        inside=(z>.3)&(z<5.5)&(pixels[:,0]>3)&(pixels[:,0]<width-4)&(pixels[:,1]>3)&(pixels[:,1]<height-4)
        ids=np.flatnonzero(inside);xy=np.rint(pixels[inside]).astype(int)
        samples=np.stack([depth[xy[:,1]+dy,xy[:,0]+dx] for dx,dy in [(0,0),(-2,0),(2,0),(0,-2),(0,2)]],axis=1)
        valid_samples=(samples>.3)&(samples<6.)
        # Requiring every sample also suppresses depth discontinuities at edges.
        valid=valid_samples.all(1)&(np.ptp(samples,axis=1)<.08+.015*z[ids])
        measured=np.median(samples,axis=1);difference=measured-z[ids]
        occ=np.zeros(len(points_world),bool);fr=occ.copy();occl=occ.copy()
        tolerance=.065+.015*z[ids]
        occ[ids]=valid&(np.abs(difference)<=tolerance)
        fr[ids]=valid&(difference>.14+.015*z[ids])
        occl[ids]=valid&(difference < -tolerance)
        occupied.append(occ);free.append(fr);occluded.append(occl);accepted.append(frame)
    def stack(rows):return np.column_stack(rows) if rows else np.zeros((len(points_world),0),bool)
    return {'occupied':stack(occupied),'free':stack(free),'occluded':stack(occluded),'frames':accepted}


def region_observation(ids, votes):
    if not len(ids):return {'occupied':0.,'free':0.,'occluded':0.,'observed':0.,'free_views':0,'occupied_views':0,'baseline_m':0.}
    occupied,free,occluded=[votes[k][ids] for k in ('occupied','free','occluded')]
    free_views=np.flatnonzero(free.mean(0)>=.20) if free.shape[1] else []
    occupied_views=np.flatnonzero(occupied.mean(0)>=.20) if occupied.shape[1] else []
    positions=np.array([np.asarray(votes['frames'][i]['depth_camera_to_world'])[:3,3] for i in free_views])
    baseline=float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2))) if len(positions)>1 else 0.
    return {'occupied':float(np.mean(occupied.sum(1)>=2)),'free':float(np.mean(free.sum(1)>=2)),
            'occluded':float(np.mean(occluded.sum(1)>=2)),
            'observed':float(np.mean((occupied|free|occluded).sum(1)>0)),
            'free_views':len(free_views),'occupied_views':len(occupied_views),'baseline_m':baseline}


def classify_patch(own, opposite, *, significance, nearest_other):
    """Presence difference only after measured occupied/free evidence agrees."""
    if significance<.55:return 'suspected','구역 전체에서 M3C2 변화 기준을 안정적으로 넘지 못했습니다.'
    if nearest_other<.08:return 'suspected','가까운 표면이 양쪽에 남아 있어 재구성 노이즈와 구별하기 어렵습니다.'
    if own['occupied']<.40:return 'suspected','해당 표면을 지지하는 깊이 관측이 충분하지 않습니다.'
    if opposite['free']>=.55 and opposite['free_views']>=2 and opposite['baseline_m']>=.15:
        return 'presence_difference','반대 시점의 여러 깊이 관측에서 이 위치의 빈 공간이 관측됐습니다.'
    if opposite['occluded']>.25:return 'suspected','반대 시점에서 다른 표면에 가려져 변화 종류를 판단하기 어렵습니다.'
    return 'suspected','반대 시점의 같은 위치를 충분히 관측하지 못했습니다.'


def rigid_patch_fit(a,b):
    """Small geometry-only multi-start rigid ICP; planar patches stay ambiguous."""
    if min(len(a),len(b))<12:return None
    ca,cb=np.median(a,axis=0),np.median(b,axis=0)
    va=np.linalg.eigvalsh(np.cov((a-ca).T));vb=np.linalg.eigvalsh(np.cov((b-cb).T))
    if min(va[0]/max(va[-1],1e-12),vb[0]/max(vb[-1],1e-12))<.012:return None
    ea=np.sort(np.ptp(a,axis=0));eb=np.sort(np.ptp(b,axis=0))
    if np.max(np.maximum(ea,eb)/np.maximum(np.minimum(ea,eb),.06))>2.2:return None
    if not .12<np.linalg.norm(ca-cb)<3.:return None
    tree=cKDTree(b);best=None
    for angle in np.linspace(0,2*np.pi,8,endpoint=False):
        c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]]);t=cb-R@ca
        for _ in range(25):
            moved=a@R.T+t;distance,nearest=tree.query(moved)
            mask=distance<.16
            if mask.sum()<max(10,len(a)*.3):break
            x,y=moved[mask],b[nearest[mask]];xm,ym=x.mean(0),y.mean(0)
            u,_,vt=np.linalg.svd((x-xm).T@(y-ym));r=vt.T@u.T
            if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
            shift=ym-r@xm;R,t=r@R,r@t+shift
            if np.linalg.norm(shift)<1e-5 and np.linalg.norm(r-np.eye(3))<1e-4:break
        moved=a@R.T+t;forward=tree.query(moved)[0];back=cKDTree(moved).query(b)[0]
        overlap=float((np.mean(forward<.055)+np.mean(back<.055))/2)
        residual=float((np.median(forward)+np.median(back))/2)
        cost=residual/.04+2*(1-overlap)
        if best is None or cost<best['cost']:
            rotation=float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
            transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t
            best={'cost':cost,'overlap':overlap,'residual_m':residual,'rotation_deg':rotation,
                  'displacement_m':float(np.linalg.norm(R@ca+t-ca)),'before_to_after_display':transform.tolist()}
    return best if best and best['overlap']>.70 and best['residual_m']<.035 else None


def choose_motion_pairs(before,after,points):
    fits={}
    for i,a in enumerate(before):
        for j,b in enumerate(after):
            fit=rigid_patch_fit(points[a['ids']],points[b['ids']])
            if fit:fits[i,j]=fit
    accepted=[];ambiguous_before=set();ambiguous_after=set()
    for (i,j),fit in fits.items():
        row=sorted([(f['cost'],b) for (a,b),f in fits.items() if a==i]);col=sorted([(f['cost'],a) for (a,b),f in fits.items() if b==j])
        unique=(len(row)<2 or row[1][0]-row[0][0]>.25) and (len(col)<2 or col[1][0]-col[0][0]>.25)
        if row[0][1]==j and col[0][1]==i and unique:accepted.append((i,j,fit))
        else:ambiguous_before.add(i);ambiguous_after.add(j)
    return accepted,ambiguous_before,ambiguous_after


def integrate_patches(patches,points):
    """Classify surface appearance/disappearance, pair unique rigid motion,
    and retain uncertain or multiply-matched patches as suspected regions.
    """
    presence=[[p for p in group if p['decision']=='presence_difference'] for group in patches]
    pairs,ambiguous_before,ambiguous_after=choose_motion_pairs(*presence,points)
    claimed=set();drafts=[]
    for i,j,motion in pairs:
        a,b=presence[0][i],presence[1][j];claimed.update([id(a),id(b)])
        drafts.append({'type':'moved','parts':[a,b],'motion':motion,'reason':'전후의 사라진 표면과 나타난 표면이 고유한 강체 형상 대응을 보입니다.'})
    ambiguous={id(presence[0][i]) for i in ambiguous_before}|{id(presence[1][j]) for j in ambiguous_after}
    for epoch,group in enumerate(patches):
        for patch in group:
            if id(patch) in claimed:continue
            kind=('removed' if epoch==0 else 'added') if patch['decision']=='presence_difference' and id(patch) not in ambiguous else 'suspected'
            reason=patch['reason'] if id(patch) not in ambiguous else '전후의 비슷한 표면 대응이 여러 개여서 이동 여부가 불확실합니다.'
            drafts.append({'type':kind,'parts':[patch],'motion':None,'reason':reason})
    return drafts,pairs,presence


def connect_photos(points_world, epoch, frames, mesh, visibility_cache):
    import open3d as o3d
    from recover_rescan_keyframes import render
    links=[]
    for frame in frames:
        if frame['epoch']!=epoch:continue
        checked=frame['status'] in SUPPORTED_RGB_POSES
        pose=frame.get('camera_to_world') if checked else frame.get('depth_camera_to_world')
        if pose is None:continue
        pose=np.asarray(pose);focal=frame.get('focal_px',585.225)
        camera=(points_world-pose[:3,3])@pose[:3,:3];z=camera[:,2]
        if frame['status']=='dataset_pose':
            # Compare against the measured, undistorted depth image; project
            # the overlay back to the untouched distorted original RGB.
            import cv2
            fx,fy,cx,cy=frame['depth_intrinsics'];width,height=frame['width'],frame['height']
            ideal=camera[:,:2]/np.maximum(z[:,None],1e-9)*[fx,fy]+[cx,cy]
            K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            pixels=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),K,np.array(frame.get('rgb_distortion',[0.]*5)))[0].reshape(-1,2)
            inside=(z>.3)&(ideal[:,0]>3)&(ideal[:,0]<width-4)&(ideal[:,1]>3)&(ideal[:,1]<height-4)&(pixels[:,0]>3)&(pixels[:,0]<width-4)&(pixels[:,1]>3)&(pixels[:,1]<height-4)
            if inside.sum()<5:continue
            if frame['id'] not in visibility_cache:
                visibility_cache[frame['id']]=np.fromfile(frame['depth_path'],dtype='<u2').reshape(height,width)/frame.get('depth_units_per_meter',1000)
            xy=np.rint(ideal[inside]).astype(int);surface=visibility_cache[frame['id']][xy[:,1],xy[:,0]]
            visible=(surface>.3)&(surface>=z[inside]-.08)
            if visible.sum()<5:continue
            relevant=pixels[inside][visible];fraction=float(visible.sum()/len(points_world));size=np.ptp(relevant,axis=0)
            if fraction<.08 or min(size)<6:continue
            links.append({'frame_id':frame['id'],'score':float(fraction*np.sqrt(np.prod(size)/(width*height))),
                'bbox_px':np.r_[relevant.min(0),relevant.max(0)].tolist(),'visible_sample_fraction':fraction,
                'camera_check':'dataset_ground_truth','relation':'independent_surface_region','absence_confirmed':False})
            continue
        pixels=camera[:,:2]/np.maximum(z[:,None],1e-9)*focal+[323.5,241.5]
        inside=(z>.3)&(pixels[:,0]>16)&(pixels[:,0]<632)&(pixels[:,1]>16)&(pixels[:,1]<468)
        if inside.sum()<5:continue
        if frame['id'] not in visibility_cache:
            _,_,depth=render(mesh,pose,focal);visibility_cache[frame['id']]=depth
        depth=visibility_cache[frame['id']];xy=np.rint(pixels[inside]).astype(int)
        surface=depth[xy[:,1],xy[:,0]]
        visible=np.isfinite(surface)&(surface>=z[inside]-.12)
        if visible.sum()<5:continue
        fraction=float(visible.sum()/len(points_world))
        if fraction<.08:continue
        relevant=pixels[inside][visible];size=np.ptp(relevant,axis=0)
        if min(size)<6:continue
        score=fraction*np.sqrt(float(np.prod(size)/(648*484)))
        links.append({'frame_id':frame['id'],'score':float(score),'bbox_px':np.r_[relevant.min(0),relevant.max(0)].tolist() if checked else None,
                      'visible_sample_fraction':fraction,'camera_check':'rgb_temporal_mesh_pnp' if checked else 'depth_registration_only',
                      'relation':'independent_surface_region','absence_confirmed':False})
    links.sort(key=lambda x:(-int(x['camera_check']=='rgb_temporal_mesh_pnp'),-x['score']))
    return links[:4]


def build(comparison_dir, review_dir, frames_dir, output):
    import open3d as o3d
    comparison=json.loads((comparison_dir/'comparison.json').read_text(encoding='utf8'))
    old=json.loads((review_dir/'changes.json').read_text(encoding='utf8'))
    review=json.loads((review_dir/'review.json').read_text(encoding='utf8'))
    cameras=json.loads((frames_dir/'cameras.json').read_text(encoding='utf8'))
    if review['scene']!=cameras['scene'] or review['source_epochs']!=cameras['source_epochs']:raise ValueError('Capture mismatch')
    output.mkdir(parents=True,exist_ok=True)
    meshes=[o3d.io.read_triangle_mesh(str(review_dir/f'cache/{e}-depth8.ply')) for e in (0,1)]
    hashes=[hashlib.sha256((review_dir/f'cache/{e}-depth8.ply').read_bytes()).hexdigest() for e in (0,1)]
    if hashes!=comparison['input_hashes']:raise ValueError('M3C2 cache does not describe these meshes')
    core=np.load(comparison_dir/'core-fields.npz');points=core['xyz'];m3=core['m3c2'];classes=core['classes'][:,0]
    rotation=np.array(comparison.get('display_rotation',ROTATION))
    origin=np.array(comparison['display_origin']);absolute=points+origin;world=absolute@rotation
    excluded,old_frames=exclusions(old)
    independent=outside_boxes(absolute,excluded)
    candidate=np.isin(classes,[1,2])&independent
    scenes=[];distances=[]
    for mesh in meshes:
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));scenes.append(scene)
        distances.append(scene.compute_distance(o3d.core.Tensor(world.astype('f4'))).numpy())
    distances=np.column_stack(distances)
    candidate &= distances.max(1)>.045
    print('Nominal M3C2 candidate cores',int(candidate.sum()),'from',len(points),flush=True)
    # A second screen uses spatial surface variation, not the standard error
    # of arbitrarily dense mesh samples. It never consumes object labels.
    from compare_mesh_methods import surface_samples, pca_normals
    samples=[]
    for epoch,mesh in enumerate(meshes):
        sample,normal=surface_samples(mesh,20261015+epoch)
        samples.append((sample@rotation.T-origin,normal@rotation.T))
    owner=distances.argmin(1);candidate_indices=np.flatnonzero(candidate)
    normals=np.zeros((len(candidate_indices),3))
    for epoch in (0,1):
        mask=owner[candidate_indices]==epoch
        normals[mask],_=pca_normals(points[candidate_indices[mask]],*samples[epoch])
    print('Computing robust surface differences at radii',RADII,flush=True)
    partial=screen_surface_changes(points[candidate_indices],normals,[s[0] for s in samples])
    screen={key:np.full((len(points),)+value.shape[1:],False if value.dtype==bool else np.nan,
                       dtype=bool if value.dtype==bool else float)
            for key,value in partial.items() if key!='radii_m'}
    for key,value in screen.items():value[candidate_indices]=partial[key]
    screen['radii_m']=partial['radii_m']
    print('Multiscale persistent cores',int(screen['persistent'].sum()),flush=True)
    # Recovered cameras/depth poses and raw depth only; none of their instance
    # or semantic metadata is read. Existing frames may support ray evidence,
    # but displayed keyframes must not occur in an old event's frame list.
    observation_frames=[]
    for frame in review['keyframes']+cameras['frames']:
        frame=frame.copy()
        if frame.get('depth_path'):
            path=Path(frame['depth_path'])
            if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
            frame['depth_path']=str(path)
        observation_frames.append(frame)
    votes=[ray_votes(world,[f for f in observation_frames if f['epoch']==e]) for e in (0,1)]
    patches=[[],[]];suppressed=[];accepted_candidate=np.zeros(len(points),bool)
    for epoch in (0,1):
        indices=np.flatnonzero(candidate&(owner==epoch)&(distances[:,epoch]<.045))
        for ids in split_components(points,indices):
            box=box_of(absolute[ids],.025)
            if intersects(box,excluded):continue
            noise=patch_noise_decision(screen,ids)
            if not noise['keep']:
                suppressed.append({'bounds':box,'epoch':epoch,'core_count':len(ids),
                    'raw_m3c2_significant_fraction':float(np.mean(classes[ids]==2)),
                    'raw_m3c2_median_distance_m':float(np.median(np.abs(m3[ids,1]))),**noise})
                continue
            original_ids=ids
            ids=ids[screen['persistent'][ids]]
            if len(ids)<MIN_POINTS:continue
            own=region_observation(ids,votes[epoch]);opposite=region_observation(ids,votes[1-epoch])
            noise=patch_noise_decision(screen,original_ids,own)
            if not noise['keep']:
                suppressed.append({'bounds':box,'epoch':epoch,'core_count':len(original_ids),
                    'raw_m3c2_significant_fraction':float(np.mean(classes[original_ids]==2)),
                    'raw_m3c2_median_distance_m':float(np.median(np.abs(m3[original_ids,1]))),
                    'own_observation':own,**noise})
                continue
            accepted_candidate[ids]=True
            box=box_of(absolute[ids],.025)
            significance=noise['persistent_fraction'];near=float(np.median(distances[ids,1-epoch]))
            decision,reason=classify_patch(own,opposite,significance=significance,nearest_other=near)
            patches[epoch].append({'ids':ids,'epoch':epoch,'bounds':box,'own':own,'opposite':opposite,
                'decision':decision,'reason':reason,'significance':significance,'nearest_other_m':near,
                'original_ids':original_ids,'noise':noise})
    print('Suppressed surface patches',len(suppressed),'retained patches',[len(p) for p in patches],flush=True)
    print('Geometry patches',[len(p) for p in patches],'presence evidence',[sum(p['decision']=='presence_difference' for p in group) for group in patches],flush=True)
    drafts,pairs,presence=integrate_patches(patches,points)
    # A suspect at the same location in both epochs is one region, not two
    # different additions/removals. Conservative spatial deduplication only.
    drafts.sort(key=lambda d:(d['type']=='suspected',-sum(len(p['ids']) for p in d['parts'])))
    kept=[]
    for draft in drafts:
        ids=np.unique(np.concatenate([p['ids'] for p in draft['parts']]))
        center=np.mean(points[ids],axis=0)
        if draft['type']=='suspected' and any(np.linalg.norm(center-other['center'])<.35 for other in kept):continue
        draft.update(ids=ids,center=center);kept.append(draft)
    print('After duplicate removal',len(kept),'regions',flush=True)
    photo_frames=[f.copy() for f in cameras['frames'] if f['status'] in SUPPORTED_RGB_POSES and f['id'] not in old_frames]
    # Weak un-used original frames may show a region even if RGB localization
    # failed. Such links carry no overlay and demote event certainty below.
    for frame in review['keyframes']:
        if frame['id'] not in old_frames and frame.get('depth_camera_to_world'):
            f=frame.copy();f['url']='/m3c2-independent/'+f['id']+'.jpg';photo_frames.append(f)
    visibility={};events=[];event_core_ids=[];unlinked=[]
    for draft in kept:
        points_by_epoch=[];bounds_by_epoch=[]
        for e in (0,1):
            part=next((p for p in draft['parts'] if p['epoch']==e),None)
            ids=part['ids'] if part else draft['ids'];points_by_epoch.append(world[ids]);bounds_by_epoch.append(part['bounds'] if part else None)
        links=[connect_photos(points_by_epoch[e],e,photo_frames,meshes[e],visibility) for e in (0,1)]
        if not all(links):
            unlinked.append({'bounds':box_of(absolute[draft['ids']],.025),'proposed_type':draft['type'],
                'classification_reason':draft['reason'],'status':'paired_original_frames_unavailable',
                'photo_candidate_counts':[len(group) for group in links],'core_count':len(draft['ids'])})
            continue
        checked=all(any(link['camera_check'] in ('rgb_temporal_mesh_pnp','dataset_ground_truth') for link in group) for group in links)
        kind=draft['type'];reason=draft['reason']
        if kind!='suspected' and not checked:
            kind='suspected';reason='전후 원본 사진의 위치 대응이 약해 변화 종류를 확정하지 않았습니다.'
        # Retain all original conditional decisions in evidence for review.
        number=len(events)+1;ids=draft['ids'];event_id=1000+number
        by_type={'added':'추가','removed':'소실','moved':'이동','suspected':'의심 구역'}
        motion=draft['motion']
        if kind!='moved':motion=None
        event={'id':event_id,'name':f'M3C2 {by_type[kind]} {number:02d}','type':kind,'source':'m3c2_independent',
               'label_independent':True,'status':'uncertain_geometry' if kind=='suspected' else 'estimated_geometry_change',
               'bounds':box_of(absolute[ids],.025),'before_bounds':bounds_by_epoch[0],'after_bounds':bounds_by_epoch[1],
               'motion':motion,'keyframes':links,'paired_photos':True,'paired_rgb_checked':checked,
               'absence_confirmed':False,'reason':reason,'candidate_id':number,
               'classification_evidence':{'geometry_proposal':draft['type'],'m3c2_significant_fraction':float(np.mean(classes[ids]==2)),
                    'm3c2_median_distance_m':float(np.median(np.abs(m3[ids,1]))),
                    'm3c2_median_ratio':float(np.median(m3[ids,0])),'core_count':len(ids),
                    'epoch_observations':[{'epoch':p['epoch'],'own':p['own'],'opposite':p['opposite']} for p in draft['parts']],
                    'noise_filter':{'persistent_fraction':float(np.mean(screen['persistent'][np.concatenate([p['original_ids'] for p in draft['parts']])])),
                        'median_envelope_m':float(np.nanmedian(screen['envelope_m'][ids])),
                        'median_ratio':float(np.nanmedian(screen['ratio'][ids])),
                        'median_distance_m':float(np.nanmedian(np.abs(screen['distance_m'][ids]))) }},
               'overlap_check':{'existing_region_core_overlap':0,'existing_display_frame_overlap':0}}
        events.append(event);event_core_ids.append(ids)
        print('Linked',event['name'],kind,[[l['frame_id'] for l in g[:1]] for g in links],flush=True)
        if len(events)>=28:break
    # An empty detection result is valid. Do not resurrect rejected noise or
    # fall back to existing labeled events just to populate this source.
    used_new={link['frame_id'] for event in events for group in event['keyframes'] for link in group}
    if used_new&old_frames:raise AssertionError('Existing frame leaked into independent events')
    published_frames=[]
    for frame in photo_frames:
        if frame['id'] not in used_new:continue
        source=(frames_dir if frame.get('new_frame') else review_dir)/(frame['id']+'.jpg')
        shutil.copy2(source,output/source.name)
        published_frames.append({k:v for k,v in frame.items() if k not in ('depth_path','depth_camera_to_world')})
    candidate_labels=np.zeros(len(points),np.uint16)
    for number,ids in enumerate(event_core_ids,1):candidate_labels[ids]=number
    tree=cKDTree(points);mask_files=[]
    for epoch in (0,1):
        packed=np.fromfile(comparison_dir/f'{epoch}-comparison.bin',dtype=np.dtype([('xyz','<f4',(3,)),('rest','u1',(48,))]))
        position=packed['xyz'].astype(float);distance,index=tree.query(position,workers=4)
        membership=candidate_labels[index].copy();membership[distance>.14]=0
        membership[~outside_boxes(position+origin,excluded)]=0
        # Candidate footprint is the detected surface, not its rectangular box.
        for number,event in enumerate(events,1):
            box=event['before_bounds'] if epoch==0 else event['after_bounds']
            box=box or event['bounds'];low=np.array(box['min'])-origin-.05;high=np.array(box['max'])-origin+.05
            membership[(membership==number)&~((position>=low)&(position<=high)).all(1)]=0
        name=f'{epoch}-region-ids.bin';membership.astype('<u2').tofile(output/name)
        mask_files.append({'url':'/m3c2-independent/'+name,'count':len(membership),'stride':2})
    manifest={'kind':'m3c2_independent_changes','version':2,'scene':review['scene'],'source_epochs':review['source_epochs'],
              'method':'M3C2 + multiscale robust reconstruction-noise screen + measured RGB-D visibility + unique rigid patch correspondence',
              'label_independent':True,'classification_uses_instance_ids':False,'existing_events_usage':'exclusion boxes and displayed-frame IDs only',
              'source_hashes':hashes,'comparison_core_sha256':hashlib.sha256((comparison_dir/'core-fields.npz').read_bytes()).hexdigest(),
              'keyframes':published_frames,'regions':events,'region_masks':mask_files,'unlinked_candidates':unlinked,
              'suppressed_regions':suppressed,
              'noise_filter':{'radii_m':list(RADII),'minimum_consistent_scales':MIN_SCALES,
                   'minimum_displacement_m':MIN_DISPLACEMENT_M,'minimum_patch_fraction':PATCH_FRACTION,
                   'registration_sigma_per_map_m':REGISTRATION_SIGMA_M,
                   'minimum_source_depth_occupied_fraction':.40,
                   'spread':'max(1.4826*MAD, (P90-P10)/2.563) of nearest supported normal-direction layer',
                   'envelope':'1.96 * sqrt(spread_before^2 + spread_after^2 + 2*registration_sigma^2); no sample-count division',
                   'probability_calibrated':False,'suppressed_patch_count':len(suppressed)},
              'summary':{kind:sum(e['type']==kind for e in events) for kind in ('added','removed','moved','suspected')},
              'exclusion':{'padding_m':.15,'original_events':len(old['regions']),'excluded_original_frame_count':len(old_frames),
                           'candidate_core_overlap':int(np.sum((candidate_labels>0)&~independent)),
                           'display_frame_overlap':len(used_new&old_frames)},
              'parameters':{'candidate_source':'M3C2 classes 1/2, other mesh distance > .045m, multiscale robust noise screen','component_voxel_m':.13,
                  'split_tile_m':1.10,'minimum_cores':MIN_POINTS,'occupied_tolerance':'0.065 + .015*z meters',
                  'free_space_margin':'0.14 + .015*z meters','presence_fraction':.55,'own_occupied_fraction':.40,
                  'minimum_distinct_free_views':2,'minimum_free_view_baseline_m':.15,
                  'motion_minimum_overlap':.70,'motion_maximum_median_residual_m':.035,'motion_unique_cost_margin':.25},
              'limitations':['Camera poses and intrinsics are estimates; labels are conditional, not ground-truth decisions.',
                  'Regions are geometric surface patches, not guaranteed whole objects.',
                  'The conservative noise screen may miss small changes or retain systematic alignment drift; it is not calibrated M3C2-EP.',
                  'Occlusion, insufficient views, flat/repeated geometry and ambiguous motion remain suspected.',
                  'No annotated object ID/class/change type is consumed by discovery or classification.'],
              'diagnostics':{'raw_m3c2_candidates':int(np.isin(classes,[1,2]).sum()),'outside_existing_candidate_cores':int(candidate.sum()),
                   'patches_by_epoch':[len(p) for p in patches],'presence_proposals':[len(p) for p in presence],
                   'multiscale_persistent_cores':int(screen['persistent'].sum()),'accepted_candidate_cores':int(accepted_candidate.sum()),
                   'suppressed_patches':len(suppressed),
                   'unique_motion_pairs':len(pairs),'linked_regions':len(events),'new_original_frames':len(published_frames)}}
    (output/'changes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    np.savez_compressed(output/'candidate-fields.npz',xyz=points,m3c2=m3,independent_mask=independent,
        raw_candidate_mask=candidate,candidate_mask=accepted_candidate,region_id=candidate_labels,
        noise_persistent=screen['persistent'],noise_distance_m=screen['distance_m'],noise_envelope_m=screen['envelope_m'])
    print(json.dumps({'summary':manifest['summary'],'diagnostics':manifest['diagnostics'],'overlap':manifest['exclusion']},ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir',type=Path,required=True);parser.add_argument('--review-dir',type=Path,required=True)
    parser.add_argument('--frames-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a new output directory')
    build(args.comparison_dir,args.review_dir,args.frames_dir,args.output)
