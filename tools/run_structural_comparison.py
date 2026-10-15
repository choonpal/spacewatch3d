"""Common floor/wall residual references, candidate exclusion and original RGB."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from compare_mesh_methods import surface_samples, PACKED
from build_m3c2_independent import ray_votes, connect_photos, SUPPORTED_RGB_POSES
from run_residual_roi_comparison import summary, load, save
from empirical_residual import feature_summary, fixed_control
from mesh_change_noise import selected_surface_support
from structural_reference import (extract_planes, match_planes, background_assignment,
    plane_basis, oriented_measurements, calibrate_plane, classify_targets,candidate_components)


def bounds(points):
    return {'min':points.min(0).tolist(),'max':points.max(0).tolist()}


def run(comparison_dir, review_dir, frames_dir, previous_dir, output):
    import open3d as o3d
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True);published=output/'published';published.mkdir()
    source=load(comparison_dir/'comparison.json');review=load(review_dir/'review.json');cameras=load(frames_dir/'cameras.json')
    previous=load(previous_dir/'comparison.json');old=np.load(previous_dir/'measurements.npz')
    config={**previous['config'],'rois':[],'structural_reference':True}
    q=np.load(comparison_dir/'core-fields.npz')['xyz'];origin=np.array(source['display_origin']);rotation=np.array(source['display_rotation']);p=q+origin;world=p@rotation
    if not np.allclose(p,old['xyz'],atol=1e-12) or review['scene']!=cameras['scene'] or review['source_epochs']!=cameras['source_epochs']:
        raise ValueError('Input coordinate/capture mismatch')
    meshes=[o3d.io.read_triangle_mesh(str(review_dir/f'cache/{e}-depth8.ply')) for e in (0,1)]
    hashes=[hashlib.sha256((review_dir/f'cache/{e}-depth8.ply').read_bytes()).hexdigest() for e in (0,1)]
    if hashes!=source['input_hashes'] or hashes!=previous['source_hashes']:
        raise ValueError('Original aligned mesh hashes changed')
    frames=[]
    for frame in review['keyframes']+cameras['frames']:
        f=frame.copy()
        if f.get('depth_path') and not Path(f['depth_path']).is_absolute():
            f['depth_path']=str(Path(__file__).resolve().parents[1]/f['depth_path'])
        if not any(g['id']==f['id'] for g in frames):frames.append(f)
    photo_frames=[f for f in frames if f['status'] in SUPPORTED_RGB_POSES]
    samples=[];surface_normals=[];planes=[]
    for e,mesh in enumerate(meshes):
        sample,normal=surface_samples(mesh,20261015+e)
        samples.append(sample@rotation.T);surface_normals.append(normal@rotation.T)
        planes.append(extract_planes(samples[-1],surface_normals[-1],20261016+e))
        print('Extracted epoch',e,[(a['kind'],len(a['sample_ids']),round(a['support_grid_area_m2'],2),a['center']) for a in planes[-1]],flush=True)
    pairs=match_planes(planes,samples)
    print('Matched planes',[(a['id'],a['overlap_fraction'],a['offset_gap_m']) for a in pairs],flush=True)
    def public_plane(a):return {k:v for k,v in a.items() if k!='sample_ids'}
    save(output/'plane-extraction.json',{'epochs':[[public_plane(a) for a in group] for group in planes],
        'pairs':[{**{k:v for k,v in pair.items() if k!='planes'},'planes':[public_plane(a) for a in pair['planes']]} for pair in pairs]})
    normals=old['normals'].copy()
    reference_background=background_assignment(p,normals,pairs,samples)
    background=background_assignment(p,normals,pairs,samples,require_both=False)
    models=[];reference_measurements=[];photo_cache={};plane_links={};support_audit=[]

    def paired_observation(measured,points):
        observed=[]
        for e in (0,1):
            centers=points[:,None,:]+measured['normals'][:,None,:]*measured['centers_m'][:,:,e,None]
            votes=ray_votes(centers.reshape(-1,3)@rotation,[f for f in frames if f['epoch']==e])
            observed.append((votes['occupied'].sum(1)>=2).reshape(len(points),-1))
        return observed[0]&observed[1]

    for k,pair in enumerate(pairs):
        ids=np.flatnonzero(reference_background==k);n=np.tile(pair['normal'],(len(ids),1))
        subsets=[samples[e][a['sample_ids']] for e,a in enumerate(pair['planes'])]
        subset_normals=[surface_normals[e][a['sample_ids']] for e,a in enumerate(pair['planes'])]
        measured=oriented_measurements(p[ids],n,subsets,subset_normals)
        paired=paired_observation(measured,p[ids])
        np.savez_compressed(output/(pair['id']+'-reference.npz'),core_ids=ids,xyz=p[ids],uv=p[ids]@np.array(pair['basis']),
            distance_m=measured['distance_m'],counts=measured['counts_by_epoch'],paired_observed=paired)
        valid=np.isfinite(measured['distance_m'])&(measured['counts_by_epoch'].min(-1)>=6)&paired
        model=calibrate_plane(p[ids]@np.array(pair['basis']),measured['distance_m'],valid)
        model.update({'id':pair['id'],'kind':pair['kind'],'normal':pair['normal'],
            'raw_roi_core_count':len(ids),'common_support_grid_area_m2':pair['common_support_grid_area_m2'],
            'overlap_fraction':pair['overlap_fraction'],'bounds':bounds(p[ids]) if len(ids) else None,
            'fit_core_ids':ids[model['fit_mask']].tolist(),'check_core_ids':ids[model['check_mask']].tolist()})
        # Audit the actual selected plane samples, not just query footprints.
        shared=[]
        for e in (0,1):
            masks=[]
            for role in ('fit_mask','check_mask'):
                per_radius=[]
                for j,radius in enumerate(config['radii_m']):
                    selected=model[role]&valid[:,j]
                    per_radius.append(selected_surface_support(p[ids[selected]],n[selected],subsets[e],[radius])[0])
                masks.append(per_radius)
            shared.append([len(a&b) for a,b in zip(*masks)])
        support_audit.append({'plane':pair['id'],'shared_samples_by_epoch_radius':shared})
        if any(any(a) for a in shared):
            model['enabled']=False;model['check_reason']='fit_check_surface_support_overlap'
        # Only original RGBs from this pair; a missing photo is shown explicitly.
        display_ids=ids[::max(1,len(ids)//450)]
        links=[connect_photos(world[display_ids],e,photo_frames,meshes[e],photo_cache) if len(display_ids)>=5 else [] for e in (0,1)]
        plane_links[pair['id']]=links;model['paired_originals']=bool(all(links))
        model['checks'][0].update({'roi_id':pair['id']+'-check','total_cores':int(model['check_mask'].sum()),
            'valid_fraction':1. if model['check_mask'].any() else 0.,'reason':model['check_reason']})
        for name in ('fit_mask','check_mask'):model.pop(name)
        models.append(model);reference_measurements.append((ids,measured,paired))
        print('Plane calibration',model['id'],'fit',model['fit_core_count'],'check',len(model['check_core_ids']),
            'P95',model['thresholds_m'],'exceed',model['checks'][0]['exceedance_fraction'],'enabled',model['enabled'],flush=True)
    save(output/'reference-models.json',models);save(output/'fit-check-support-audit.json',support_audit)
    # Choose a parallel, nearby observed reference plane. All queries assigned
    # to one plane use that same direction as its reference measurements.
    reference=np.full((len(p),3),np.nan);bias=np.full((len(p),3),np.nan);model_index=np.full(len(p),-1,int);best=np.full(len(p),np.inf)
    original_normals=normals.copy()
    for k,(pair,model) in enumerate(zip(pairs,models)):
        if not model['enabled']:continue
        fit_ids=np.array(model['fit_core_ids'],int)
        distance=cKDTree(p[fit_ids]).query(p,workers=4)[0]
        compatible=(np.abs(original_normals@np.array(pair['normal']))>=np.cos(np.deg2rad(20)))&(distance<=2.5)
        take=compatible&(distance<best)
        model_index[take]=k;best[take]=distance[take]
        normal=np.array(pair['normal']);normals[take]=normal
        reference[take]=np.array(model['thresholds_m'])
        design=np.column_stack([np.ones(take.sum()),p[take]@np.array(pair['basis'])])
        bias[take]=design@np.array(model['bias_coefficients_m']).T
    print('Measuring furniture/remaining surfaces with normal-gated neighborhoods',flush=True)
    measured=oriented_measurements(p,normals,samples,surface_normals)
    paired=paired_observation(measured,p)
    empirical=classify_targets(measured,reference,paired,background,bias)
    fixed=fixed_control({'distance_m':old['distance_m'],'spreads_m':old['spreads_m']},old['paired_observed'])
    # Structural surfaces retain plane-only measurements even at furniture
    # contacts. They are cyan background, never a target measurement.
    for k,(ids,m,obs) in enumerate(reference_measurements):
        measured['distance_m'][ids]=m['distance_m'];measured['spreads_m'][ids]=m['spreads_m'];paired[ids]=obs
    areas=[];geometry=[];vertex_maps=[];tree=cKDTree(q);prefix='/residual-comparison/'
    for e,display in enumerate(source['displays']):
        name=f'{e}-comparison.bin';packed=np.fromfile(comparison_dir/name,dtype=PACKED)
        xyz=packed['xyz'].astype(float);d,index=tree.query(xyz,workers=4)
        tri=xyz.reshape(-1,3,3);face_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        weights=np.repeat(face_area/3,3);weights[d>.14]=0
        areas.append(np.bincount(index,weights=weights,minlength=len(q)))
        shutil.copy2(comparison_dir/name,published/name);geometry.append({**display,'url':prefix+name})
        vertex_maps.append((packed,xyz,d,index))
    areas=np.array(areas)
    groups=candidate_components(p,empirical,areas)
    for e,(packed,xyz,d,index) in enumerate(vertex_maps):
        for mode,result,distances in [('fixed',fixed,old['distance_m']),('empirical',empirical,empirical['corrected_distance_m'])]:
            stage=result['status'][index].astype(float);stage[d>.14]=2
            # Exclusion is also checked at the actual mesh vertex, preventing
            # a floor core from painting the bottom of an adjacent chair cyan.
            if mode=='empirical':
                nearest_background=background[index]>=0
                near_ids=np.flatnonzero(nearest_background)
                vertex_assignment=background_assignment(xyz[near_ids]+origin,packed['normal'][near_ids],pairs,samples,maximum_distance=.07,require_both=False)
                stage[near_ids[vertex_assignment<0]]=2
            thresholds=result['threshold_m'];t=np.full(len(p),np.nan);supported=np.isfinite(thresholds).any(1)
            t[supported]=np.nanmedian(thresholds[supported],axis=1)
            distance=np.full(len(p),np.nan);good=np.isfinite(distances).any(1);distance[good]=np.nanmedian(np.abs(distances[good]),axis=1)
            ratio=np.divide(distance,t,out=np.zeros(len(p)),where=np.isfinite(t)&(t>0))
            np.column_stack([np.nan_to_num(distance[index]),np.nan_to_num(t[index]),stage,ratio[index]]).astype('<f4').tofile(published/f'{e}-{mode}.bin')
    regions=[];used_frames=set()
    def add_region(identifier,title,kind,ids,links=None,model=None):
        if len(ids)<5:return
        if links is None:
            subset=ids[::max(1,len(ids)//250)]
            links=[connect_photos(world[subset],e,photo_frames,meshes[e],photo_cache) for e in (0,1)]
        for group in links:used_frames.update(a['frame_id'] for a in group)
        statistics=[summary(ids,result,p,areas,config) for result in (fixed,empirical)]
        if kind=='background':
            statistics[1].update({'status':'background','background_cores':len(ids),'reason':'common_structural_surface_excluded'})
        regions.append({'id':identifier,'name':title,'title':title,'kind':kind,'bounds':bounds(p[ids]),
            'keyframes':links,'paired_originals':bool(all(links)),'statistics':statistics,
            'distance_by_radius_m':[feature_summary(np.abs(measured['distance_m'][ids,j])) for j in range(3)],
            'corrected_distance_by_radius_m':[feature_summary(np.abs(empirical['corrected_distance_m'][ids,j])) for j in range(3)],
            'reasons':{r:int((empirical['reason'][ids]==r).sum()) for r in np.unique(empirical['reason'][ids])},
            'conditions':{'roughness_m':feature_summary(np.median(np.max(measured['spreads_m'][ids],axis=-1),axis=1))},
            'reference_plane':model['id'] if model else None})
    for i,ids in enumerate(groups):
        add_region('candidate-'+str(i+1),f'변화 후보 {i+1:02d} · 연결 표면','local_patch',ids)
    # Keep a compact object-sized browsing list; no change/instance labels read.
    tiles=np.floor((p-p.min(0))/.8).astype(int);_,membership=np.unique(tiles,axis=0,return_inverse=True)
    patch_ids=[np.flatnonzero((membership==i)&(background<0)) for i in np.unique(membership)]
    patch_ids=[a for a in patch_ids if len(a)>=9]
    patch_ids.sort(key=lambda ids:(-int(empirical['persistent'][ids].sum()),-int(fixed['persistent'][ids].sum()),-len(ids)))
    for i,ids in enumerate(patch_ids):
        add_region('patch-'+str(i+1),f'변화 비교 구역 {i+1:03d}','local_patch',ids)
    for k,(pair,model) in enumerate(zip(pairs,models)):
        ids=np.flatnonzero(background==k)
        add_region(pair['id'],('바닥' if pair['kind']=='floor' else '벽')+' 기준면 '+str(k+1)+' · 후보 제외','background',ids,plane_links[pair['id']],model)
    published_frames=[]
    for frame in photo_frames:
        if frame['id'] not in used_frames:continue
        name=frame['id']+'.jpg';file=next((folder/name for folder in (frames_dir,review_dir) if (folder/name).is_file()),None)
        if file is None:raise ValueError('Missing original RGB '+name)
        shutil.copy2(file,published/name)
        published_frames.append({**{k:v for k,v in frame.items() if k not in ('depth_path','depth_camera_to_world')},'url':prefix+name})
    np.savez_compressed(output/'measurements.npz',xyz=p,normals=normals,background_plane=background,model_index=model_index,
        distance_m=measured['distance_m'],corrected_distance_m=empirical['corrected_distance_m'],bias_m=bias,reference_threshold_m=reference,threshold_m=empirical['threshold_m'],
        paired_observed=paired,counts=measured['counts_by_epoch'],spreads_m=measured['spreads_m'],
        status=empirical['status'],fixed_status=fixed['status'],reason=empirical['reason'],areas_m2=areas,
        reference_background_plane=reference_background,raw_persistent=empirical.get('raw_persistent',empirical['persistent']))
    document={'kind':'structural_plane_residual_comparison','version':2,'scene':source['scene'],'source_epochs':review['source_epochs'],
        'source_hashes':hashes,'display_origin':source['display_origin'],'display_rotation':source['display_rotation'],
        'bounds':source['bounds'],'displays':geometry,'new_registration':False,'same_input_meshes_and_poses':True,'units':'meters',
        'config':config,'models':models,'regions':regions,'keyframes':published_frames,'fit_check_support_audit':support_audit,
        'variants':[{'label':'기존 고정 2cm','summary':summary(np.arange(len(p)),fixed,p,areas,config)},
                    {'label':'공통 벽·바닥 기준','summary':summary(np.arange(len(p)),empirical,p,areas,config)}],
        'background':{'cores':int((background>=0).sum()),'area_by_epoch_m2':areas[:,background>=0].sum(1).tolist(),
            'reference_common_cores':int((reference_background>=0).sum()),'candidate_regions':len(groups),
            'fixed_exceeding_cores_on_background':int(fixed['persistent'][background>=0].sum()),
            'remaining_cores':int((background<0).sum())},
        'limitations':['Common large floor/wall planes are assumed stable; this is a geometry-based background prior, not semantic or change ground truth.',
            'Occluded or missing counterparts are not classified as addition/removal.',
            'The original aligned maps and camera poses are unchanged; no new ICP or object registration.',
            'Target threshold is max(empirical plane P95, local robust spread envelope), not a sum of duplicate residual variances. The existing 6cm application minimum is shared with the fixed control.',
            'Background exclusion only applies to mutually supported surface points, never the inside of a plane bounding box.',
            'P95 is an empirical residual reference, not calibrated object accuracy. References failing a buffered spatial holdout are disabled.']}
    save(output/'comparison.json',document);save(published/'comparison.json',document)
    print(json.dumps({'output':str(output),'background':document['background'],'variants':document['variants'],
        'regions':len(regions),'photos':len(published_frames)},ensure_ascii=False),flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('comparison-dir','review-dir','frames-dir','previous-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.comparison_dir,args.review_dir,args.frames_dir,args.previous_dir,args.output)
