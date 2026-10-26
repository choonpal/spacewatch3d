"""Compare fixed 2cm and reviewed-ROI empirical residuals on aligned Rescan.

Original meshes/poses and the default viewer are read only. This is a surface
comparison experiment, not an object addition/removal/motion classifier.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from compare_mesh_methods import surface_samples,pca_normals,PACKED
from mesh_change_noise import surface_pair_measurements,selected_surface_support
from empirical_residual import evaluate,fit_models,fixed_control,in_bounds,region_decision,feature_summary
from build_m3c2_independent import ray_votes,connect_photos,SUPPORTED_RGB_POSES


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def largest_component(points,ids,radius=.16):
    if not len(ids):return 0
    parent=np.arange(len(ids));sizes=np.ones(len(ids),int)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in cKDTree(points[ids]).query_pairs(radius):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a;sizes[a]+=sizes[b]
    return int(max(sizes[find(i)] for i in range(len(ids))))


def summary(ids,result,points,areas,config):
    eligible=result['eligible'][ids];over=result['persistent'][ids]
    valid_fraction=float(eligible.mean()) if len(ids) else 0.
    exceed_fraction=float(over.sum()/max(eligible.sum(),1))
    over_ids=ids[over];component=largest_component(points,over_ids,config['connectivity_radius_m'])
    area=float(areas[:,over_ids].mean(axis=0).sum()) if len(over_ids) else 0.
    status,reason=region_decision(valid_fraction,exceed_fraction,component,area,config)
    return {'status':status,'reason':reason,'core_count':len(ids),'valid_cores':int(eligible.sum()),
        'valid_fraction':valid_fraction,'exceeding_cores':int(over.sum()),'exceeding_valid_fraction':exceed_fraction,
        'largest_connected_exceedance_cores':component,'exceeding_mean_epoch_area_m2':area,
        'total_area_by_epoch_m2':areas[:,ids].sum(axis=1).tolist(),
        'valid_area_by_epoch_m2':areas[:,ids[eligible]].sum(axis=1).tolist(),
        'exceeding_area_by_epoch_m2':areas[:,over_ids].sum(axis=1).tolist(),
        'threshold_by_radius_m':[feature_summary(result['threshold_m'][ids,j]) for j in range(result['threshold_m'].shape[1])]}


def run(comparison_dir,review_dir,frames_dir,roi_config,output):
    import open3d as o3d
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True);published=output/'published';published.mkdir()
    source=load(comparison_dir/'comparison.json');review=load(review_dir/'review.json');cameras=load(frames_dir/'cameras.json')
    config=load(roi_config);old=load(review_dir/'changes.json') if (review_dir/'changes.json').is_file() else {'regions':[]}
    if config['scene']!=review['scene'] or config['source_epochs']!=review['source_epochs'] or cameras['scene']!=review['scene']:
        raise ValueError('Capture/ROI mismatch')
    meshes=[o3d.io.read_triangle_mesh(str(review_dir/f'cache/{e}-depth8.ply')) for e in (0,1)]
    hashes=[hashlib.sha256((review_dir/f'cache/{e}-depth8.ply').read_bytes()).hexdigest() for e in (0,1)]
    if hashes!=source['input_hashes']:raise ValueError('Mesh/cache mismatch')
    core=np.load(comparison_dir/'core-fields.npz');q=core['xyz'];origin=np.array(source['display_origin']);rotation=np.array(source['display_rotation'])
    p=q+origin;world=p@rotation
    frames=[]
    for frame in review['keyframes']+cameras['frames']:
        f=frame.copy()
        if f.get('depth_path') and not Path(f['depth_path']).is_absolute():f['depth_path']=str(Path(__file__).resolve().parents[1]/f['depth_path'])
        if not any(g['id']==f['id'] for g in frames):frames.append(f)
    photo_frames=[f for f in frames if f['status'] in SUPPORTED_RGB_POSES]
    photo_cache={};roi_links={}
    for roi in config['rois']:
        ids=np.flatnonzero(in_bounds(p,roi['bounds']))
        roi_links[roi['id']]=[connect_photos(world[ids],e,photo_frames,meshes[e],photo_cache) if len(ids)>=5 else [] for e in (0,1)]
    # Record explicit ROI definitions and source photo links before residual fit.
    save(output/'roi-selection.json',{'config':config,'source_hashes':hashes,'keyframes':roi_links})
    print('Measuring identical normal-direction surfaces in fit/check/targets',flush=True)
    samples=[];mesh_distances=[]
    for e,mesh in enumerate(meshes):
        sample,normal=surface_samples(mesh,20261015+e);samples.append((sample@rotation.T-origin,normal@rotation.T))
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        mesh_distances.append(scene.compute_distance(o3d.core.Tensor(world.astype('f4'))).numpy())
    owner=np.column_stack(mesh_distances).argmin(axis=1)
    per_epoch=[pca_normals(q,*sample) for sample in samples]
    normals=np.where((owner==0)[:,None],per_epoch[0][0],per_epoch[1][0])
    quality=np.minimum(per_epoch[0][1],per_epoch[1][1]);agreement=np.abs(np.einsum('ij,ij->i',per_epoch[0][0],per_epoch[1][0]))
    measured=surface_pair_measurements(q,normals,[s[0] for s in samples],config['radii_m'],config['half_depth_m'])
    observed=[];counts=[]
    for e in (0,1):
        centers=q[:,None,:]+normals[:,None,:]*measured['centers_m'][:,:,e,None]
        votes=ray_votes(((centers.reshape(-1,3)+origin)@rotation),[f for f in frames if f['epoch']==e])
        number=votes['occupied'].sum(axis=1).reshape(len(q),-1)
        counts.append(number);observed.append(number>=2)
    paired=observed[0]&observed[1]
    reviewed={key:all(links) for key,links in roi_links.items()}
    preliminary,pre_valid,_,_,shape_ok=fit_models(p,measured,quality,agreement,paired,config,reviewed)
    supports={};overlap_audit=[]
    for roi in config['rois']:
        ids=np.array(next(m['fit_core_ids'] for m in preliminary if m['roi_id']==roi['id']),int) if roi['role']=='fit' else np.flatnonzero(in_bounds(p,roi['bounds'])&shape_ok&(pre_valid.sum(1)>=2))
        supports[roi['id']]=[]
        for sample,_ in samples:
            per_radius=[]
            for j,radius in enumerate(config['radii_m']):
                supported_ids=ids[pre_valid[ids,j]]
                per_radius.append(selected_surface_support(q[supported_ids],normals[supported_ids],sample,[radius],config['half_depth_m'])[0])
            supports[roi['id']].append(per_radius)
    disjoint={roi['id']:True for roi in config['rois'] if roi['role']=='fit'}
    for fit in [r for r in config['rois'] if r['role']=='fit']:
        for check in [r for r in config['rois'] if r['role']=='check']:
            overlap=[[len(supports[fit['id']][e][j]&supports[check['id']][e][j]) for j in range(len(config['radii_m']))] for e in (0,1)]
            if any(any(row) for row in overlap):disjoint[fit['id']]=False
            overlap_audit.append({'fit_roi':fit['id'],'check_roi':check['id'],'shared_sample_counts_by_epoch_radius':overlap})
    save(output/'fit-check-support-audit.json',overlap_audit)
    empirical=evaluate(p,measured,quality,agreement,paired,config,reviewed,disjoint)
    fixed=fixed_control(measured,paired)
    print('Reference/check outcomes',[(m['id'],m['fit_core_count'],m['thresholds_m'],m['enabled'],m['check_reason']) for m in empirical['models']],flush=True)
    areas=[];nearest_vertices=[];geometry=[]
    tree=cKDTree(q);prefix='/residual-comparison/'
    for e,display in enumerate(source['displays']):
        name=f'{e}-comparison.bin';packed=np.fromfile(comparison_dir/name,dtype=PACKED)
        xyz=packed['xyz'].astype(float);d,index=tree.query(xyz,workers=4);nearest_vertices.append((d,index))
        tri=xyz.reshape(-1,3,3);face_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        weights=np.repeat(face_area/3,3);weights[d>.14]=0
        areas.append(np.bincount(index,weights=weights,minlength=len(q)))
        shutil.copy2(comparison_dir/name,published/name);geometry.append({**display,'url':prefix+name})
        for mode,result in [('fixed',fixed),('empirical',empirical)]:
            stage=result['status'][index].astype(float);stage[d>.14]=2
            thresholds=result['threshold_m'].copy()
            if mode=='empirical':
                missing=~np.isfinite(thresholds)
                thresholds[missing]=result['provisional_threshold_m'][missing]
            mask=np.isfinite(thresholds).any(axis=1)
            threshold=np.full(len(q),np.nan);threshold[mask]=np.nanmedian(thresholds[mask],axis=1)
            distance=np.full(len(q),np.nan);good=np.isfinite(measured['distance_m']).any(axis=1)
            distance[good]=np.nanmedian(np.abs(measured['distance_m'][good]),axis=1)
            fields=np.column_stack([np.nan_to_num(distance[index]),np.nan_to_num(threshold[index]),stage,np.nan_to_num(distance[index]/threshold[index])])
            fields.astype('<f4').tofile(published/f'{e}-{mode}.bin')
    areas=np.array(areas);regions=[];used_frames=set();reference_mask=np.zeros(len(q),bool)
    def add_region(identifier,title,bounds,kind,ids,links=None):
        if not len(ids):return
        if links is None:links=[connect_photos(world[ids],e,photo_frames,meshes[e],photo_cache) if len(ids)>=5 else [] for e in (0,1)]
        for group in links:used_frames.update(l['frame_id'] for l in group)
        region={'id':identifier,'name':title,'title':title,'kind':kind,'bounds':bounds,'source':'residual_comparison',
            'keyframes':links,'paired_originals':bool(all(links)),
            'statistics':[summary(ids,fixed,p,areas,config),summary(ids,empirical,p,areas,config)],
            'distance_by_radius_m':[feature_summary(np.abs(measured['distance_m'][ids,j])) for j in range(len(config['radii_m']))],
            'reasons':{reason:int(np.sum(empirical['reason'][ids]==reason)) for reason in np.unique(empirical['reason'][ids])},
            'provisional_reference':{'not_a_change_decision':True,'exceeding_cores':int(empirical['provisional_persistent'][ids].sum()),
                'threshold_by_radius_m':[feature_summary(empirical['provisional_threshold_m'][ids,j]) for j in range(len(config['radii_m']))]},
            'conditions':{'roughness_m':feature_summary(empirical['roughness_m'][ids]),
                'density':feature_summary(empirical['density'][ids]),'planarity':feature_summary(quality[ids]),
                'normal_agreement':feature_summary(agreement[ids])}}
        regions.append(region)
    for roi in config['rois']:
        ids=np.flatnonzero(in_bounds(p,roi['bounds']));reference_mask[ids]=True
        add_region(roi['id'],roi['label'],roi['bounds'],roi['role'],ids,roi_links[roi['id']])
    for event in old['regions']:
        ids=np.flatnonzero(in_bounds(p,event['bounds']))
        add_region('object-'+str(event['id']),'객체 범위 · '+event['name'],event['bounds'],'provided_object',ids)
    tiles=np.floor((p-p.min(axis=0))/config['local_patch_size_m']).astype(int);_,membership=np.unique(tiles,axis=0,return_inverse=True)
    patch_ids=[]
    for k in range(int(membership.max())+1):
        ids=np.flatnonzero((membership==k)&~reference_mask)
        if len(ids)>=9:patch_ids.append(ids)
    patch_ids.sort(key=lambda ids:(-int(empirical['persistent'][ids].sum()),-int(fixed['persistent'][ids].sum()),-len(ids)))
    for k,ids in enumerate(patch_ids):
        bounds={'min':p[ids].min(axis=0).tolist(),'max':p[ids].max(axis=0).tolist()}
        add_region('patch-'+str(k+1),f'국소 패치 {k+1:03d}',bounds,'local_patch',ids)
    published_frames=[]
    for frame in photo_frames:
        if frame['id'] not in used_frames:continue
        name=frame['id']+'.jpg';source_file=next((folder/name for folder in (frames_dir,review_dir) if (folder/name).is_file()),None)
        if source_file is None:raise ValueError('Original frame unavailable: '+name)
        shutil.copy2(source_file,published/name)
        published_frames.append({**{k:v for k,v in frame.items() if k not in ('depth_path','depth_camera_to_world')},'url':prefix+name})
    evaluation=np.flatnonzero(~reference_mask)
    document={'kind':'reviewed_roi_residual_comparison','version':1,'scene':source['scene'],'source_epochs':review['source_epochs'],
        'source_hashes':hashes,'comparison_core_sha256':hashlib.sha256((comparison_dir/'core-fields.npz').read_bytes()).hexdigest(),
        'camera_metadata_sha256':hashlib.sha256((frames_dir/'cameras.json').read_bytes()).hexdigest(),
        'review_metadata_sha256':hashlib.sha256((review_dir/'review.json').read_bytes()).hexdigest(),
        'roi_config_sha256':hashlib.sha256(roi_config.read_bytes()).hexdigest(),
        'distance_operator_sha256':hashlib.sha256(Path(__file__).with_name('mesh_change_noise.py').read_bytes()).hexdigest(),
        'display_origin':source['display_origin'],'display_rotation':source['display_rotation'],'bounds':source['bounds'],'displays':geometry,
        'new_registration':False,'same_input_meshes_and_poses':True,'units':'meters','config':config,'models':empirical['models'],
        'fit_check_support_audit':overlap_audit,
        'regions':regions,'keyframes':published_frames,'variants':[
            {'label':'고정 2cm · 표면 판정','summary':summary(evaluation,fixed,p,areas,config)},
            {'label':'수동 ROI · 경험적 기준','summary':summary(evaluation,empirical,p,areas,config)}],
        'scope':'Same robust normal-direction surface operator and radii in all fitting/check/target locations. Surface-stage comparison; default legacy added/removed/moved results are untouched.',
        'formula':'For each radius T=P95 of absolute fit-ROI distances with equal spatial-block total weights. No sigma insertion, residual subtraction or additional local-variance sum. Exceed only when abs(d)>T+floating-point tolerance; >=2 radii same direction; matched geometry/roughness/density/position and spatial check required.',
        'reference_evaluation_excluded_cores':int(reference_mask.sum()),
        'probability_calibrated':False,'limitations':['Manual carpet-floor stability is a reviewed assumption, not ground truth.','References are restricted to comparable near-horizontal near-ground surfaces; chairs/walls/unknown observation conditions can remain unassessable.','P95 is an empirical normal-residual reference, not 95% object accuracy.','Sparse, occluded or missing counterparts are not classified as addition or removal.','Area uses triangle area split between nearest shared cores, reported separately per epoch.']}
    save(output/'comparison.json',document);save(published/'comparison.json',document)
    np.savez_compressed(output/'measurements.npz',xyz=p,normals=normals,distance_m=measured['distance_m'],
        spreads_m=measured['spreads_m'],counts=measured['counts_by_epoch'],paired_observed=paired,
        empirical_threshold_m=empirical['threshold_m'],provisional_threshold_m=empirical['provisional_threshold_m'],
        empirical_status=empirical['status'],fixed_status=fixed['status'],reason=empirical['reason'],areas_m2=areas,
        quality=quality,normal_agreement=agreement,occupied_view_counts=np.stack(counts,axis=-1))
    with (output/'samples.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['core','x_m','y_m','z_m','role','fixed_status','empirical_status','reason',
            *[f'distance_r{int(r*100)}cm_m' for r in config['radii_m']],*[f'threshold_r{int(r*100)}cm_m' for r in config['radii_m']]])
        for i in range(len(q)):writer.writerow([i,*p[i], 'reference_or_check' if reference_mask[i] else 'target',int(fixed['status'][i]),int(empirical['status'][i]),empirical['reason'][i],*measured['distance_m'][i],*empirical['threshold_m'][i]])
    print(json.dumps({'output':str(output),'variants':document['variants'],'models':[(m['id'],m['fit_core_count'],m['thresholds_m'],m['enabled'],m['check_reason']) for m in empirical['models']],'regions':len(regions),'photos':len(published_frames)},ensure_ascii=False),flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-dir','review-dir','frames-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--roi-config',type=Path,default=Path(__file__).with_name('residual-roi-rescan.json'))
    args=parser.parse_args();run(args.comparison_dir,args.review_dir,args.frames_dir,args.roi_config,args.output)
