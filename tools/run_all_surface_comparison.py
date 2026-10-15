"""Compare all furniture orientations and audit existing Rescan items afterward."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from compare_mesh_methods import surface_samples,PACKED
from build_m3c2_independent import ray_votes,connect_photos,SUPPORTED_RGB_POSES
from build_rescan_review import read_cloud
from run_residual_roi_comparison import load,save,summary
from structural_reference import candidate_components
from empirical_residual import feature_summary
from all_surface_change import clearance_measurements,classify_clearance,evaluate_reference_items


def bounds(p):return {'min':p.min(0).tolist(),'max':p.max(0).tolist()}


def run(comparison_dir,review_dir,frames_dir,previous_dir,normal_reference_dir,output):
    import open3d as o3d
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True);published=output/'published';published.mkdir()
    source=load(comparison_dir/'comparison.json');review=load(review_dir/'review.json');cameras=load(frames_dir/'cameras.json')
    previous=load(previous_dir/'comparison.json');old=np.load(previous_dir/'measurements.npz')
    models=[m for m in previous['models'] if m['kind']=='floor' and m['enabled']]
    if not models:raise ValueError('No approved empirical floor reference; no fixed-error fallback')
    floor_budget=np.max([m['thresholds_m'] for m in models],axis=0)
    source_core=np.load(comparison_dir/'core-fields.npz')['xyz'];origin=np.array(source['display_origin']);rotation=np.array(source['display_rotation'])
    p=source_core+origin;world=p@rotation
    if not np.allclose(p,old['xyz'],atol=1e-12) or source['scene']!=previous['scene'] or review['scene']!=cameras['scene']:
        raise ValueError('Capture/core coordinate mismatch')
    hashes=[hashlib.sha256((review_dir/f'cache/{e}-depth8.ply').read_bytes()).hexdigest() for e in (0,1)]
    if hashes!=source['input_hashes'] or hashes!=previous['source_hashes']:raise ValueError('Aligned meshes changed')
    config={**previous['config'],'all_surface_directions':True,'noise_budget_m':floor_budget.tolist(),
        'global_direction_gate':False,'local_normal_neighbor_angle_deg':50,'minimum_displacement_m':.06,
        'minimum_source_samples':6,'minimum_consistent_scales':2,'roughness_radius_m':.10,
        'normal_reference_sha256':hashlib.sha256((normal_reference_dir/'measurements.npz').read_bytes()).hexdigest(),
        'floor_reference_sha256':hashlib.sha256((previous_dir/'comparison.json').read_bytes()).hexdigest(),
        'evaluation_rule':'After prediction: connected source-instance surface >=9 cores and >=0.04m2. No boxes or labels enter detection.'}
    save(output/'frozen-config.json',config)
    meshes=[o3d.io.read_triangle_mesh(str(review_dir/f'cache/{e}-depth8.ply')) for e in (0,1)]
    scenes=[];samples=[];sample_normals=[];mesh_distances=[]
    for e,mesh in enumerate(meshes):
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));scenes.append(scene)
        sample,normal=surface_samples(mesh,20261015+e);samples.append(sample@rotation.T);sample_normals.append(normal@rotation.T)
        mesh_distances.append(scene.compute_distance(o3d.core.Tensor(world.astype('f4'))).numpy())
    mesh_distances=np.column_stack(mesh_distances);owner=mesh_distances.argmin(1)
    # Restore each source surface's own normal. Do not replace it with a floor
    # normal as the previous target/reference compatibility path did.
    original=np.load(normal_reference_dir/'measurements.npz')
    if not np.allclose(original['xyz'],p,atol=1e-12):raise ValueError('Original source-normal coordinates changed')
    normals=original['normals'].copy()
    clearance=[]
    for e,sample in enumerate(samples):
        clearance.append(scenes[1-e].compute_distance(o3d.core.Tensor((sample@rotation).astype('f4'))).numpy())
    print('Comparing source surfaces in ALL directions',len(p),'queries, floor noise budget',floor_budget.tolist(),flush=True)
    measured=clearance_measurements(p,normals,owner,samples,sample_normals,clearance)
    frames=[]
    for f in review['keyframes']+cameras['frames']:
        f=f.copy()
        if f.get('depth_path') and not Path(f['depth_path']).is_absolute():f['depth_path']=str(Path(__file__).resolve().parents[1]/f['depth_path'])
        if not any(g['id']==f['id'] for g in frames):frames.append(f)
    photo_frames=[f for f in frames if f['status'] in SUPPORTED_RGB_POSES]
    clouds=[];source_distances=[];source_nearest=[]
    transform=np.array(review['registration']['after_to_before'])
    for e,file in enumerate(review['source_files']):
        c=read_cloud(Path(file['path']));xyz=c['xyz']
        if e:xyz=xyz@transform[:3,:3].T+transform[:3,3]
        distance,nearest=cKDTree(xyz).query(world,workers=4)
        clouds.append(c);source_distances.append(distance);source_nearest.append(nearest)
    own_source_distance=np.choose(owner,source_distances)
    source_support=np.maximum(own_source_distance,np.min(mesh_distances,axis=1))
    result=classify_clearance(measured,floor_budget,old['background_plane'],source_support)
    geometry_raw=result['persistent'].copy()
    counts={}
    for key in ('occupied','free','occluded'):counts[key]=np.zeros((len(p),2),int)
    for e in (0,1):
        votes=ray_votes(world,[f for f in frames if f['epoch']==e])
        for key in counts:counts[key][:,e]=votes[key].sum(1)
    own_occupied=counts['occupied'][np.arange(len(p)),owner]
    opposite_free=counts['free'][np.arange(len(p)),1-owner]
    opposite_occluded=counts['occluded'][np.arange(len(p)),1-owner]
    visibility_supported=(own_occupied>=2)&(opposite_free>=2)
    result['score']=np.clip(result['score']+10*visibility_supported-10*((opposite_occluded>=2)&~visibility_supported),0,100)
    print('All-direction geometrically assessable',int(result['eligible'].sum()),'raw differences',int(geometry_raw.sum()),flush=True)
    areas=[];vertex_maps=[];geometry=[];tree=cKDTree(source_core);prefix='/residual-comparison/'
    for e,display in enumerate(source['displays']):
        name=f'{e}-comparison.bin';packed=np.fromfile(comparison_dir/name,dtype=PACKED);xyz=packed['xyz'].astype(float)
        distance,index=tree.query(xyz,workers=4);tri=xyz.reshape(-1,3,3)
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        weights=np.repeat(area/3,3);weights[distance>.14]=0
        areas.append(np.bincount(index,weights=weights,minlength=len(p)));vertex_maps.append((distance,index))
        shutil.copy2(comparison_dir/name,published/name);geometry.append({**display,'url':prefix+name})
    areas=np.array(areas);groups=candidate_components(p,result,areas)
    groups.sort(key=lambda ids:-float(np.mean(result['score'][ids])))
    old_result={'status':old['status'],'eligible':np.isin(old['status'],[0,1]),'persistent':old['status']==1,
        'threshold_m':old['threshold_m'],'reason':old['reason']}
    for e,(distance,index) in enumerate(vertex_maps):
        shutil.copy2(previous_dir/'published'/f'{e}-empirical.bin',published/f'{e}-fixed.bin')
        stage=result['status'][index].astype(float);stage[distance>.14]=2
        previous_field=np.fromfile(previous_dir/'published'/f'{e}-empirical.bin',dtype='<f4').reshape(-1,4)
        stage[(stage==3)&(previous_field[:,2]!=3)]=2
        med_distance=np.zeros(len(p));good=np.isfinite(measured['distance_m']).any(1)
        med_distance[good]=np.nanmedian(measured['distance_m'][good],axis=1)
        threshold=result['threshold_m'];med_threshold=np.zeros(len(p));good=np.isfinite(threshold).any(1)
        med_threshold[good]=np.nanmedian(threshold[good],axis=1)
        ratio=np.divide(med_distance,med_threshold,out=np.zeros(len(p)),where=med_threshold>0)
        np.column_stack([med_distance[index],med_threshold[index],stage,ratio[index]]).astype('<f4').tofile(published/f'{e}-empirical.bin')
    # Freeze prediction artifacts BEFORE reading change identities/types.
    np.savez_compressed(output/'prediction.npz',xyz=p,normals=normals,owner=owner,source_distance_m=source_support,
        distance_m=measured['distance_m'],threshold_m=result['threshold_m'],counts=measured['counts'],roughness_m=measured['roughness_m'],
        status=result['status'],score=result['score'],reason=result['reason'],areas_m2=areas,
        background_plane=old['background_plane'],visibility_supported=visibility_supported,own_occupied=own_occupied,
        opposite_free=opposite_free,opposite_occluded=opposite_occluded,raw_persistent=geometry_raw)
    legacy=load(review_dir/'changes.json')
    nearest_instance=np.choose(owner,[c['instance'][nearest] for c,nearest in zip(clouds,source_nearest)])
    annotation_supported=own_source_distance<=.04
    evaluated=[evaluate_reference_items(legacy['regions'],p,r['persistent'],owner,nearest_instance,annotation_supported,areas)
        for r in (old_result,result)]
    for i,item in enumerate(evaluated[1]):item['previous']=evaluated[0][i]
    evaluation={'reference_count':len(legacy['regions']),'items':evaluated[1],
        'previous_captured':sum(r['outcome']=='captured' for r in evaluated[0]),
        'captured':sum(r['outcome']=='captured' for r in evaluated[1]),
        'partial':sum(r['outcome']=='partial' for r in evaluated[1]),
        'missed':sum(r['outcome']=='missed' for r in evaluated[1]),
        'rule':config['evaluation_rule'],'localization_only':True,'change_type_predicted':False,
        'labels_used_for_detection':False,'candidate_parameters_tuned_to_labels':False,
        'reference_note':'Existing twelve annotated-instance comparison items, including estimated associations; not an exhaustive change ground truth or precision benchmark.'}
    all_reference=np.zeros(len(p),bool)
    for item in evaluated[1]:all_reference[item['core_ids']]=True
    evaluation['candidate_cores_outside_reference_items']=int((result['persistent']&~all_reference).sum())
    save(output/'reference-evaluation.json',evaluation)
    print('Reference localization AFTER detection',json.dumps({k:v for k,v in evaluation.items() if k!='items'},ensure_ascii=False),flush=True)
    for r in evaluated[1]:print('EVAL',r['name'],r['previous']['outcome'],'->',r['outcome'],r['candidate_cores'],'/',r['source_cores'],flush=True)
    photo_cache={};used_frames=set();regions=[]
    def add_region(identifier,title,kind,ids,links=None,model=None,item=None):
        if not len(ids):return
        if links is None:
            subset=ids[::max(1,len(ids)//300)]
            links=[connect_photos(world[subset],e,photo_frames,meshes[e],photo_cache) if len(subset)>=5 else [] for e in (0,1)]
        available={f['id'] for f in photo_frames};all_available={f['id'] for f in frames}
        original_links=links
        links=[[l for l in group if l['frame_id'] in available] for group in original_links]
        if kind=='reference':
            for e in (0,1):
                if links[e]:continue
                own_ids=ids[owner[ids]==e]
                subset=(own_ids if len(own_ids)>=5 else ids)[::max(1,len(own_ids)//300)]
                links[e]=connect_photos(world[subset],e,photo_frames,meshes[e],photo_cache)
                if not links[e]:
                    # Preserve a real original reference photo even where its
                    # legacy camera lacks RGB validation. Do not invent a box.
                    links[e]=[{**l,'bbox_px':None,'camera_check':'legacy_view_candidate',
                        'relation':'reference_original_photo_unverified_projection'}
                        for l in original_links[e] if l['frame_id'] in all_available][:1]
        for group in links:used_frames.update(a['frame_id'] for a in group)
        stats=[summary(ids,r,p,areas,config) for r in (old_result,result)]
        if kind=='background':stats[1].update({'status':'background','background_cores':len(ids)})
        if item:
            for stat,ev in zip(stats,[item['previous'],item]):
                stat['status']='change_candidate' if ev['outcome']=='captured' else 'insufficient_change_evidence' if stat['valid_cores'] else 'unassessable'
        regions.append({'id':identifier,'name':title,'title':title,'kind':kind,'bounds':bounds(p[ids]),
            'keyframes':links,'paired_originals':bool(all(links)),'statistics':stats,'evaluation':item,
            'distance_by_radius_m':[feature_summary(np.abs(old['corrected_distance_m'][ids,j])) for j in range(3)],
            'corrected_distance_by_radius_m':[feature_summary(measured['distance_m'][ids,j]) for j in range(3)],
            'score':feature_summary(result['score'][ids]),'visibility_supported_cores':int(visibility_supported[ids].sum()),
            'conditions':{'roughness_m':feature_summary(measured['roughness_m'][ids])},
            'reasons':{r:int((result['reason'][ids]==r).sum()) for r in np.unique(result['reason'][ids])},
            'reference_plane':model['id'] if model else None})
    for event,item in zip(legacy['regions'],evaluated[1]):
        add_region('reference-'+str(event['id']),event['name'],'reference',np.array(item['core_ids'],int),event['keyframes'],item=item)
    for i,ids in enumerate(groups):add_region('candidate-'+str(i+1),f'변화 후보 {i+1:02d}','local_patch',ids)
    for k,model in enumerate(previous['models']):
        ids=np.flatnonzero(old['background_plane']==k)
        previous_region=next(r for r in previous['regions'] if r['id']==model['id'])
        add_region(model['id'],('바닥' if model['kind']=='floor' else '수직 구조')+' 기준면 '+str(k+1)+' · 후보 제외',
            'background',ids,previous_region['keyframes'],model)
    published_frames=[]
    for f in frames:
        if f['id'] not in used_frames:continue
        name=f['id']+'.jpg';file=next((folder/name for folder in (frames_dir,review_dir,previous_dir/'published') if (folder/name).is_file()),None)
        if file is None:raise ValueError('Original photo missing '+name)
        shutil.copy2(file,published/name)
        published_frames.append({**{k:v for k,v in f.items() if k not in ('depth_path','depth_camera_to_world')},'url':prefix+name})
    nonbackground=old['background_plane']<0
    document={'kind':'all_surface_residual_comparison','version':3,'scene':source['scene'],'source_epochs':review['source_epochs'],
        'source_hashes':hashes,'new_registration':False,'same_input_meshes_and_poses':True,'units':'meters',
        'display_origin':source['display_origin'],'display_rotation':source['display_rotation'],'bounds':source['bounds'],'displays':geometry,
        'config':config,'models':models,'evaluation':evaluation,'regions':regions,'keyframes':published_frames,
        'variants':[{'label':'이전 방식 · 방향 제한','summary':summary(np.arange(len(p)),old_result,p,areas,config)},
                    {'label':'모든 방향 · 3D 표면 비교','summary':summary(np.arange(len(p)),result,p,areas,config)}],
        'background':{**previous['background'],'candidate_regions':len(groups),'matched_planes':len(previous['models'])},
        'assessable_nonbackground':{'total':int(nonbackground.sum()),'previous':int(old_result['eligible'][nonbackground].sum()),'current':int(result['eligible'][nonbackground].sum())},
        'prediction_sha256':hashlib.sha256((output/'prediction.npz').read_bytes()).hexdigest(),
        'reference_evaluation_sha256':hashlib.sha256((output/'reference-evaluation.json').read_bytes()).hexdigest(),
        'limitations':['Floor residual P95 is an empirical scalar noise budget, not validated 3D directional covariance.',
            'All orientations are attempted; insufficient local samples or source support remain unassessable.',
            'Geometry-only differences with missing/occluded opposite views remain suspects, not confirmed additions/removals.',
            'Localization against twelve source-instance reference items does not establish change-type recognition or dataset-wide precision.',
            'Candidate scores are ranking heuristics, not probabilities. No instance/change labels are inputs to detection.']}
    save(output/'comparison.json',document);save(published/'comparison.json',document)
    print(json.dumps({'output':str(output),'localization':{k:v for k,v in evaluation.items() if k!='items'},
        'assessable':document['assessable_nonbackground'],'candidate_patches':len(groups),'candidate_cores':int(result['persistent'].sum()),'photos':len(published_frames)},ensure_ascii=False),flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-dir','review-dir','frames-dir','previous-dir','normal-reference-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.comparison_dir,args.review_dir,args.frames_dir,args.previous_dir,args.normal_reference_dir,args.output)
