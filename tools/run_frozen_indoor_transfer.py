"""Apply the frozen Rescan all-surface detector to an unused Bonn sequence.

Only acquisition/coordinate/publication adapters are new. The geometry,
background extraction and candidate functions must match pre-download hashes.
No instance, object correspondence or change ground truth is read.
"""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from all_surface_change import clearance_measurements, classify_clearance
from build_m3c2_independent import ray_votes, connect_photos
from compare_mesh_methods import surface_samples, voxel_indices, pca_normals, PACKED
from empirical_residual import feature_summary
from prepare_bonn_pair import prepare, read_index, camera_points
from run_residual_roi_comparison import load, save, summary
from structural_reference import extract_planes, match_planes, background_assignment, candidate_components

PROJECT=Path(__file__).resolve().parents[1]
PREFIX='/residual-comparison/bonn-placement/'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(protocol,root=PROJECT):
    for name,expected in protocol['source_code_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen detector/preprocessing source changed: '+name)
    config=protocol['frozen_rescan_config']
    if config['radii_m']!=[.08,.16,.28] or config['minimum_displacement_m']!=.06:
        raise ValueError('Unexpected baseline configuration')
    return config


def temporal_windows(rgb_index):
    """Predeclared first/last 20%; never inspect detector output to select."""
    duration=rgb_index[-1][0]-rgb_index[0][0]
    if duration<6:raise ValueError('Capture too short for frozen window rule')
    return ((.5,duration*.20),(duration*.80,duration-.5))


def original_depth_support(frames,epoch):
    points=[]
    for frame in frames:
        if frame['epoch']!=epoch:continue
        w,h=frame['depth_size'];depth=np.fromfile(frame['depth_path'],dtype='<u2').reshape(h,w)
        local,valid=camera_points(depth)
        pose=np.array(frame['depth_camera_to_world'])
        points.append(local[valid]@pose[:3,:3].T+pose[:3,3])
    if not points:raise ValueError('No original depth support for epoch')
    return np.vstack(points)


def box(points):return {'min':points.min(0).tolist(),'max':points.max(0).tolist()}


def run(data,protocol_path,output):
    import open3d as o3d
    protocol=load(protocol_path);config=verify_frozen(protocol)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty run directory')
    output.mkdir(parents=True,exist_ok=True)
    save(output/'frozen-protocol.json',protocol)
    windows=temporal_windows(read_index(data/'rgb.txt'))
    save(output/'acquisition-selection.json',{'sequence':protocol['selected_sequence'],'windows_s':windows,
        'selection_rule':protocol['extra_operational_parameters']['epoch_windows_rule'],
        'selected_before_reconstruction':True,'change_ground_truth_used':False})
    o3d.utility.random.seed(20261016)
    reconstructed=output/'reconstruction'
    metadata=prepare(data,reconstructed,windows=windows,stride=5)
    # The legacy acquisition adapter defaults to removal metadata; correct its
    # descriptive fields only. Its reconstruction and calibration stay frozen.
    for path in (reconstructed/'source.json',reconstructed/'prepared/review.json',
                 reconstructed/'prepared/changes.json',reconstructed/'frames/cameras.json'):
        document=load(path);document.update({'scene':'Bonn unused placement capture',
            'dataset_id':'bonn-placement','sequence':protocol['selected_sequence']})
        save(path,document)
    frames=load(reconstructed/'frames/cameras.json')['frames']
    meshes=[o3d.io.read_triangle_mesh(str(reconstructed/f'prepared/cache/{e}-depth8.ply')) for e in (0,1)]
    hashes=[sha(reconstructed/f'prepared/cache/{e}-depth8.ply') for e in (0,1)]
    samples=[];surface_normals=[];scenes=[];planes=[]
    for e,mesh in enumerate(meshes):
        sample,normal=surface_samples(mesh,20261015+e);samples.append(sample);surface_normals.append(normal)
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));scenes.append(scene)
        planes.append(extract_planes(sample,normal,20261016+e))
        print('Frozen sampling/background epoch',e,len(sample),'samples',len(planes[-1]),'structural planes',flush=True)
    all_vertices=np.vstack([np.asarray(m.vertices) for m in meshes])
    origin=(all_vertices.min(0)+all_vertices.max(0))/2
    centered=[s-origin for s in samples];union=np.vstack(centered);q=union[voxel_indices(union,.08)];p=q+origin
    sample_distances=np.column_stack([cKDTree(s).query(q,workers=4)[0] for s in centered])
    normals=np.zeros_like(q)
    for e in (0,1):
        selected=sample_distances.argmin(1)==e
        normals[selected],_=pca_normals(q[selected],centered[e],surface_normals[e])
    distances=np.column_stack([s.compute_distance(o3d.core.Tensor(p.astype('f4'))).numpy() for s in scenes])
    owner=distances.argmin(1);pairs=match_planes(planes,samples)
    background=background_assignment(p,normals,pairs,samples,require_both=False)
    def plane_public(a):return {k:v for k,v in a.items() if k!='sample_ids'}
    save(output/'structural-extraction.json',{'epochs':[[plane_public(a) for a in group] for group in planes],
        'pairs':[{**{k:v for k,v in a.items() if k!='planes'},'planes':[plane_public(v) for v in a['planes']]} for a in pairs]})
    support_distances=[]
    for e in (0,1):support_distances.append(cKDTree(original_depth_support(frames,e)).query(p,workers=4)[0])
    source_support=np.maximum(np.choose(owner,support_distances),distances.min(1))
    clearance=[scenes[1-e].compute_distance(o3d.core.Tensor(sample.astype('f4'))).numpy() for e,sample in enumerate(samples)]
    print('Running frozen all-surface detector',len(q),'cores; same floor budget',config['noise_budget_m'],flush=True)
    measurement=clearance_measurements(p,normals,owner,samples,surface_normals,clearance)
    result=classify_clearance(measurement,np.array(config['noise_budget_m']),background,source_support)
    counts={k:np.zeros((len(p),2),int) for k in ('occupied','free','occluded')}
    for e in (0,1):
        votes=ray_votes(p,[f for f in frames if f['epoch']==e])
        for key in counts:counts[key][:,e]=votes[key].sum(1)
    own=counts['occupied'][np.arange(len(p)),owner];free=counts['free'][np.arange(len(p)),1-owner]
    occluded=counts['occluded'][np.arange(len(p)),1-owner];supported=(own>=2)&(free>=2)
    result['score']=np.clip(result['score']+10*supported-10*((occluded>=2)&~supported),0,100)
    published=output/'published';published.mkdir();areas=[];displays=[];vertex_maps=[];tree=cKDTree(q)
    for e,mesh in enumerate(meshes):
        vertices=np.asarray(mesh.vertices)-origin;faces=np.asarray(mesh.triangles);ids=faces.ravel()
        packed=np.zeros(len(ids),PACKED);packed['xyz']=vertices[ids];packed['normal']=np.asarray(mesh.vertex_normals)[ids]
        packed['rgb']=np.clip(np.asarray(mesh.vertex_colors)[ids]*255,0,255).astype('u1')
        packed.tofile(published/f'{e}-comparison.bin')
        distance,index=tree.query(packed['xyz'].astype(float),workers=4)
        triangles=packed['xyz'].astype(float).reshape(-1,3,3)
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        weights=np.repeat(area/3,3);weights[distance>.14]=0
        areas.append(np.bincount(index,weights=weights,minlength=len(p)));vertex_maps.append((distance,index,packed))
        displays.append({'epoch':e,'url':PREFIX+f'{e}-comparison.bin','count':len(packed),'triangles':len(faces)})
    areas=np.array(areas);groups=candidate_components(p,result,areas)
    groups.sort(key=lambda ids:-float(np.mean(result['score'][ids])))
    verify_frozen(protocol)
    np.savez_compressed(output/'prediction.npz',xyz=p,normals=normals,owner=owner,source_distance_m=source_support,
        distance_m=measurement['distance_m'],threshold_m=result['threshold_m'],counts=measurement['counts'],
        roughness_m=measurement['roughness_m'],status=result['status'],score=result['score'],reason=result['reason'],areas_m2=areas,
        background_plane=background,visibility_supported=supported,own_occupied=own,opposite_free=free,opposite_occluded=occluded)
    median_distance=np.zeros(len(p));good=np.isfinite(measurement['distance_m']).any(1)
    median_distance[good]=np.nanmedian(measurement['distance_m'][good],axis=1)
    thresholds=result['threshold_m'];median_threshold=np.zeros(len(p));good=np.isfinite(thresholds).any(1)
    median_threshold[good]=np.nanmedian(thresholds[good],axis=1)
    ratio=np.divide(median_distance,median_threshold,out=np.zeros(len(p)),where=median_threshold>0)
    for e,(distance,index,packed) in enumerate(vertex_maps):
        np.zeros((len(index),4),'<f4').tofile(published/f'{e}-fixed.bin')
        stage=result['status'][index].astype(float);stage[distance>.14]=2
        ids=np.flatnonzero(background[index]>=0)
        if len(ids):
            assignment=background_assignment(packed['xyz'][ids]+origin,packed['normal'][ids],pairs,samples,maximum_distance=.07,require_both=False)
            stage[ids[assignment<0]]=2
        np.column_stack([median_distance[index],median_threshold[index],stage,ratio[index]]).astype('<f4').tofile(published/f'{e}-empirical.bin')
    regions=[];photo_cache={};used=set()
    def add_region(identifier,title,kind,ids):
        if not len(ids):return
        subset=ids[::max(1,len(ids)//300)]
        links=[connect_photos(p[subset],e,frames,meshes[e],photo_cache) for e in (0,1)]
        for group in links:used.update(l['frame_id'] for l in group)
        stat=summary(ids,result,p,areas,config)
        if kind=='background':stat.update({'status':'background','background_cores':len(ids)})
        regions.append({'id':identifier,'title':title,'kind':kind,'bounds':box(p[ids]),'statistics':[stat.copy(),stat],
            'keyframes':links,'paired_originals':bool(all(links)),'evaluation':None,
            'distance_by_radius_m':[feature_summary(measurement['distance_m'][ids,j]) for j in range(3)],
            'corrected_distance_by_radius_m':[feature_summary(measurement['distance_m'][ids,j]) for j in range(3)],
            'score':feature_summary(result['score'][ids]),'visibility_supported_cores':int(supported[ids].sum()),
            'conditions':{'roughness_m':feature_summary(measurement['roughness_m'][ids])},
            'reasons':{r:int((result['reason'][ids]==r).sum()) for r in np.unique(result['reason'][ids])}})
    for i,ids in enumerate(groups):add_region(f'candidate-{i+1}',f'변화 후보 {i+1:02d}','local_patch',ids)
    for k,pair in enumerate(pairs):add_region(pair['id'],('바닥' if pair['kind']=='floor' else '수직 구조')+' 기준면 · 후보 제외','background',np.flatnonzero(background==k))
    add_region('overview','전체 촬영 범위','overview',np.arange(len(p)))
    originals=[]
    for frame in frames:
        if frame['id'] not in used:continue
        name=frame['id']+'.jpg';shutil.copy2(reconstructed/'frames'/name,published/name)
        originals.append({**{k:v for k,v in frame.items() if k not in ('depth_path','depth_camera_to_world')},'url':PREFIX+name})
    selected_rgb=[{'id':f['id'],'epoch':f['epoch'],'path':f['original_rgb_file'],'sha256':f['original_rgb_sha256']} for f in frames]
    source_manifest=load(data.parent/'source.json')
    save(output/'selected-originals.json',selected_rgb)
    stats=summary(np.arange(len(p)),result,p,areas,config);total_nonbackground=int((background<0).sum())
    document={'kind':'frozen_indoor_transfer','version':1,'scene':'Bonn · 처음 적용한 상자 놓기 촬영',
        'dataset_id':'bonn-placement','source_epochs':[0,1],'new_registration':False,'same_input_meshes_and_poses':True,
        'source_hashes':hashes,'units':'meters','display_origin':origin.tolist(),'display_rotation':np.eye(3).tolist(),
        'bounds':box(all_vertices-origin),'displays':displays,'config':config,'models':[],'regions':regions,'keyframes':originals,
        'variants':[{'label':'원본 메쉬 · 같은 촬영','summary':stats},{'label':'고정 설정 · 변화 후보','summary':stats}],
        'background':{'cores':int((background>=0).sum()),'area_by_epoch_m2':areas[:,background>=0].sum(1).tolist(),
            'candidate_regions':len(groups),'matched_planes':len(pairs)},
        'assessable_nonbackground':{'total':total_nonbackground,'current':int(result['eligible'].sum())},
        'prediction_sha256':sha(output/'prediction.npz'),'protocol_sha256':sha(protocol_path),
        'transfer':{'frozen_before_download':True,'frozen_source_verified':True,'parameters_retuned':False,
            'floor_reference_refit':False,'labels_used_for_detection':False,'change_ground_truth_read':False,
            'sequence':protocol['selected_sequence'],'source_url':source_manifest['url'],'zip_sha256':source_manifest['sha256'],
            'windows_s':windows,'capture_relation':'Two temporal windows of one new recording, not independent repeat visits.',
            'environment_relation':'Different from Rescan; same Bonn capture environment as the previously used removal sequence.',
            'floor_reference_origin':'Rescan scene_e_0 -> scene_e_2, exact existing empirical P95 reused',
            'visibility_supported_candidates':int((result['persistent']&supported).sum()),
            'candidate_patches':len(groups),'candidate_cores':int(result['persistent'].sum()),
            'candidate_patches_with_paired_originals':sum(r['kind']=='local_patch' and r['paired_originals'] for r in regions)},
        'limitations':['No change annotations or instance labels are loaded; no precision/recall is claimed.',
            'A geometry difference may be a coverage/reconstruction artifact; depth support remains a ranking signal.',
            'This is an unseen capture pair in an already used Bonn environment, not a wholly new-room benchmark.',
            'Original RGB PNGs are preserved; displayed still JPEGs are generated by the unchanged Bonn adapter.']}
    save(output/'comparison.json',document);save(published/'comparison.json',document)
    print(json.dumps({'transfer':document['transfer'],'frames':[sum(f['epoch']==e for f in frames) for e in (0,1)],
        'core_count':len(p),'assessable':document['assessable_nonbackground'],'background':document['background']},ensure_ascii=False),flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.data,args.protocol,args.output)
