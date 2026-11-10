"""Apply the shared frozen statistical mechanism to a capture's own mesh input."""
import argparse,copy,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();capture=args.capture
    release=Path('/workspace/spacewatch_pipeline_v5_1')
    template=release/'profiles/dorm2_statistical_scored'
    out=args.output;out.mkdir(exist_ok=False)
    profile=out/'profile';profile.mkdir()
    for name in ['code','native','algorithms']:
        shutil.copytree(template/name,profile/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['profile_runner.py','integrated_runner.py','cache_optimization.py','fast_execution.py','manual.json','prompt.txt']:
        shutil.copyfile(template/name,profile/name)
    sys.path.insert(0,str(profile/'algorithms'))
    spec=importlib.util.spec_from_file_location('statistical_adapter',Path(__file__).with_name('statistical_adapter.py'))
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    results=read(args.baseline_run/'results.json')
    source=results[capture+'.semantic'];base_scene=read(results[capture+'.scene']['scene'])
    base_plan=read(results[capture+'.plan']['plan'])
    reconstruct=results[capture+'.reconstruct']
    with np.load(source['arrays'],allow_pickle=False) as z:data={k:z[k] for k in ['vertices','faces','rgb','face_class_id']}
    v,f,labels=(data[k] for k in ['vertices','faces','face_class_id'])
    catalog={r['id']:r for r in read(source['vocabulary'])}
    rows,groups,reports=adapter.classify(data,catalog)
    assets=profile/'assets';assets.mkdir()
    targets=np.zeros(len(f),'u4');objects=[];mapping=[]
    for i,(row,ids) in enumerate(zip(rows,groups),1):
        if row['excluded']:continue
        file=assets/(row['instance_id']+'.obj');adapter.export_obj(file,v,f[ids])
        assert not targets[ids].any()
        targets[ids]=i
        objects.append(dict(instance_id=row['instance_id'],mesh_uri='assets/'+file.name,
            class_name=row['class_name'],class_id=row['class_id'],source_triangle_count=row['faces'],
            solid=False,description='Connected semantic region; physical object identity unverified'))
        mapping.append(dict(region_id=i,instance_id=row['instance_id'],class_id=row['class_id'],class_name=row['class_name'],faces=row['faces']))
    adapter.export_obj(assets/'remainder.obj',v,f[targets==0])
    import flat_floor
    flat_floor.SERVER_PRIOR=profile/'algorithms/ground_roles.py'
    original_floor=shapely.union_all([Polygon(p['polygon'],p.get('holes',[])) for p in base_scene['floor']['components']])
    floor,collision,accepted,ground_report=flat_floor.prepare_navigation(v,f,labels,{k:r['name'] for k,r in catalog.items()},original_floor,base_scene['floor']['height_m'])
    adapter.export_obj(assets/'navigation_collision.obj',v,f[collision])
    scene=copy.deepcopy(base_scene)
    scene.update(objects=objects,structures=[dict(mesh_uri='assets/remainder.obj',solid=False)],
        navigation_geometry=[dict(mesh_uri='assets/navigation_collision.obj',solid=False)])
    scene['floor']['components']=[dict(polygon=list(p.exterior.coords),holes=[list(r.coords) for r in p.interiors]) for p in flat_floor.polygon_parts(floor)]
    scene['floor']['estimation_method']='semantic_floor_full_xy_flat_indoor_prior_v2'
    scene['navigation_profile']=dict(adapter_version='flat-ground-role-v2',
        route_grid_anchor_xy_m=base_plan['route_graph']['grid_origin_xy_m'],
        fallback_tile_origin_xy_m=base_plan['coverage_report']['surface_sampling']['fallback_tiles']['origin_xy_m'],
        ground_role_report='ground-role-report.json',original_observation_geometry_unchanged=True,physical_safety_verified=False)
    scene['input_assumptions'].update(single_level_flat_indoor_assumption_from_v518=True,
        navigation_floor_projection_not_source_mesh_reconstruction=True,physical_safety_verified=False)
    scene['source_context'].update(original_mesh_faces=len(f),required_objects=len(objects),residual_faces=int((targets==0).sum()),
        class_size_templates_used=False,geometry_deleted_faces=0,source_arrays=source['arrays'],
        target_policy='shared statistical semantic components with residual-shape screen',physical_object_identity_verified=False)
    cfg=read(template/'config.json');cfg['camera_height_m']=base_plan['provenance']['config']['camera_height_m'];cfg['clearance_m']=.2
    class_scores=read(release/'profiles/dorm3_statistical_scored/score_summary.json')['class_scores']
    scores={};fallback=[]
    for row in objects:
        name=row['class_name']
        if name in class_scores:
            score=class_scores[name]
            scores[row['instance_id']]=dict(score=score['score'],reason=f"Existing shared class-level score: L{score['level']} / {score['line']}")
        else:fallback.append(name)
    save(profile/'importance_cache.json',dict(scene_id=scene['scene_id'],capture_id=capture,scores=scores,
        default_classes=sorted(set(fallback)),default_importance=cfg['default_importance'],new_llm_calls=0))
    save(profile/'scene.json',scene);save(profile/'config.json',cfg);save(profile/'ground-role-report.json',ground_report)
    save(profile/'target_mapping.json',mapping)
    save(profile/'filter.json',dict(status='algorithm_applied_physical_labels_unverified',
        source_semantic_components=len(rows),excluded_components=sum(r['excluded'] for r in rows),
        retained_components=len(objects),geometry_deleted_faces=0,source_faces=len(f),
        class_size_templates_used=False,classes=reports,objects=rows))
    np.savez_compressed(profile/'face_roles.npz',source_face_id=np.arange(len(f)),target_region_index=targets,source_class_id=labels)
    np.savez_compressed(profile/'navigation_roles.npz',source_face_id=np.arange(len(f)),collision_mask=collision,accepted_unknown_support=accepted)
    save(profile/'input_binding.json',dict(capture_id=capture,frame_id=scene['coordinates']['frame_id'],
        source_run=reconstruct['source_run'],coordinates_sha256=sha(reconstruct['coordinates']),
        source_arrays=source['arrays'],arrays={k:digest(a) for k,a in data.items()}))
    # Independently reopen the exported observation geometry; collision proxy
    # is deliberately separate and must not replace or double-count these faces.
    triangles=np.concatenate([trimesh.load_scene(profile/o['mesh_uri'],process=False).to_mesh().triangles for o in scene['objects']+scene['structures']])
    def signature(t):
        t=np.ascontiguousarray(t,dtype='f8').reshape(-1,9)
        return np.sort(t.view(np.dtype((np.void,72))).ravel())
    assert len(triangles)==len(f) and np.array_equal(signature(triangles),signature(v[f]))
    save(profile/'checksums.json',{p.relative_to(profile).as_posix():sha(p) for p in sorted(profile.rglob('*')) if p.is_file() and p.name!='checksums.json'})
    jobcfg=read(release/'configs/dorm3.json')
    original=next(c for c in read(args.baseline_run/'identity.json')['config']['captures'] if c['id']==capture)
    cap=copy.deepcopy(original)
    cap.update(semantic_arrays=source['arrays'],semantic_frame_id=scene['coordinates']['frame_id'],
        semantic_coordinates_sha256=sha(reconstruct['coordinates']),expected_candidates=len(objects),
        planning_profile=dict(path=str(profile),sha256=sha(profile/'checksums.json')))
    jobcfg['captures']=[cap];jobcfg['segmentation']['vocabulary']=source['vocabulary']
    save(out/'config.json',jobcfg)
    protected={str(Path(p)):sha(p) for p in [source['arrays'],source['vocabulary'],reconstruct['coordinates'],results[capture+'.scene']['scene'],results[capture+'.plan']['plan']]}
    save(out/'preparation.json',dict(capture=capture,policy_version='5.1.9',clearance_m=.2,
        faces=len(f),targets=len(objects),components=len(rows),excluded=sum(r['excluded'] for r in rows),
        all_original_observation_triangles_preserved=True,default_importance_classes=sorted(set(fallback)),
        camera_height_m=cfg['camera_height_m'],source_hashes=protected,baseline_run=str(args.baseline_run)))
    print(json.dumps(dict(capture=capture,targets=len(objects),components=len(rows),faces=len(f),config=str(out/'config.json'))))

if __name__=='__main__':main()
