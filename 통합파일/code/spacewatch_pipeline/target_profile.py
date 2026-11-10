"""Hash-bound saved planning profiles. Never infer targets for a different map."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .io import ROOT,read,save,sha,result


def array_digest(value):
    import numpy as np
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def checked_profile(config,capture,data=None):
    import numpy as np
    opt=capture['planning_profile'];folder=Path(opt['path'])
    folder=folder if folder.is_absolute() else ROOT/folder
    manifest=folder/'checksums.json'
    if sha(manifest)!=opt['sha256']:raise ValueError('Planning profile manifest changed; select a new verified profile')
    records=read(manifest)
    for name,digest in records.items():
        p=folder/name
        if not p.resolve().is_relative_to(folder.resolve()) or sha(p)!=digest:raise ValueError('Planning profile asset changed: '+name)
    bind=read(folder/'input_binding.json')
    if (capture['id']!=bind['capture_id'] or capture['semantic_frame_id']!=bind['frame_id']
            or sha(capture['coordinates'])!=bind['coordinates_sha256']
            or str(Path(capture['source_run']).resolve())!=bind['source_run']):
        raise ValueError('Planning profile capture/coordinate/source mismatch')
    if data is None:
        from .provided import checked_arrays
        data=checked_arrays(capture['semantic_arrays'])
    for key,digest in bind['arrays'].items():
        if array_digest(data[key])!=digest:raise ValueError('Planning profile input differs: '+key)
    scene=read(folder/'scene.json');scores=read(folder/'importance_cache.json')
    if scene['capture_id']!=capture['id'] or scene['coordinates']['frame_id']!=bind['frame_id']:
        raise ValueError('Planning profile scene identity mismatch')
    if scores['scene_id']!=scene['scene_id'] or scores['capture_id']!=scene['capture_id']:
        raise ValueError('Planning profile importance identity mismatch')
    objects={o['instance_id']:o for o in scene['objects']}
    missing=set(objects)-set(scores['scores'])
    if set(scores['scores'])-set(objects):raise ValueError('Planning profile score/target mismatch')
    if missing:
        # Explicit defaults remain defaults, rather than fabricated cached scores.
        classes={objects[i].get('class_name') for i in missing}
        defaults=scores.get('default_classes',[])
        default=scores.get('default_importance')
        import math
        if (None in classes or classes!=set(defaults) or len(defaults)!=len(set(defaults))
                or isinstance(default,bool) or not isinstance(default,(int,float)) or not math.isfinite(default)
                or not 0<=default<=1 or default!=read(folder/'config.json').get('default_importance')):
            raise ValueError('Planning profile score/target mismatch: undeclared or inconsistent defaults')
    elif scores.get('default_classes'):
        raise ValueError('Planning profile score/target mismatch: unused defaults')
    roles=np.load(folder/'face_roles.npz',allow_pickle=False)
    if not np.array_equal(roles['source_face_id'],np.arange(len(data['faces']))) or not np.array_equal(roles['source_class_id'],data['face_class_id']):
        raise ValueError('Planning profile face correspondence mismatch')
    mapping=read(folder/'target_mapping.json')
    positive={int(x) for x in np.unique(roles['target_region_index']) if x}
    if positive!={r['region_id'] for r in mapping}:raise ValueError('Planning profile target mapping mismatch')
    if {r['instance_id'] for r in mapping}!={o['instance_id'] for o in scene['objects']}:raise ValueError('Planning profile object identities differ')
    for row in mapping:
        mask=roles['target_region_index']==row['region_id']
        if int(mask.sum())!=row['faces'] or not np.all(data['face_class_id'][mask]==row['class_id']):
            raise ValueError('Planning profile target faces/classes differ')
    return folder,scene,mapping,roles['target_region_index'].astype(np.uint32),data


def instances(job):
    import numpy as np
    from .size_instances import export
    c=job['capture'];out=Path(job['output']);src=job['dependencies'][c['id']+'.semantic']
    d=dict(np.load(src['arrays'],allow_pickle=False))
    folder,scene,mapping,targets,d=checked_profile(job['config'],c,d)
    targets=targets.copy();kinds=np.where(targets>0,2,1).astype(np.uint8)
    remainder=max(int(targets.max(initial=0))+1,1);regions=targets.copy();regions[regions==0]=remainder
    rows=[dict(id=r['region_id'],instance_id=r['instance_id'],class_id=r['class_id'],name=r['class_name'],
               faces=r['faces'],kind='object_candidate',method='statistical semantic region; physical identity unverified') for r in mapping]
    if np.any(targets==0):rows.append(dict(id=remainder,class_id=255,faces=int((targets==0).sum()),kind='structure',method='all residual geometry preserved'))
    for key in ['face_instance_id','instance_observations','instance_agreement']:
        if key in d:d['upstream_'+key]=d.pop(key)
    d.update(face_instance_id=targets,face_region_id=regions,face_region_kind=kinds)
    np.savez_compressed(out/'mesh_arrays.npz',**d);save(out/'regions.json',rows)
    policy=dict(method='frozen verified statistical target profile',class_size_templates_used=False,
        require_all_objects=read(folder/'config.json')['require_all_objects'],
        target_count=len(mapping),profile_sha256=c['planning_profile']['sha256'],
        residual_policy='spatial coverage, occlusion and collision; not independent object quality obligation')
    save(out/'target_policy.json',policy);export(out,d,targets,kinds,regions)
    summary=dict(instances=len(mapping),candidate_instances=len(mapping),size_references=0,
        counts={'object_candidate':len(mapping),'structure':int(np.any(targets==0))},
        geometry_preserved=True,all_faces_have_region_id=True,discarded_faces=0,
        physical_instance_accuracy_measured=False,known_per_physical_instance=False,
        method=policy['method'],required_object_policy='aggregate_count_target_not_all_objects',
        require_all_objects=policy['require_all_objects'],class_size_templates_used=False,
        excluded_semantic_components=read(folder/'filter.json')['excluded_components'],
        residual_policy=policy['residual_policy'],profile=str(folder))
    save(out/'summary.json',summary)
    result(out,arrays=str(out/'mesh_arrays.npz'),regions=str(out/'regions.json'),assumptions=str(out/'target_policy.json'),
        instance_mesh=str(out/'mesh_instances.glb'),instance_ply=str(out/'mesh_instances.ply'),
        summary=summary,vocabulary=src['vocabulary'],planning_profile=str(folder),
        external_artifacts=[str(folder/'checksums.json')]+[str(folder/p) for p in read(folder/'checksums.json')])


def scene(job):
    folder,spec,mapping,targets,data=checked_profile(job['config'],job['capture'])
    out=Path(job['output']);spec=json.loads(json.dumps(spec))
    for group in ['objects','structures','navigation_geometry']:
        for row in spec.get(group,[]):
            if row.get('mesh_uri'):row['mesh_uri']=str((folder/row['mesh_uri']).resolve())
    save(out/'scene.json',spec)
    result(out,scene=str(out/'scene.json'),floor_height_m=spec['floor']['height_m'],objects=len(mapping),
        residual_faces=int((targets==0).sum()),all_faces_preserved=True,planning_profile=str(folder),
        external_artifacts=[str(folder/'checksums.json')]+[str(folder/p) for p in read(folder/'checksums.json')])


def plan(job):
    folder,_,_,_,_=checked_profile(job['config'],job['capture'])
    out=Path(job['output'])
    # An isolated process prevents profile/native module names contaminating
    # the integration's pinned implementations used by other captures.
    from .planning_policy import effective_planning_config
    effective=effective_planning_config(job['config'],read(folder/'config.json'))
    save(out/'effective_planning_config.json',effective)
    subprocess.run([sys.executable,str(ROOT/'spacewatch_pipeline/profile_planning_runner.py'),
        '--profile',str(folder),'--config',str(out/'effective_planning_config.json'),'--scene',
        job['dependencies'][job['capture']['id']+'.scene']['scene'],'--output',str(out/'plan')],check=True,
        timeout=min(job['config']['runtime'].get('stage_timeout_seconds',86400),2700))
    p=read(out/'plan/tour_plan.json')
    result(out,plan=str(out/'plan/tour_plan.json'),requests=str(out/'plan/render_requests.json'),
        planning_status=p['planning_status'],viewpoints=len(p['viewpoints']),quality_status=p['quality_status'],
        input_coverage_status=p['input_coverage_status'],planning_profile=str(folder),
        external_artifacts=[str(folder/'checksums.json')]+[str(folder/name) for name in read(folder/'checksums.json')])


def validate_plan(job,capture,plan_path,requests_path):
    folder,*_=checked_profile(job['config'],capture)
    subprocess.run([job['config']['runtime']['planning_python'],str(folder/'integrated_runner.py'),
        '--validate',str(plan_path),'--requests',str(requests_path)],check=True,timeout=120)


def target_ids(rows):
    return {r.get('instance_id','region_'+str(r['id'])) for r in rows if r['kind']=='object_candidate'}
