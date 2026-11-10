"""Gaussian-removal boundary ONLY; original 5.1.9 computation stays in place.

No replacement segmentation, person mask, mesh, denoising, visibility, objective,
selection, validation or cache implementation lives in this adapter.
"""
import copy
import subprocess
import sys
from pathlib import Path
from .io import ROOT, read, save, result


def adapt_stages(rows, config):
    """Replace train/synthetic-render dependencies with recorded source poses."""
    rgb=config.get('changes',{}).get('enabled',False) and config.get('changes',{}).get('backend')=='mesh_rgb_simsac'
    removed={r['name'] for r in rows if r['action'] in ('gs_prepare','gs_train')}
    if rgb:removed.update(('pair.align','pair.changes'))
    adapted=[]
    for original in rows:
        if original['name'] in removed:continue
        row=copy.deepcopy(original)
        row['dependencies']=[d for d in row['dependencies'] if d not in removed]
        prefix=row.get('capture')
        if row['action']=='plan':row['dependencies'].append(prefix+'.trajectory')
        if row['action']=='render':
            row['dependencies'].append(prefix+'.trajectory');row['gpu']=False
        if row['action'] in ('validate','publish','viewer'):
            row['dependencies'].extend(c['id']+'.trajectory' for c in config['captures'])
            if rgb:row['dependencies'].append('pair.changes')
        adapted.append(row)
        if row['action']=='reconstruct':
            adapted.append(dict(name=prefix+'.trajectory',action='trajectory',capture=prefix,
                dependencies=[row['name']],gpu=False))
    if rgb:
        # Registration/classification live in the pinned original RGB upstream,
        # never also run the obsolete Gaussian/legacy-mesh pair interpretation.
        for c in config['captures']:
            p=c['id']
            adapted.append(dict(name=p+'.ooops',action='rgb_predictions',capture=p,
                dependencies=[p+'.reconstruct','discovery'],gpu=False))
        deps=[c['id']+'.'+a for c in config['captures'] for a in ('reconstruct','semantic','scene','ooops')]
        if config.get('changes',{}).get('keypoint_nearby'):
            deps.extend(c['id']+'.plan' for c in config['captures'])
        adapted.extend([
            dict(name='pair.rgb_inputs',action='rgb_inputs',dependencies=deps,gpu=False),
            dict(name='pair.rgb_prepare',action='rgb_prepare',dependencies=['pair.rgb_inputs'],gpu=False),
            dict(name='pair.rgb_infer',action='rgb_infer',dependencies=['pair.rgb_prepare'],gpu=True),
            dict(name='pair.changes',action='rgb_finalize',dependencies=['pair.rgb_inputs','pair.rgb_infer'],gpu=False)])
    return adapted


def dispatch(job):
    action=job['action']
    if action=='rgb_predictions':
        import_predictions(job)
    elif action in ('rgb_inputs','rgb_prepare','rgb_infer','rgb_finalize'):
        from . import mesh_rgb
        if action=='rgb_inputs' and job['config'].get('changes',{}).get('keypoint_nearby'):
            from .keypoint_rgb import rgb_inputs
            rgb_inputs(job)
        else:getattr(mesh_rgb,action)(job)
    elif action=='trajectory':
        from .existing_path import trajectory
        payload=copy.deepcopy(job)
        payload['config']['media'].update(job['capture'].get('media',{}))
        trajectory(payload)
    elif action=='plan':plan(job)
    elif action=='render':
        from .capture_media import render
        render(job)
    else:return False
    return True


def import_predictions(job):
    """Bind unchanged model results to exact original RGB bytes, never old IDs."""
    from .io import sha,verify
    c=job['capture'];out=Path(job['output']);p=c['id']
    binding=c['captured_prediction']
    for name in ('inference','vocabulary'):
        verify(binding[name])
    original=read(binding['inference']['path'])
    source=job['dependencies'][p+'.reconstruct']['source_run']
    manifest_path=Path(source)/'input/manifest.json';manifest=read(manifest_path)
    if len(original['frames'])!=len(manifest['frames']):raise ValueError('Prediction/source frame count differs')
    external=[binding['inference']['path'],binding['vocabulary']['path'],str(manifest_path)]
    for i,(record,frame) in enumerate(zip(original['frames'],manifest['frames'])):
        path=Path(binding['predictions'])/f'{i:05d}.npz'
        if record['source_sha256']!=frame['sha256'] or record['prediction_sha256']!=sha(path):
            raise ValueError('Prediction does not belong to exact original RGB frame '+str(i))
        external.append(str(path))
    if sha(job['dependencies']['discovery']['vocabulary'])!=binding['vocabulary']['sha256']:
        raise ValueError('Prediction vocabulary differs from the semantic mesh vocabulary')
    value=dict(original,source_manifest_sha256=sha(manifest_path),
        verified_import=dict(original_inference=binding['inference'],original_source_manifest_sha256=original['source_manifest_sha256'],
            each_source_frame_rgb_sha256_equal=True,new_model_calls=0))
    save(out/'inference.json',value)
    result(out,predictions=binding['predictions'],vocabulary=binding['vocabulary']['path'],
        inference=str(out/'inference.json'),classes=[r['name'] for r in read(binding['vocabulary']['path'])],
        frames=len(manifest['frames']),verified_prediction_import=True,new_model_calls=0,external_artifacts=external)


def planning_options(config, profile_config=None):
    from .planning_policy import effective_planning_config
    options=effective_planning_config(config,profile_config)
    for name in ('object_relative_projected_quality','object_min_projected_pixels'):
        if name in config.get('planning',{}):options[name]=config['planning'][name]
    if options.get('object_relative_projected_quality') is False:
        options.pop('object_relative_projected_quality')
    options['max_link_m']=None
    options['lateral_candidate_refinement']=False
    if config.get('paper_integration',{}).get('enabled'):
        for name in ('space_weight','object_weight','coverage_repair','coverage_repair_runs',
                     'coverage_repair_sample_multiplier','independent_coverage_validation',
                     'validation_runs','validation_sample_multiplier','surface_samples',
                     'seed','max_triangles','tile_m','max_view_distance_m',
                     'shape_normalized_object_quality','solid_angle_target_sr'):
            if name in config.get('planning',{}):options[name]=config['planning'][name]
        options['route_aware_selection']=False
        options['paper_selection']=config['paper_integration']['selection']
    return options


def plan(job):
    """Profile-specific original loader/native caches run in their own process."""
    c=job['capture'];out=Path(job['output']);prefix=c['id']
    profile=None
    if c.get('planning_profile'):
        from .target_profile import checked_profile
        profile,*_=checked_profile(job['config'],c)
    options=planning_options(job['config'],read(profile/'config.json') if profile else None)
    if options.get('object_relative_projected_quality',False):
        # The imported trajectory already binds native dimensions to the exact
        # video hash; do not introduce a new optional ffprobe dependency.
        measured=read(job['dependencies'][prefix+'.trajectory']['trajectory'])
        width,height=measured['source_size']
        if (type(width) is not int or type(height) is not int or width!=2*height
                or Path(measured['video']).resolve()!=Path(c['video']).resolve()):
            raise ValueError('Relative object quality requires source-bound native ERP dimensions')
        options['object_observation_width']=width
        save(out/'relative_quality_input.json',dict(policy='object_relative_surface_and_projection_v1',
            source_video=c['video'],source_width=width,source_height=height,
            min_angular_equivalent_pixels=options.get('object_min_projected_pixels',64),
            guard_is_recognition_guarantee=False,physical_object_identity_verified=False))
    # These two options define virtual cameras, which have no original image.
    # Observation targets, scoring, seeds and sampling remain exactly original.
    scene=job['dependencies'][prefix+'.scene']['scene']
    trajectory=job['dependencies'][prefix+'.trajectory']['trajectory']
    save(out/'effective_planning_config.json',options)
    command=[sys.executable,'-m','spacewatch_pipeline.capture519_runner',
        '--scene',scene,'--trajectory',trajectory,'--output',str(out/'plan'),
        '--config',str(out/'effective_planning_config.json')]
    if profile:command+=['--profile',str(profile)]
    importance=job['dependencies'].get(prefix+'.importance',{}).get('importance_cache',c.get('importance_cache'))
    if importance:command+=['--importance',importance]
    if c.get('plan_validation_recovery'):command+=['--recover-draft',c['plan_validation_recovery']]
    subprocess.run(command,check=True,timeout=job['config']['runtime'].get('stage_timeout_seconds',2700))
    p=read(out/'plan/tour_plan.json')
    result(out,plan=str(out/'plan/tour_plan.json'),requests=str(out/'plan/render_requests.json'),
        planning_status=p['planning_status'],viewpoints=len(p['viewpoints']),
        quality_status=p['quality_status'],input_coverage_status=p['input_coverage_status'],
        media_origin='original_video_frame',original_planner='5.1.9 profile' if profile else '5.1.9 vendor',
        external_artifacts=[trajectory]+([importance] if importance else []))


def validate_plan(job,capture,plan_path,requests_path):
    command=[job['config']['runtime']['planning_python'],'-m','spacewatch_pipeline.capture519_validate',
        '--plan',str(plan_path),'--requests',str(requests_path)]
    if capture.get('planning_profile'):
        from .target_profile import checked_profile
        profile,*_=checked_profile(job['config'],capture)
        command+=['--profile',str(profile)]
    subprocess.run(command,check=True,timeout=120)
