"""System adapter for fresh mesh/Gaussian inputs; keeps object-event contract honest."""
import os
from pathlib import Path
from spacewatch3d.contracts import ContractError, read_json, validate_artifact
from spacewatch3d.task_api import TaskRequest


def analyze(manifest, output, runtime=None, *, validate_only=False):
    """Direct system API: one pair manifest -> fresh 3D change-candidate result."""
    from tools.gaussian_change.pipeline import run
    from tools.gaussian_change.remote import run_remote
    config = read_json(Path(runtime)) if isinstance(runtime, (str, Path)) else (runtime or {})
    if config.get('remote_session') and not validate_only:
        return run_remote(manifest, output, config)
    return run(manifest, output, config, validate_only=validate_only)


def run(request: TaskRequest) -> Path:
    from tools.mesh_simsac.core import write
    if set(request.inputs) != {'before', 'after'}:
        raise ContractError('Task 5a requires before/after objects lineage manifests')
    before, after = [validate_artifact(request.inputs[p], check_files=True) for p in ['before', 'after']]
    if any(d['kind'] != 'objects' for d in [before, after]) or before['scene_id'] != after['scene_id'] or before['capture_id'] == after['capture_id']:
        raise ContractError('Supply different captures of the same scene')
    if not request.config.get('pair_manifest'):
        raise ContractError('mesh_gaussian_simsac requires pair_manifest')
    pair = read_json(Path(request.config['pair_manifest']))
    if pair.get('scene_id') != after['scene_id'] or any(pair.get(p, {}).get('capture_id') != data['capture_id']
                                                       for p, data in [('before', before), ('after', after)]):
        raise ContractError('Pair scene_id / before.capture_id / after.capture_id must match the objects lineage')
    out = request.output_dir.resolve()
    try:
        references = {p: os.path.relpath(request.inputs[p], out).replace('\\', '/') for p in ['before', 'after']}
    except ValueError as exc:
        raise ContractError('The shared contract requires relative lineage paths; choose output on the same filesystem drive as the objects manifests') from exc
    path = analyze(request.config['pair_manifest'], out/'analysis', request.config.get('runtime'))
    details = read_json(path)
    registration = read_json(path.parent/'registration.json')
    result = {'schema_version': '0.1.0', 'kind': 'changes', 'scene_id': after['scene_id'],
              'capture_id': after['capture_id'], 'baseline_capture_id': before['capture_id'],
              'synthetic': before['synthetic'] or after['synthetic'], 'producer': 'spacewatch3d.task_5a.mesh_gaussian_simsac',
              'baseline_objects_manifest': references['before'],
              'current_objects_manifest': references['after'],
              'parameters': {'backend': 'mesh_gaussian_simsac', 'candidate_manifest_uri': 'analysis/changes.json',
                             'regional_alignment_uri': 'analysis/registration.json', 'candidate_count': len(details['candidates']),
                             'output_scope': '3d_surface_change_candidates', 'object_events_evaluated': False,
                             'new_input_registration_fitted': True},
              'registration': {'status': 'ok', 'model': 'Sim3', 'baseline_from_current': registration['global_after_to_before'],
                               'fitness': None, 'rmse': None,
                               'reason': 'Fresh mesh global fit; final geometry uses linked regional SE3 corrections and residual bias'},
              'changes': []}
    write(out/'changes.json', result)
    validate_artifact(out/'changes.json', check_files=True)
    return out/'changes.json'
