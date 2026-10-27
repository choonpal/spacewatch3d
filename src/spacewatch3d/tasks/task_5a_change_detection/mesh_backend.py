"""System adapter for the same segmented-mesh/original-RGB worker as the viewer."""
import json
import os
from pathlib import Path
import zipfile

from spacewatch3d.contracts import ContractError, read_json, validate_artifact
from spacewatch3d.task_api import TaskRequest


def analyze(manifest, output, runtime=None, *, validate_only=False):
    from tools.mesh_simsac.bundle import inspect_bundle
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    _, files = inspect_bundle(manifest.parent, manifest.name)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory for this input pair')
    output.mkdir(parents=True, exist_ok=True)
    if validate_only:
        path = output/'validated-inputs.json'
        path.write_text(json.dumps(dict(state='validated_manifest_files', source_files=files,
            geometry_validated=False, inference_performed=False)), encoding='utf-8')
        return path
    config = read_json(Path(runtime)) if isinstance(runtime, (str, Path)) else (runtime or {})
    if config.get('remote_session'):
        if manifest.name != 'manifest.json' and 'manifest.json' in files:
            raise ValueError('manifest.json is reserved for the input manifest')
        bundle = output/'inputs.zip'
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name in files:
                z.write(manifest.parent/name, 'manifest.json' if name == manifest.name else name)
        from tools.mesh_simsac.remote_worker import execute
        execute(bundle, output, config)
    else:
        from tools.mesh_simsac.worker import execute
        execute(manifest, output, config)
    return output/'detection/results/events.json'


def run(request: TaskRequest) -> Path:
    if set(request.inputs) != {'before', 'after'}:
        raise ContractError('Task 5a requires before/after objects lineage manifests')
    before, after = [validate_artifact(request.inputs[p], check_files=True) for p in ['before', 'after']]
    if any(d['kind'] != 'objects' for d in [before, after]) or before['scene_id'] != after['scene_id'] or before['capture_id'] == after['capture_id']:
        raise ContractError('Supply different captures of the same scene')
    if not request.config.get('pair_manifest'):
        raise ContractError('mesh_rgb_simsac requires pair_manifest')
    pair = read_json(Path(request.config['pair_manifest']))
    if pair.get('scene_id') != after['scene_id'] or any(
        pair.get(p, {}).get('capture_id', pair.get(p, {}).get('id')) != d['capture_id']
        for p, d in [('before', before), ('after', after)]):
        raise ContractError('Pair scene/capture IDs must match the objects lineage')
    out = request.output_dir.resolve()
    try:
        references = {p: os.path.relpath(request.inputs[p], out).replace('\\', '/') for p in ['before', 'after']}
    except ValueError as error:
        raise ContractError('Choose output on the same filesystem drive as the lineage manifests') from error
    path = analyze(request.config['pair_manifest'], out/'analysis', request.config.get('runtime'))
    details = read_json(path)
    root = path.parent.parent
    registration = read_json(root/'registration.json')
    # Local surface-candidate IDs are not upstream object IDs. Keep the shared
    # contract honest and link the detailed results instead of inventing IDs.
    result = dict(schema_version='0.1.0', kind='changes', scene_id=after['scene_id'],
        capture_id=after['capture_id'], baseline_capture_id=before['capture_id'],
        synthetic=before['synthetic'] or after['synthetic'], producer='spacewatch3d.task_5a.mesh_rgb_simsac',
        baseline_objects_manifest=references['before'], current_objects_manifest=references['after'],
        parameters=dict(backend='mesh_rgb_simsac', candidate_manifest_uri='analysis/detection/results/events.json',
            viewer_manifest_uri='analysis/detection/viewer-data.json',
            regional_alignment_uri='analysis/detection/registration.json', candidate_count=len(details['events']),
            output_scope='mesh_rgb_change_candidates', object_events_evaluated=False,
            fresh_inference=True, original_rgb_used=True),
        registration=dict(status='ok', model='Sim3', baseline_from_current=registration['common_from_world'][1],
            fitness=None, rmse=None,
            reason='Global map-frame initializer; final comparison uses linked regional transforms and error bounds'),
        changes=[])
    out.mkdir(parents=True, exist_ok=True)
    (out/'changes.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    validate_artifact(out/'changes.json', check_files=True)
    return out/'changes.json'
