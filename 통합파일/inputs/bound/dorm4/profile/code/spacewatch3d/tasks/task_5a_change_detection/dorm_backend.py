"""Adapt prepared whole-mesh regions to the unchanged shared task contract.

The shared changes array describes object events. Surface patches have no
verified object IDs, so their actual results live in parameters' linked tour
manifest rather than being assigned invented instance IDs or motion labels.
"""
import json
import os
from pathlib import Path

from spacewatch3d.contracts import ContractError, read_json, resolve, validate_artifact
from spacewatch3d.task_api import TaskRequest


def run(request: TaskRequest) -> Path:
    if set(request.inputs) != {'before', 'after'}:
        raise ContractError('Task 5a requires before and after objects manifests')
    before, after = [validate_artifact(request.inputs[key], check_files=True)
                     for key in ('before', 'after')]
    if any(d['kind'] != 'objects' for d in (before, after)):
        raise ContractError('Task 5a requires objects lineage manifests')
    if before['scene_id'] != after['scene_id'] or before['capture_id'] == after['capture_id']:
        raise ContractError('Task 5a needs different captures of the same scene')
    if not request.config.get('source_dir'):
        raise ContractError('dorm_geometry_first requires source_dir for a prepared aligned mesh pair')
    source = Path(request.config['source_dir']).resolve()
    meta = read_json(source / 'maps/mesh-metadata.json')
    for key, map_key, data in [('before', 'reference', before), ('after', 'after', after)]:
        uri = data['parameters'].get('whole_mesh_uri')
        if not uri or resolve(request.inputs[key], uri) != (source / meta[map_key]['url']).resolve():
            raise ContractError(f'{key}.parameters.whole_mesh_uri must reference this source bundle\'s {map_key} mesh')
    output = request.output_dir.resolve()
    if output == source or output.is_relative_to(source):
        raise ContractError('Write task output outside the read-only source bundle')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError('Use a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    materials = Path(request.config['materials']).resolve() if request.config.get('materials') else None
    previous = Path(request.config['previous']).resolve() if request.config.get('previous') else None
    from .dorm_surface import main as publish
    publish(source, output / 'published', materials, previous)
    tour = read_json(output / 'published/tour.json')
    context = read_json(source / 'alignment-context.json')
    alignment_file = output / 'regional-alignment.json'
    alignment_file.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {
        'schema_version': '0.1.0', 'kind': 'changes', 'scene_id': after['scene_id'],
        'capture_id': after['capture_id'], 'baseline_capture_id': before['capture_id'],
        'synthetic': before['synthetic'] or after['synthetic'],
        'producer': 'spacewatch3d.task_5a.dorm_geometry_first',
        'baseline_objects_manifest': os.path.relpath(request.inputs['before'], output).replace('\\', '/'),
        'current_objects_manifest': os.path.relpath(request.inputs['after'], output).replace('\\', '/'),
        'parameters': {
            'backend': 'dorm_geometry_first', 'output_scope': 'surface_regions_not_object_events',
            'viewer_manifest_uri': 'published/tour.json', 'surface_audit_uri': 'published/surface-audit.json',
            'regional_alignment_uri': 'regional-alignment.json', 'summary': tour['summary'],
            'instance_segmentation_used_for_detection': False,
            'registration_scope': 'global initializer below; frozen regional transforms in regional_alignment_uri',
            'object_events_evaluated': False,
        },
        'registration': {
            'status': 'ok', 'model': 'Sim3', 'baseline_from_current': context['base_matrix'],
            'fitness': None, 'rmse': None,
            'reason': 'Prepared global initializer only. Final comparisons use the separate frozen regional transforms and error budgets; no new registration was fitted.',
        },
        'changes': [],
    }
    path = output / 'changes.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    validate_artifact(path, check_files=True)
    return path
