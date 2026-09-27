"""Validate JSON structure and the IDs that cross task boundaries."""

import csv
import json
import math
from collections import Counter
from functools import lru_cache
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator


class ContractError(ValueError):
    pass


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(files('spacewatch3d').joinpath('schemas/artifact.schema.json').read_text(encoding='utf-8'))


def _fail(message: str) -> None:
    raise ContractError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream, parse_constant=lambda value: _fail(f'Non-finite JSON number: {value}'))
    _require(isinstance(data, dict), f'{path}: JSON object required')
    return data


def resolve(artifact: Path, relative: str) -> Path:
    """Every path is relative to its owning manifest, never the process cwd."""
    _require(not Path(relative).is_absolute() and not PureWindowsPath(relative).is_absolute() and '://' not in relative,
             f'Use a relative local path, got {relative!r}')
    return (artifact.parent / relative).resolve()


def _unique(rows: list[dict], key: str) -> dict:
    mapped = {row[key]: row for row in rows}
    _require(len(mapped) == len(rows), f'Duplicate {key}')
    return mapped


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _finite(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), 'Non-finite numeric value')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _affine(matrix: list, name: str) -> None:
    _require(all(abs(a-b) < 1e-8 for a, b in zip(matrix[3], [0, 0, 0, 1])),
             f'{name}: homogeneous last row must be [0, 0, 0, 1]')


REFERENCE_KINDS = {
    'source_manifest': 'video', 'frames_manifest': 'frames',
    'reconstruction_manifest': 'reconstruction', 'instances_manifest': 'instances',
    'objects_manifest': 'objects', 'baseline_objects_manifest': 'objects',
    'current_objects_manifest': 'objects',
}


def validate_artifact(path: str | Path, *, check_files: bool = False,
                      _stack: frozenset = frozenset(), _cache: dict | None = None) -> dict:
    """Check a manifest and its upstream manifests; optionally check asset files.

    This checks contracts, not geometric accuracy or model quality. Large binary
    arrays / mesh content are not decoded. Label CSVs are checked when requested.
    """
    path = Path(path).resolve()
    cache = {} if _cache is None else _cache
    _require(path not in _stack, f'Cyclic manifest reference: {path}')
    if path in cache:
        return cache[path]
    data = read_json(path)
    _finite(data)
    kind = data.get('kind')
    definitions = schema()['$defs']
    _require(kind in ('video', 'frames', 'reconstruction', 'instances', 'objects', 'changes', 'keyframes'),
             f'{path}: unknown artifact kind {kind!r}')
    validator = Draft202012Validator({'$defs': definitions, '$ref': '#/$defs/' + kind})
    errors = list(validator.iter_errors(data))
    if errors:
        error = errors[0]
        _fail(f'{path}: {".".join(map(str, error.absolute_path))}: {error.message}')

    references = {}
    for key, expected in REFERENCE_KINDS.items():
        if key not in data:
            continue
        ref_path = resolve(path, data[key])
        target = validate_artifact(ref_path, check_files=check_files,
                                   _stack=_stack | {path}, _cache=cache)
        _require(target['kind'] == expected, f'{key}: expected {expected}')
        _require(target['scene_id'] == data['scene_id'], f'{key}: scene_id mismatch')
        expected_capture = data['baseline_capture_id'] if key == 'baseline_objects_manifest' else data['capture_id']
        _require(target['capture_id'] == expected_capture, f'{key}: capture_id mismatch')
        _require(not target['synthetic'] or data['synthetic'], 'Synthetic input must remain marked synthetic')
        references[key] = (ref_path, target)

    if kind == 'frames':
        _unique(data['frames'], 'frame_id')
    elif kind == 'reconstruction':
        frames = _unique(references['frames_manifest'][1]['frames'], 'frame_id')
        cameras = _unique(data['cameras'], 'frame_id')
        _require(bool(cameras), 'Reconstruction requires registered cameras')
        _require(cameras.keys() <= frames.keys(), 'Camera refers to an unknown frame_id')
        if data['coordinates']['units'] == 'm':
            _require(data['coordinates']['scale_to_meters'] == 1, 'Metre coordinates require scale_to_meters=1')
        for camera in cameras.values():
            _affine(camera['world_from_camera'], 'world_from_camera')
            _require((camera['depth_uri'] is None) == (camera['depth_convention'] is None),
                     'Depth path and depth convention must be provided together')
    elif kind == 'instances':
        recon = references['reconstruction_manifest'][1]
        instances = _unique(data['instances'], 'instance_id')
        _require(data['point_count'] == recon['point_count'], 'Point count/order must match reconstruction')
        _require(sum(row['point_count'] for row in instances.values()) <= data['point_count'],
                 'Instance point counts exceed the reconstruction')
        frame_ids = {c['frame_id'] for c in recon['cameras']}
        for row in instances.values():
            _require(set(row['evidence_frame_ids']) <= frame_ids, 'Unknown evidence frame_id')
            _require(row['class_name'] is not None or row['class_confidence'] is None,
                     'Unknown class must have null class_confidence')
        if check_files:
            _validate_labels(resolve(path, data['point_labels_uri']), data, instances)
    elif kind == 'objects':
        instances = _unique(references['instances_manifest'][1]['instances'], 'instance_id')
        objects = _unique(data['objects'], 'instance_id')
        _require(objects.keys() <= instances.keys(), 'Object has an unknown instance_id')
        for item in objects.values():
            _affine(item['world_from_object'], 'world_from_object')
            _require(all(a <= b for a, b in zip(item['bbox_min'], item['bbox_max'])), 'Invalid bounding box')
            _require(set(item['evidence_frame_ids']) <= set(instances[item['instance_id']]['evidence_frame_ids']),
                     'Object evidence must refer to instance evidence')
            _require(Path(item['mesh_uri']).suffix.lower() == '.' + item['mesh_format'], 'Mesh extension/format mismatch')
    elif kind == 'changes':
        _require(data['baseline_capture_id'] != data['capture_id'], 'Changes require two different captures')
        before = references['baseline_objects_manifest'][1]
        after = references['current_objects_manifest'][1]
        before_ids = {o['instance_id'] for o in before['objects']}
        after_ids = {o['instance_id'] for o in after['objects']}
        registration = data['registration']
        if registration['status'] == 'ok':
            _require(registration['baseline_from_current'] is not None, 'Successful registration requires a transform')
            _affine(registration['baseline_from_current'], 'baseline_from_current')
        else:
            _require(registration['baseline_from_current'] is None, 'Failed registration must have null transform')
            _require(all(c['type'] == 'unknown' for c in data['changes']), 'Unregistered captures cannot assert changes')
        frame_sets = {}
        for key in ('baseline_objects_manifest', 'current_objects_manifest'):
            p, obj = references[key]
            frames = _ancestor(p, obj, 'frames', cache)[1]
            frame_sets[obj['capture_id']] = {f['frame_id'] for f in frames['frames']}
        for event in data['changes']:
            b, a = event['before_instance_id'], event['after_instance_id']
            _require(b is None or b in before_ids, 'Unknown before_instance_id')
            _require(a is None or a in after_ids, 'Unknown after_instance_id')
            _require(b is not None or a is not None, 'A change needs at least one object')
            if event['type'] == 'added':
                _require(b is None and a is not None, 'Added objects exist only in current capture')
            elif event['type'] == 'removed':
                _require(b is not None and a is None, 'Removed objects exist only in baseline capture')
            elif event['type'] in ('moved', 'shape_changed', 'unchanged'):
                _require(b is not None and a is not None, 'Matched change needs both object IDs')
            for evidence in event['evidence']:
                _require(evidence['frame_id'] in frame_sets.get(evidence['capture_id'], set()), 'Unknown change evidence')
    elif kind == 'keyframes':
        obj_path, objects = references['objects_manifest']
        object_ids = {o['instance_id'] for o in objects['objects']}
        frame_path, frames = _ancestor(obj_path, objects, 'frames', cache)
        frame_map = _unique(frames['frames'], 'frame_id')
        selected = set()
        for item in data['keyframes']:
            _require(item['instance_id'] in object_ids, 'Unknown keyframe instance_id')
            _require(item['frame_id'] in frame_map, 'Unknown keyframe frame_id')
            pair = (item['instance_id'], item['frame_id'])
            _require(pair not in selected, 'Duplicate keyframe/object pair')
            selected.add(pair)
            frame = frame_map[item['frame_id']]
            for key in ('source_frame_index', 'timestamp_s', 'view_id'):
                _require(item[key] == frame[key], f'Keyframe {key} does not match source frame')
            _require(resolve(path, item['image_uri']) == resolve(frame_path, frame['image_uri']),
                     'Keyframe must select the original extracted frame')

    for key, value in _walk(data):
        values = value if key.endswith('_uris') else [value]
        if not (key.endswith('_uri') or key.endswith('_uris')):
            continue
        for name in values:
            if name is not None:
                asset = resolve(path, name)
                if check_files:
                    _require(asset.is_file(), f'Missing asset: {asset}')
    cache[path] = data
    return data


def _ancestor(path: Path, data: dict, kind: str, cache: dict) -> tuple[Path, dict]:
    parents = {'objects':'instances_manifest', 'instances':'reconstruction_manifest', 'reconstruction':'frames_manifest'}
    while data['kind'] != kind:
        path = resolve(path, data[parents[data['kind']]])
        data = cache[path]
    return path, data


def _validate_labels(path: Path, data: dict, instances: dict) -> None:
    counts = Counter()
    rows = 0
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ['point_index', 'instance_id'], 'Labels require point_index,instance_id columns')
        for i, row in enumerate(reader):
            _require(row['point_index'] == str(i), 'Labels must follow the original zero-based point order')
            identity = row['instance_id']
            _require(identity == '' or identity in instances, 'Label references an unknown instance_id')
            counts[identity] += 1
            rows += 1
    _require(rows == data['point_count'], 'Label row count differs from point_count')
    for identity, item in instances.items():
        _require(counts[identity] == item['point_count'], 'Instance point_count differs from label CSV')
