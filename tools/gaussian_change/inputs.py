"""Input contract shared by the CLI and system integration API. No planning/training."""
from pathlib import Path
import numpy as np
from tools.mesh_simsac.core import read, sha, similarity, camera_transform


def resolve(base, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def load_manifest(path):
    path = Path(path).resolve()
    doc = read(path)
    if doc.get('schema') != 'mesh-gaussian-change-input-v1':
        raise ValueError('Expected mesh-gaussian-change-input-v1')
    hashes = {'manifest': sha(path)}
    sides = []
    for phase in ['before', 'after']:
        spec = dict(doc[phase])
        for field in ['mesh', 'gaussian', 'cameras']:
            spec[field] = resolve(path.parent, spec[field])
            if not spec[field].is_file():
                raise FileNotFoundError(spec[field])
            hashes[f'{phase}.{field}'] = sha(spec[field])
        # These are within-capture coordinate bridges, never an inter-map fit.
        for field in ['world_from_mesh', 'world_from_gaussian', 'world_from_camera']:
            if field not in spec:
                raise ValueError(f'{phase}.{field} is required; use explicit identity if frames coincide')
            spec[field] = similarity(spec[field])[0]
        if spec.get('camera_convention') not in ['opencv', 'opengl']:
            raise ValueError('camera_convention must be opencv or opengl')
        cams = read(spec['cameras'])
        if cams.get('instance_segmentation'):
            spec['_instance_segmentation'] = cams['instance_segmentation']
        if spec.get('instance_observations'):
            spec['instance_observations'] = resolve(path.parent,spec['instance_observations'])
            hashes[f'{phase}.instance_observations'] = sha(spec['instance_observations'])
        rows = cams.get('requests', cams.get('cameras', cams.get('poses', [])))
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
            raise ValueError('Supply 1..2000 saved camera poses, without running a planner')
        parsed = []
        ids = set()
        for i, row in enumerate(rows):
            pose = row.get('camera_to_world', row.get('reference_from_camera'))
            if pose is None:
                raise ValueError('Camera requires camera_to_world or reference_from_camera')
            pose = camera_transform(pose, spec['world_from_camera'])
            if spec['camera_convention'] == 'opengl':
                pose = pose @ np.diag([1., -1., -1., 1.])
            key = str(row.get('viewpoint_id', row.get('id', i)))
            if key in ids:
                raise ValueError('Duplicate camera ID in a capture')
            ids.add(key)
            parsed.append({'id': f'{phase}:{key}', 'source_index': i, 'pose': pose})
        spec['views'] = parsed
        sides.append(spec)
    settings = {'height_u': None, 'samples': 90000, 'min_overlap': .2,
                'region_width_heights': 2., 'max_regions': 48,
                'probability_threshold': .5, 'alpha_threshold': .85,
                'width': 1024, 'height': 512, 'cube_size': 384,
                'max_views_per_capture': None, 'min_component_voxels': 5}
    supplied = doc.get('settings', {})
    if set(supplied) - set(settings):
        raise ValueError('Unknown settings: ' + str(sorted(set(supplied) - set(settings))))
    settings.update(supplied)
    for name in ['samples', 'max_regions', 'width', 'height', 'cube_size', 'min_component_voxels']:
        v = settings[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(name + ' must be a positive integer')
    if settings['samples'] < 3000 or settings['width'] != 2 * settings['height']:
        raise ValueError('Need >=3000 mesh samples and a 2:1 panorama')
    if settings['width'] > 4096 or settings['cube_size'] > 2048:
        raise ValueError('Render size exceeds supported limits')
    for name in ['min_overlap', 'probability_threshold', 'alpha_threshold']:
        if not 0 < settings[name] <= 1:
            raise ValueError(name + ' must lie in (0,1]')
    for name in ['height_u', 'region_width_heights']:
        v = settings[name]
        if v is not None and (not np.isfinite(v) or v <= 0):
            raise ValueError(name + ' must be finite and positive')
    limit = settings['max_views_per_capture']
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('max_views_per_capture must be null or a positive integer')
    if doc.get('world_up') != '+Y' or not doc.get('units'):
        raise ValueError('Declare shared unit convention and world_up=+Y using the coordinate bridges')
    return doc, sides, settings, hashes
