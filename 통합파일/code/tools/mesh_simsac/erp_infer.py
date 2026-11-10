"""Whole-panorama SimSaC input, read back into the prepared perspective views.

For a query view and its reference view, SimSaC compares the two source 360-degree (ERP) frames:
the reference panorama is rotated into the query camera's orientation (rotation only, from the
registered poses), both go to SimSaC whole, and the ERP change probability is sampled at the query
view's own pixel rays. Everything downstream (evidence, voting, classification) is unchanged.
One inference per (query frame, reference frame) pair, cached across that frame's views.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .core import cube_rotations, erp_project, rays

H, W = 512, 1024
_y, _x = np.mgrid[:H, :W]
_lon, _lat = ((_x+.5)/W-.5)*2*np.pi, ((_y+.5)/H-.5)*np.pi
GRID = np.stack([np.cos(_lat)*np.sin(_lon), np.sin(_lat), np.cos(_lat)*np.cos(_lon)], -1)
_cache, _images, _records = {}, {}, {}


def record(view, root=None):
    """The view's manifest record: in run.py the geometry array named 'face' shadows the face name."""
    if root is None:
        return view
    if not _records:
        import json
        _records.update({v['id']: v for v in json.loads((Path(root)/'manifest.json').read_text())['views']})
    return _records[view['id']]


def face_rotation(view, root=None):
    meta = record(view, root)
    rotation = meta.get('rotation')
    return np.asarray(rotation, float) if rotation is not None else cube_rotations()[meta['face']]


def camera_rotation(view, root=None):
    # prepare.py sets pose[:3,:3] = camera rotation @ face rotation.
    return np.asarray(view['pose'])[:3, :3]@face_rotation(view, root).T


def panorama(root, view):
    path = (Path(root).parent/'inputs'/view['source_image']).resolve()
    if path not in _images:
        _images[path] = np.asarray(Image.open(path).convert('RGB').resize((W, H), Image.Resampling.BILINEAR))
    return _images[path]


def erp_probability(model, root, query, reference):
    key = (query['epoch'], query['frame_id'], reference['epoch'], reference['frame_id'])
    if key not in _cache:
        relative = camera_rotation(reference, root).T@camera_rotation(query, root)
        aligned = erp_project(panorama(root, reference), GRID, relative)
        _cache[key] = model.predict(panorama(root, query), aligned).astype('f4')
    directions = rays(np.asarray(query['K']), query['valid'].shape)
    return erp_project(_cache[key], directions, face_rotation(query, root)).astype('f4')


def cache_size():
    return len(_cache)
