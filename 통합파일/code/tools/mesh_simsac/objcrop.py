"""Object-centred narrow-FOV comparison views.

For each source frame, small objects found in the frame's own 2D segmentation get a
view aimed at them with a field of view fitted to their angular size. The object's
3D target point (first mesh hit) is then viewed with the same FOV from the nearest
frames of the other capture that share a keypoint, so both sides of a pair look at the
same place at the same scale even when the object exists in only one capture.
No known change, object ID or ground truth is used.
"""
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import label

from .core import camera_transform, transform

# Historical accepted-reference ranking, not an exact copy of the saved five-run
# means. Kept unchanged when no explicit current class binding is supplied.
# Strict paper mode below includes every supplied nonstructural/nonperson class.
IMPORTANCE = {'door': 5, 'fire extinguisher': 5, 'box': 5, 'fire hose cabinet': 5,
              'vacuum cleaner': 4, 'umbrella': 4, 'window': 3, 'chair': 3, 'table': 3,
              'water dispenser': 3, 'clothes': 3, 'sign': 3, 'ventilation grille': 3}
MIN_PIXELS, MAX_EXTENT_DEG, MIN_SEPARATION_DEG = 64, 45.0, 20.0
# The walking photographer fills the lower panorama; narrow crops there magnify unmasked residue.
MAX_DOWN_DEG, MAX_PHOTOGRAPHER_FRACTION = 90.0, 0.30
NON_TARGET_CLASSES = {'wall', 'walls', 'floor', 'ceiling', 'ceilings',
                      'person', 'people', 'capturer', 'photographer'}


def crop_importance(settings, labels):
    """Validate an explicit 1--5 class-mean binding; absence is legacy mode.

    Eligibility follows the supplied class list, never the legacy allowlist.
    No fallback rating is invented for an unprovided semantic class.
    """
    if 'crop_importance_by_class' not in settings:
        return dict(IMPORTANCE)
    values = settings['crop_importance_by_class']
    if not isinstance(values, dict) or not values:
        raise ValueError('crop_importance_by_class must be a nonempty class-to-mean mapping')
    if not isinstance(labels, (list, tuple)) or len(labels) != len(set(labels)):
        raise ValueError('Crop semantic vocabulary must contain unique class names')
    known, result = set(labels), {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or name.strip() != name or name not in known:
            raise ValueError('Crop importance class is not in the current semantic vocabulary: ' + str(name))
        if name.casefold() in NON_TARGET_CLASSES:
            raise ValueError('Structural/person class cannot be a paper object crop target: ' + name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 1 <= value <= 5:
            raise ValueError('Crop importance must be a finite 1--5 class mean: ' + name)
        result[name] = float(value)
    return result


def basis(forward):
    forward = forward/np.linalg.norm(forward)
    down = np.array([0., 1., 0.])-forward[1]*forward
    if np.linalg.norm(down) < 1e-3:
        down = np.array([0., 0., 1.])-forward[2]*forward
    down /= np.linalg.norm(down)
    # Same column convention as core.cube_rotations: right, down, forward.
    return np.column_stack([np.cross(down, forward), down, forward])


def plan_crops(sides, matrices, base, settings):
    labels = sorted({name for side in sides for name in side['spec'].get('segmentation_labels', [])})
    importance = crop_importance(settings, labels)
    per_frame = settings.get('crops_per_frame', 3)
    matches = settings.get('crop_matches', 2)
    max_distance = settings.get('crop_max_distance_u', 7.5)
    frames, scenes = [], []
    for side, common in zip(sides, matrices):
        vertices = transform(side['vertices_world'], common)
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(vertices.astype('f4')), o3d.core.Tensor(side['faces'].astype('u4')))
        scenes.append(scene)
        for frame in side['spec']['frames']:
            frames.append(dict(epoch=side['epoch'], id=str(frame['id']), keypoints=set(frame.get('keypoint_ids', [])),
                               pose=camera_transform(np.array(frame['world_from_camera']), common),
                               segmentation=frame.get('segmentation'), labels=side['spec'].get('segmentation_labels', [])))

    def first_hit(epoch, origin, direction):
        ray = np.concatenate([origin, direction])[None].astype('f4')
        return float(scenes[epoch].cast_rays(o3d.core.Tensor(ray))['t_hit'].numpy()[0])

    masks = {}

    def load(frame):
        """Class map and photographer mask."""
        key = (frame['epoch'], frame['id'])
        if key not in masks:
            with np.load(Path(base)/frame['segmentation'], allow_pickle=False) as z:
                classes = z['class_id'].astype('i4')
                person = z['person'].astype(bool)
            masks[key] = (classes, person)
        return masks[key]

    unit = {}

    def erp_directions(shape):
        if shape not in unit:
            h, w = shape
            y, x = np.mgrid[:h, :w]
            lon, lat = ((x+.5)/w-.5)*2*np.pi, ((y+.5)/h-.5)*np.pi
            unit[shape] = np.stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)], -1)
        return unit[shape]

    def photographer_free(frame, direction, fov):
        """Reject crops whose cone is more than 30% photographer (extended mask)."""
        if np.degrees(np.arcsin(np.clip(direction[1], -1, 1))) > MAX_DOWN_DEG:
            return False
        if not frame['segmentation']:
            return True
        _, person = load(frame)
        cone = erp_directions(person.shape)@direction >= np.cos(np.radians(fov/2))
        return bool(person[cone].mean() <= MAX_PHOTOGRAPHER_FRACTION) if cone.any() else True

    views, report, target, rejected = {}, [], 0, dict(source=0, matched=0)
    for frame in frames:
        if not frame['segmentation']:
            continue
        classes, _ = load(frame)
        with np.load(Path(base)/frame['segmentation'], allow_pickle=False) as z:
            person = z['person'].astype(bool)
        directions = erp_directions(classes.shape)
        candidates = []
        for index, name in enumerate(frame['labels']):
            if name not in importance:
                continue
            mask = (classes == index+1) & ~person
            if mask.sum() < MIN_PIXELS:
                continue
            components, count = label(mask)
            for k in range(1, count+1):
                component = components == k
                pixels = int(component.sum())
                if pixels < MIN_PIXELS:
                    continue
                d = directions[component].mean(0)
                d /= np.linalg.norm(d)
                extent = 2*float(np.degrees(np.arccos(np.clip(directions[component]@d, -1, 1))).max())
                if extent <= MAX_EXTENT_DEG:
                    candidates.append(dict(label=name, importance=importance[name], extent=extent, direction=d, pixels=pixels))
        chosen = []
        for c in sorted(candidates, key=lambda c: (-c['importance'], c['extent'], -c['pixels'])):
            if len(chosen) >= per_frame:
                break
            if all(np.degrees(np.arccos(np.clip(c['direction']@o['direction'], -1, 1))) >= MIN_SEPARATION_DEG for o in chosen):
                chosen.append(c)
        for c in chosen:
            pose = frame['pose']
            fov = float(np.clip(2.5*c['extent'], 35, 70))
            if not photographer_free(frame, c['direction'], fov):
                rejected['source'] += 1
                continue
            world = pose[:3, :3]@c['direction']
            depth = first_hit(frame['epoch'], pose[:3, 3], world)
            if not np.isfinite(depth):
                continue
            point = pose[:3, 3]+world*depth
            name = f'crop{target:04d}'
            views.setdefault(f"{frame['epoch']}:{frame['id']}", []).append(
                dict(name=name, rotation=basis(c['direction']).tolist(), fov=fov, target=target))
            others = []
            for other in frames:
                if other['epoch'] == frame['epoch'] or not (other['keypoints'] & frame['keypoints']):
                    continue
                offset = point-other['pose'][:3, 3]
                distance = float(np.linalg.norm(offset))
                if distance > max_distance:
                    continue
                local = other['pose'][:3, :3].T@(offset/distance)
                if not photographer_free(other, local, fov):
                    rejected['matched'] += 1
                    continue
                hit = first_hit(other['epoch'], other['pose'][:3, 3], offset/distance)
                occluded = bool(np.isfinite(hit) and hit < distance-settings['visibility_tolerance_u']*2)
                others.append((occluded, distance, other))
            matched = []
            for occluded, distance, other in sorted(others, key=lambda r: (r[0], r[1]))[:matches]:
                local = other['pose'][:3, :3].T@(point-other['pose'][:3, 3])
                views.setdefault(f"{other['epoch']}:{other['id']}", []).append(
                    dict(name=f'{name}m{len(matched)}', rotation=basis(local).tolist(), fov=fov, target=target))
                matched.append(dict(frame=other['id'], distance_u=distance, occluded=occluded))
            report.append(dict(target=target, epoch=frame['epoch'], frame=frame['id'], label=c['label'],
                               importance=c['importance'], extent_deg=c['extent'], fov_deg=fov, pixels=c['pixels'],
                               point=point.tolist(), matched=matched))
            target += 1
    return dict(views=views, report=dict(targets=report, settings=dict(per_frame=per_frame, matches=matches,
                max_distance_u=max_distance, min_pixels=MIN_PIXELS, max_extent_deg=MAX_EXTENT_DEG,
                min_separation_deg=MIN_SEPARATION_DEG, importance=importance,
                importance_source='current_class_mean_binding' if 'crop_importance_by_class' in settings else 'legacy_static_reference',
                max_down_deg=MAX_DOWN_DEG,
                max_photographer_fraction=MAX_PHOTOGRAPHER_FRACTION),
                rejected_for_photographer=rejected, view_count=sum(len(v) for v in views.values())))
