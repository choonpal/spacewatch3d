"""Diagnostic only: compare a patch with nearby geometric wall/floor support.

No object fitting, semantic classes, input modification or event relabeling.
Parallel walls and floors do not constrain translation along their intersection.
"""
import numpy as np

from .surface_geometry import BARYCENTRIC, compare_surfaces, face_geometry, triangle_scene
from .surface_materials import large_wall_planes


def fit_plane(points, weights):
    weights = weights / weights.sum()
    center = np.sum(points * weights[:, None], axis=0)
    _, vectors = np.linalg.eigh(((points-center)*weights[:, None]).T @ (points-center))
    normal = vectors[:, 0]
    return normal, center


def paired_profile(points, faces, selected, normal, anchor, height):
    """Common grid rays hit each independently selected structural surface."""
    import open3d as o3d
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, axis); u /= np.linalg.norm(u); v = np.cross(n, u)
    uv = [points[e][np.unique(faces[e][selected[e]])] @ np.array([u, v]).T for e in [0, 1]]
    low = np.maximum(*[q.min(0) for q in uv]); high = np.minimum(*[q.max(0) for q in uv])
    if np.any(high <= low): raise ValueError('Structural patches have no common footprint')
    grid = np.stack(np.meshgrid(*[np.arange(a, b, .035*height) for a,b in zip(low, high)]), axis=-1).reshape(-1,2)
    origins = grid[:, :1]*u + grid[:, 1:]*v + (anchor@n+2*height)*n
    rays = np.c_[origins, np.broadcast_to(-n, origins.shape)].astype('f4')
    depths = []
    for e in [0, 1]:
        scene = triangle_scene(points[e], faces[e][selected[e]])
        depths.append(scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)['t_hit'].numpy())
    valid = np.isfinite(depths[0]) & np.isfinite(depths[1])
    delta = depths[0][valid]-depths[1][valid]  # signed after minus before along n
    if len(delta) < 20: raise ValueError('Too few common structural grid observations')
    return {'normal': n.tolist(), 'common_samples': len(delta),
            'signed_offset_u': float(np.median(delta)),
            'absolute_p50_u': float(np.median(np.abs(delta))),
            'absolute_p90_u': float(np.quantile(np.abs(delta), .9)),
            'scatter_nmad_u': float(1.4826*np.median(np.abs(delta-np.median(delta)))),
            'triangles': [int(np.count_nonzero(m)) for m in selected]}


def supported_translation(profiles):
    normals = np.array([p['normal'] for p in profiles])
    shifts = np.array([p['signed_offset_u'] for p in profiles])
    u, singular, vt = np.linalg.svd(normals, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0]*.15))
    correction = -(vt[:rank].T @ ((u[:, :rank].T @ shifts)/singular[:rank]))
    return correction, rank, vt[rank:]


def diagnose(points, faces, parts, patch_ids, height):
    centers, normals, areas = zip(*[face_geometry(points[e], faces[e]) for e in [0, 1]])
    patch_vertices = np.concatenate([points[e][np.unique(faces[e][patch_ids[e]])] for e in [0, 1]])
    lo, hi = patch_vertices.min(0), patch_vertices.max(0); anchor = (lo+hi)/2
    selected = {'wall': [], 'floor': []}; wall_normals = []
    for e in [0, 1]:
        part = parts[e]; region = np.zeros(len(faces[e]), bool)
        region[part['face_start']:part['face_start']+part['faces']] = True
        c, n, area = centers[e], normals[e], areas[e]
        wall_planes = large_wall_planes(points[e], faces[e][region], height)
        plane = min(wall_planes, key=lambda p: abs(anchor@p['normal']+p['offset']))
        wall = (region & (np.abs(c@plane['normal']+plane['offset']) < .035*height)
                & (np.abs(n@plane['normal']) > .97)
                & (np.abs((c-anchor)@plane['axis']) < .80*height)
                & (c[:, 1] > hi[1]+.08*height) & (c[:, 1] < .98*height))
        near = np.linalg.norm((c-anchor)[:, [0, 2]], axis=1) < .80*height
        outside_object = ~((c[:, 0] > lo[0]-.10*height) & (c[:, 0] < hi[0]+.10*height)
                           & (c[:, 2] > lo[2]-.10*height) & (c[:, 2] < hi[2]+.10*height))
        floor = region & near & outside_object & (np.abs(n[:, 1]) > .97) & (c[:, 1] < .12*height)
        ids = np.flatnonzero(floor)
        if len(ids) < 30 or wall.sum() < 30: raise ValueError('Insufficient nearby wall/floor geometry')
        bins = np.floor(c[ids, 1]/(.035*height)).astype(int)
        values = np.unique(bins); mode = max(values, key=lambda b: area[ids[bins==b]].sum())
        floor &= np.abs(c[:, 1]-(mode+.5)*.035*height) < .045*height
        wn, _ = fit_plane(c[wall], area[wall]); wn *= np.sign(wn@plane['normal'])
        wall_normals.append(wn); selected['wall'].append(wall); selected['floor'].append(floor)
    profiles = []
    for name, normal in [('wall', wall_normals[0]), ('floor', np.array([0., 1., 0.]))]:
        profile = paired_profile(points, faces, selected[name], normal, anchor, height)
        profile['surface'] = name; profiles.append(profile)
    correction, rank, null = supported_translation(profiles)
    raw, corrected, along_null = [], [], []
    for e in [0, 1]:
        p = points[e] + (correction if e else 0)
        target = points[1-e] + (correction if 1-e else 0)
        ids = np.asarray(patch_ids[e], int)
        d, _, nearest = compare_surfaces(points[e], faces[e][ids], points[1-e], faces[1-e])
        dc, _, _ = compare_surfaces(p, faces[e][ids], target, faces[1-e])
        raw.append(float(np.median(d))); corrected.append(float(np.median(dc)))
        vectors = nearest-centers[e][ids]
        along_null.append(float(np.median(np.linalg.norm(vectors@null.T, axis=1))))
    return {'method': 'nearby geometric wall/floor; common-grid ray profiles; minimum-norm translation only',
            'diagnostic_only': True, 'map_modified': False, 'object_fit': False,
            'structure_profiles': profiles, 'correction_after_to_before_u': correction.tolist(),
            'translation_rank': rank, 'unconstrained_directions': null.tolist(),
            'raw_surface_difference_u': float(np.median(raw)), 'raw_per_epoch_u': raw,
            'after_structure_correction_u': float(np.median(corrected)), 'corrected_per_epoch_u': corrected,
            'unconstrained_surface_difference_p50_u': float(np.median(along_null)),
            'selection': {'horizontal_radius_h': .80, 'wall_above_patch_h': .08,
                          'floor_object_margin_h': .10, 'semantic_labels_used': False},
            'selection_faces': {name:[np.flatnonzero(m).tolist() for m in masks] for name,masks in selected.items()}}
