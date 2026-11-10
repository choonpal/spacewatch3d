"""Geometry-only surface change patches. No instance/class/image detector inputs.

The threshold is a reconstruction-unit distance, never a probability. Alignment
is frozen upstream; this module does not fit objects or alter either source map.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Policy:
    sample_quorum: float = .75
    neighborhood_quorum: float = .50
    neighborhood_radius_h: float = .065
    minimum_faces: int = 6
    minimum_area_h2: float = .0064
    minimum_span_h: float = .09
    join_radius_h: float = .035
    stable_normal_cosine: float = .94
    minimum_stable_samples: int = 80
    minimum_own_visibility: float = .30
    minimum_free_fraction: float = .35
    minimum_finite_fraction: float = .80
    minimum_view_baseline_h: float = .06
    minimum_frame_gap: int = 4
    minimum_views: int = 2


POLICY = Policy()
BARYCENTRIC = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])


def face_geometry(points, faces):
    triangles = points[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return triangles.mean(1), cross/np.maximum(length[:, None], 1e-12), length/2


def triangle_scene(points, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(points, dtype='f4')),
                        o3d.core.Tensor(np.asarray(faces, dtype='u4')))
    return scene


def compare_surfaces(points, faces, target_points, target_faces, scene=None, target_normals=None):
    """Four interior samples per source triangle against actual target triangles."""
    import open3d as o3d
    if scene is None: scene = triangle_scene(target_points, target_faces)
    samples = np.einsum('sj,fjk->fsk', BARYCENTRIC, points[faces])
    near = scene.compute_closest_points(o3d.core.Tensor(samples.reshape(-1, 3).astype('f4')), nthreads=4)
    closest = near['points'].numpy().reshape(samples.shape)
    distances = np.linalg.norm(samples-closest, axis=2)
    ids = near['primitive_ids'].numpy().reshape(len(faces), -1)
    if target_normals is None: _, target_normals, _ = face_geometry(target_points, target_faces)
    return distances, target_normals[ids[:, 0]], closest[:, 0]


def error_baseline(distances, normals, target_normals, area, registration_error, height, policy=POLICY):
    """Use only nearby, parallel surfaces to estimate excess noise.

    The supplied structural registration tolerance is a floor. The robust
    within-match scatter can increase it; it cannot silently lower that floor.
    """
    center_distance = distances[:, 0]
    stable = ((center_distance <= registration_error) &
              (np.abs((normals*target_normals).sum(1)) >= policy.stable_normal_cosine) &
              (area > 1e-12))
    inside_triangle_spread = np.median(np.abs(distances-distances[:, :1]), axis=1)
    noise = float(np.median(inside_triangle_spread[stable])) if stable.sum() >= policy.minimum_stable_samples else 0.
    # Sampling variation and local mesh roughness only increase the frozen
    # alignment budget. Object boundaries are not used to learn this budget.
    threshold = float(np.hypot(registration_error, 2.5*noise))
    return {'registration_error_u': float(registration_error), 'mesh_noise_u': noise,
            'threshold_u': threshold, 'stable_triangles': int(stable.sum()),
            'stable_surface_supported': bool(stable.sum() >= policy.minimum_stable_samples),
            'height_u': float(height)}


def coherent_patches(points, faces, distances, threshold, height, policy=POLICY, excluded=None):
    """Cluster only discrepant faces, never connected unchanged furniture/floor."""
    centers, normals, area = face_geometry(points, faces)
    seed = ((distances > threshold).mean(1) >= policy.sample_quorum) & (area > 1e-12)
    if excluded is not None: seed &= ~np.asarray(excluded, dtype=bool)
    raw = int(seed.sum())
    tree = cKDTree(centers)
    k = min(24, len(centers))
    near_distance, near = tree.query(centers, k=k, distance_upper_bound=policy.neighborhood_radius_h*height)
    if k == 1:
        near_distance, near = near_distance[:, None], near[:, None]
    valid = np.isfinite(near_distance)
    safe = np.minimum(near, len(centers)-1)
    # Equal sample votes avoid a single coarse floor triangle vetoing a small
    # chair surface. Mesh area, support size and physical span are checked below.
    support = (seed[safe]*valid).sum(1)/np.maximum(valid.sum(1), 1)
    keep = seed & (support >= policy.neighborhood_quorum)
    ids = np.flatnonzero(keep)
    if not len(ids):
        return [], {'raw_faces': raw, 'coherent_faces': 0, 'retained_faces': 0}
    # Shared vertices link only changed triangles; a common chair/floor mesh
    # does not connect regions through unchanged faces.
    vertex_faces = {}
    edges = []
    for local, fi in enumerate(ids):
        for v in faces[fi]:
            v = int(v)
            if v in vertex_faces: edges.append((local, vertex_faces[v]))
            else: vertex_faces[v] = local
    # Also connect short gaps caused by disconnected original triangle pieces.
    for a, b in cKDTree(centers[ids]).query_pairs(policy.join_radius_h*height):
        edges.append((a, b))
    if edges:
        a, b = np.asarray(edges).T
        graph = coo_matrix((np.ones(len(a)*2), (np.r_[a,b], np.r_[b,a])), shape=(len(ids), len(ids)))
        count, labels = connected_components(graph, directed=False)
    else:
        count, labels = len(ids), np.arange(len(ids))
    patches = []
    for group in range(count):
        fi = ids[labels == group]
        vertices = np.unique(faces[fi]); bounds = np.array([points[vertices].min(0), points[vertices].max(0)])
        surface_area = float(area[fi].sum())
        if (len(fi) < policy.minimum_faces or surface_area < policy.minimum_area_h2*height**2 or
                np.linalg.norm(bounds[1]-bounds[0]) < policy.minimum_span_h*height):
            continue
        patches.append({'face_ids': fi, 'bounds': bounds, 'area_u2': surface_area,
                        'distance_p50_u': float(np.median(distances[fi])),
                        'distance_p90_u': float(np.quantile(distances[fi], .9)),
                        'threshold_ratio': float(np.median(distances[fi])/threshold),
                        'coherence': float(np.median(support[fi]))})
    return patches, {'raw_faces': raw, 'coherent_faces': int(keep.sum()),
                     'retained_faces': sum(len(p['face_ids']) for p in patches)}


def independent_views(views, field, minimum, height, policy=POLICY):
    selected = []
    for v in sorted(views, key=lambda v: -v['score']):
        if v[field] < minimum or v['finite_fraction'] < policy.minimum_finite_fraction:
            continue
        if all(abs(v['frame']-w['frame']) >= policy.minimum_frame_gap and
               np.linalg.norm(np.array(v['camera_in_map'])-w['camera_in_map']) >= policy.minimum_view_baseline_h*height
               for w in selected):
            selected.append(v)
    return selected


def classify_patch(sides, alignment_supported, height, policy=POLICY):
    """A missing opposite surface is evidence only if cameras see through it.

    Ray evidence comes from the reconstructions, not independent RGB-D truth.
    These outputs intentionally remain change *suspicions*, never identities.
    """
    own_counts, free_counts = [], []
    for side in sides:
        own_counts.append(len(independent_views(side['own_views'], 'visible_fraction', policy.minimum_own_visibility, height, policy)))
        free_counts.append(len(independent_views(side['opposite_views'], 'free_fraction', policy.minimum_free_fraction, height, policy)))
    if not alignment_supported:
        return 'deferred', 'alignment_underconstrained', own_counts, free_counts
    if any(a >= policy.minimum_views and b >= policy.minimum_views for a, b in zip(own_counts, free_counts)):
        return 'candidate', 'coherent_surface_difference', own_counts, free_counts
    if not any(a >= policy.minimum_views for a in own_counts):
        return 'deferred', 'insufficient_surface_observation', own_counts, free_counts
    return 'deferred', 'occlusion_or_incomplete_coverage', own_counts, free_counts
