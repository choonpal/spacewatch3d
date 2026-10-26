"""Geometry-only M3C2 / GLS-core comparison on two already aligned meshes.

This Source Code Form is subject to the terms of the Mozilla Public License,
v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
one at https://mozilla.org/MPL/2.0/.

GLS equations checked against Ponca (MPL-2.0), commit
fd5150bf9079aabdb82a6f05e419b76111c1f788, orientedSphereFit.hpp and gls.h.
This is a NumPy reproduction of the descriptor core, NOT Palma's full detector.
No instance labels, object correspondences, or existing change masks are read.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

SCALES = np.linspace(.03, .60, 20)
ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
PACKED = np.dtype([('xyz', '<f4', (3,)), ('normal', '<f4', (3,)),
                   ('rgb', 'u1', (3,)), ('pad', 'u1'),
                   ('m3c2', '<f4', (4,)), ('gls', '<f4', (4,))])


def voxel_indices(points, spacing):
    return np.unique(np.floor(points / spacing).astype(np.int32), axis=0, return_index=True)[1]


def surface_samples(mesh, seed, spacing=.035):
    """Area-weighted barycentric sampling; independent of tessellation density."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    if not len(areas) or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('Mesh has no finite, non-degenerate surface')
    rng = np.random.default_rng(seed)
    count = min(1_200_000, max(100_000, int(areas.sum() / spacing**2 * 10)))
    face = rng.choice(len(faces), count, p=areas/areas.sum())
    uv = rng.random((count, 2)); u = np.sqrt(uv[:, 0]); v = uv[:, 1]
    bary = np.column_stack([1-u, u*(1-v), u*v])
    positions = np.einsum('ni,nij->nj', bary, triangles[face])
    keep = voxel_indices(positions, spacing)
    positions, face, bary = positions[keep], face[keep], bary[keep]
    mesh.compute_vertex_normals()
    normals = np.einsum('ni,nij->nj', bary, np.asarray(mesh.vertex_normals)[faces[face]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return positions, normals


def pca_normals(queries, points, mesh_normals):
    tree = cKDTree(points)
    nearest = tree.query(queries, workers=4)[1]
    result = mesh_normals[nearest].copy()
    quality = np.zeros(len(queries))
    for radius in (.16, .24):
        neighbors = tree.query_ball_point(queries, radius, workers=4)
        for i, ids in enumerate(neighbors):
            if len(ids) < 6:
                continue
            p = points[ids]; p = p-p.mean(0)
            values, vectors = np.linalg.eigh(p.T @ p / len(p))
            planarity = (values[1]-values[0]) / max(values[2], 1e-12)
            if planarity > quality[i]:
                normal = vectors[:, 0]
                result[i] = normal * (1 if normal @ mesh_normals[nearest[i]] >= 0 else -1)
                quality[i] = planarity
    return result, quality


def m3c2(queries, normals, a, b, *, radius=.10, half_depth=.65, registration_sigma=.02):
    """Signed cylinder-mean distance and classic nominal LoD95 expression.

    Returned columns: |distance|/LoD95, signed distance, LoD95, min count.
    Mesh samples are correlated; this LoD is a conditional demonstration bound,
    not a calibrated 95% confidence interval for this reconstructed scene.
    """
    stats = []
    for points in (a, b):
        tree = cKDTree(points)
        mean = np.full(len(queries), np.nan); variance = mean.copy(); counts = np.zeros(len(queries))
        for start in range(0, len(queries), 512):
            end = min(start+512, len(queries))
            for local, ids in enumerate(tree.query_ball_point(queries[start:end], np.hypot(radius, half_depth), workers=4)):
                i = start+local
                p = points[ids]-queries[i]
                along = p @ normals[i]
                chosen = (np.einsum('ij,ij->i', p, p)-along**2 <= radius**2) & (np.abs(along) <= half_depth)
                z = along[chosen]; counts[i] = len(z)
                if len(z) >= 6:
                    mean[i], variance[i] = z.mean(), z.var(ddof=1)
        stats.append((mean, variance, counts))
    distance = stats[1][0]-stats[0][0]
    lod = 1.96*(np.sqrt(stats[0][1]/np.maximum(stats[0][2], 1)+stats[1][1]/np.maximum(stats[1][2], 1))+registration_sigma)
    return np.column_stack([np.abs(distance)/lod, distance, lod, np.minimum(stats[0][2], stats[1][2])])


def gls_descriptors(queries, points, normals, scales=SCALES):
    """APSS oriented sphere fit, Pratt-normalized tau/kappa, raw fit fitness."""
    tree = cKDTree(points)
    output = np.full((len(queries), len(scales), 6), np.nan)
    for start in range(0, len(queries), 256):
        end = min(start+256, len(queries)); size = end-start
        lists = tree.query_ball_point(queries[start:end], float(max(scales)), workers=4)
        lengths = np.array([len(ids) for ids in lists])
        if not lengths.sum():
            continue
        rows = np.repeat(np.arange(size), lengths)
        ids = np.concatenate(lists).astype(int)
        p = points[ids]-queries[start:end][rows]; n = normals[ids]
        d2 = np.einsum('ij,ij->i', p, p); pn = np.einsum('ij,ij->i', p, n)
        for s, radius in enumerate(scales):
            w = np.maximum(1-d2/radius**2, 0)**2
            def sums(values):
                return np.bincount(rows, weights=w*values, minlength=size)
            sw = sums(np.ones(len(w))); safe = np.maximum(sw, 1e-15)
            mp = np.column_stack([sums(p[:, k])/safe for k in range(3)])
            mn = np.column_stack([sums(n[:, k])/safe for k in range(3)])
            pp = sums(d2)/safe; dotpn = sums(pn)/safe
            denominator = pp-np.einsum('ij,ij->i', mp, mp)
            uq = .5*(dotpn-np.einsum('ij,ij->i', mp, mn))/np.maximum(denominator, 1e-15)
            ul = mn-2*uq[:, None]*mp
            uc = -np.einsum('ij,ij->i', ul, mp)-uq*pp
            pratt2 = np.einsum('ij,ij->i', ul, ul)-4*uc*uq
            pratt = np.sqrt(np.maximum(pratt2, 1e-15))
            norm = np.linalg.norm(ul, axis=1)
            effective_n = sw**2/np.maximum(np.bincount(rows, weights=w*w, minlength=size), 1e-15)
            valid = (effective_n >= 6) & (denominator > 1e-8) & (pratt2 > .01) & (norm > .1)
            descriptor = np.column_stack([uc/pratt/radius, 2*uq/pratt*radius, 1-pratt2, ul/np.maximum(norm[:, None], 1e-15)])
            output[start:end, s] = np.where(valid[:, None], descriptor, np.nan)
    return output


def gls_field(positions, centers, first, second, scales=SCALES):
    """Palma Eq2-4: minimum containing scale per query, distance-weighted map.

    Occupied grid spacing is fixed at .30m rather than estimated Gabriel radius.
    Columns: raw ch_p, weighted agreement, valid query count, weighted delta.
    No exponential/sigmoid transform and no interpretation as probability.
    """
    second = second.copy()
    flip = np.einsum('nsi,nsi->ns', first[:, :, 3:], second[:, :, 3:]) < 0
    second[:, :, :2] *= np.where(flip, -1, 1)[:, :, None]
    delta = np.sum((first[:, :, :3]-second[:, :, :3])**2, axis=2)
    tree = cKDTree(centers); result = np.full((len(positions), 4), np.nan)
    for start in range(0, len(positions), 512):
        end = min(start+512, len(positions)); size = end-start
        neighbors = tree.query_ball_point(positions[start:end], float(scales[-1]), workers=4)
        lengths = np.array([len(ids) for ids in neighbors])
        if not lengths.sum():
            continue
        rows = np.repeat(np.arange(size), lengths); ids = np.concatenate(neighbors).astype(int)
        distance = np.linalg.norm(positions[start:end][rows]-centers[ids], axis=1)
        scale = np.minimum(np.searchsorted(scales, distance), len(scales)-1)
        values = delta[ids, scale]; valid = np.isfinite(values)
        rows, values, distance = rows[valid], values[valid], distance[valid]
        weights = scales[-1]-distance
        sw = np.bincount(rows, weights=weights, minlength=size)
        score = np.bincount(rows, weights=weights*values, minlength=size)/np.maximum(sw, 1e-15)
        agreement = np.bincount(rows, weights=weights*(values > .05), minlength=size)/np.maximum(sw, 1e-15)
        counts = np.bincount(rows, minlength=size)
        score[counts < 3] = np.nan
        result[start:end] = np.column_stack([score, agreement, counts, score])
    return result


def categories(m, g, normal_quality, nearest_other):
    # 0 little difference, 1 suspected, 2 strong evidence, 3 insufficient support.
    # These are our display rules, not native labels or ground truth.
    m_valid = np.isfinite(m[:, 0]) & (m[:, 3] >= 6)
    g_valid = np.isfinite(g[:, 0]) & (g[:, 2] >= 3)
    mc = np.where(m_valid, 0, 3).astype(np.uint8)
    gc = np.where(g_valid, 0, 3).astype(np.uint8)
    mc[m_valid & (np.abs(m[:, 1]) >= .035) & ((m[:, 0] >= .5) | (normal_quality < .25))] = 1
    mc[m_valid & (np.abs(m[:, 1]) >= .06) & (m[:, 0] >= 1) & (normal_quality >= .25)] = 2
    gc[g_valid & (g[:, 0] >= .025)] = 1
    gc[g_valid & (g[:, 0] >= .08) & (g[:, 1] >= .65) & (g[:, 2] >= 6) & (nearest_other >= .035)] = 2
    return mc, gc


def regions(queries, classes, fields):
    """Evidence components, split into 1.6m horizontal exploration tiles.

    Tiles are navigation aids, not object segmentation or independent events.
    The list is capped; every surface score remains visible on the map.
    """
    selected = np.flatnonzero((classes[0] == 2) | (classes[1] == 2))
    if not len(selected):
        return []
    grid = np.floor(queries[selected]/.14).astype(int); base = grid.min(0); grid -= base
    shape = grid.max(0)+1
    if np.prod(shape) > 10_000_000:
        raise ValueError('Comparison grid too large')
    occupied = np.zeros(shape, bool); occupied[tuple(grid.T)] = True
    components, _ = label(occupied, np.ones((3, 3, 3)))
    ids = components[tuple(grid.T)]; result = []
    for k in range(1, int(ids.max())+1):
        members = selected[ids == k]
        if len(members) < 8:
            continue
        p = queries[members]
        if np.max(np.ptp(p[:, :2], axis=0)) > 2.:
            cells = np.floor(p[:, :2]/1.6).astype(int)
            _, tile_ids = np.unique(cells, axis=0, return_inverse=True)
            patches = [members[tile_ids == t] for t in range(int(tile_ids.max())+1)]
        else:
            patches = [members]
        for patch in patches:
            if len(patch) < 8:
                continue
            p = queries[patch]
            finite_m = fields[0][patch, 1]; finite_m = finite_m[np.isfinite(finite_m)]
            finite_g = fields[1][patch, 0]; finite_g = finite_g[np.isfinite(finite_g)]
            result.append({'bounds': {'min': (p.min(0)-.08).tolist(), 'max': (p.max(0)+.08).tolist()},
                           'core_count': len(patch), 'm3c2_strong': int(np.sum(classes[0][patch] == 2)),
                           'gls_strong': int(np.sum(classes[1][patch] == 2)),
                           'm3c2_median_abs_m': float(np.median(np.abs(finite_m))) if len(finite_m) else None,
                           'gls_median_score': float(np.median(finite_g)) if len(finite_g) else None})
    result.sort(key=lambda r: -r['core_count'])
    for i, r in enumerate(result):
        r['id'] = i+1
    return result[:24]


def build(before, after, output, m3c2_cache=None, *, scene='Rescan scene_e · 0 → 2', display_rotation=ROTATION):
    import open3d as o3d
    started = time.time(); output.mkdir(parents=True, exist_ok=True)
    meshes = [o3d.io.read_triangle_mesh(str(p)) for p in (before, after)]
    for mesh in meshes:
        if not len(mesh.vertices) or not np.isfinite(np.asarray(mesh.vertices)).all():
            raise ValueError('Invalid input mesh')
    rotation=np.asarray(display_rotation,dtype=float)
    if rotation.shape!=(3,3) or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6) or np.linalg.det(rotation)<.999:
        raise ValueError('Display rotation must be a proper rigid rotation')
    all_vertices = np.vstack([np.asarray(mesh.vertices) @ rotation.T for mesh in meshes])
    origin = (all_vertices.min(0)+all_vertices.max(0))/2
    samples = []
    for epoch, mesh in enumerate(meshes):
        positions, normals = surface_samples(mesh, 20261015+epoch)
        samples.append((positions @ rotation.T-origin, normals @ rotation.T))
        print(f'Surface {epoch}: {len(positions)} samples', flush=True)
    union = np.vstack([p for p, n in samples]); queries = union[voxel_indices(union, .08)]
    print(f'{len(queries)} shared core locations; estimating normals', flush=True)
    trees = [cKDTree(p) for p, n in samples]
    distances = np.column_stack([t.query(queries, workers=4)[0] for t in trees])
    normals = np.zeros_like(queries); quality = np.zeros(len(queries))
    for epoch, (p, n) in enumerate(samples):
        mask = distances.argmin(1) == epoch
        normals[mask], quality[mask] = pca_normals(queries[mask], p, n)
    print('M3C2 cylinders', flush=True)
    if m3c2_cache is not None:
        previous = json.loads((m3c2_cache/'comparison.json').read_text(encoding='utf-8'))
        expected = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (before, after)]
        if previous['input_hashes'] != expected or previous['parameters']['surface_voxel_m'] != .035 or previous['parameters']['core_voxel_m'] != .08:
            raise ValueError('Cached M3C2 input/parameter mismatch')
        for key, value in {'normal_radii_m': [.16, .24], 'm3c2_radius_m': .10, 'm3c2_half_depth_m': .65, 'registration_sigma_m': .02}.items():
            if previous['parameters'][key] != value:
                raise ValueError('Cached M3C2 parameter mismatch')
        cached = np.load(m3c2_cache/'core-fields.npz')
        if not np.array_equal(queries, cached['xyz']) or not np.allclose(quality, cached['normal_quality']):
            raise ValueError('Cached core samples mismatch')
        m = cached['m3c2'].copy()
    else:
        m = m3c2(queries, normals, samples[0][0], samples[1][0])
    # Eq2-4 score mapping from shared occupied-grid queries to surface cores.
    cells = np.unique(np.floor(union/.30).astype(int), axis=0)
    centers = (cells+.5)*.30
    print(f'GLS descriptors: {len(centers)} occupied grid cells, 20 scales, both epochs', flush=True)
    descriptors = [gls_descriptors(centers, p, n) for p, n in samples]
    g = gls_field(queries, centers, *descriptors)
    mc, gc = categories(m, g, quality, distances.max(1))
    # Store per-core diagnostics separately, including validity, without NaN JSON.
    np.savez_compressed(output/'core-fields.npz', xyz=queries, normal_quality=quality, m3c2=m, gls=g, classes=np.column_stack([mc, gc]))
    displays = []
    for epoch, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices) @ rotation.T-origin
        nearest_dist, nearest = cKDTree(queries).query(vertices, workers=4)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals) @ rotation.T
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full(vertices.shape, .65)
        faces = np.asarray(mesh.triangles); v = faces.ravel()
        packed = np.zeros(len(v), PACKED)
        packed['xyz'], packed['normal'] = vertices[v], normals[v]
        packed['rgb'] = np.clip(colors[v]*255, 0, 255).astype('u1')
        ci = nearest[v]
        packed['m3c2'] = np.column_stack([np.nan_to_num(m[ci, 0]), np.nan_to_num(m[ci, 1]), mc[ci], quality[ci]])
        packed['gls'] = np.column_stack([np.nan_to_num(g[ci, 0]), np.nan_to_num(g[ci, 1]), gc[ci], g[ci, 2]])
        packed['m3c2'][nearest_dist[v] > .14, 2] = 3
        packed['gls'][nearest_dist[v] > .14, 2] = 3
        name = f'{epoch}-comparison.bin'; packed.tofile(output/name)
        areas = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)/2
        summaries = {}
        for method, classes in [('m3c2', mc), ('gls', gc)]:
            # Area-weighted surface fractions; not sampling-count percentages.
            vertex_classes = np.where(nearest_dist > .14, 3, classes[nearest])
            face_class = vertex_classes[faces]
            summaries[method] = {str(k): float(np.sum(areas*np.mean(face_class == k, axis=1))) for k in range(4)}
        displays.append({'url': '/mesh-comparison/'+name, 'count': len(v), 'stride': PACKED.itemsize,
                         'triangles': len(faces), 'area_m2': float(areas.sum()), 'areas': summaries})
    data = {'kind': 'geometry_method_comparison', 'scene': scene,
            'input_hashes': [hashlib.sha256(p.read_bytes()).hexdigest() for p in (before, after)],
            'alignment': 'Both cached meshes already share the existing refined world frame; no second transform applied.',
            'display_origin': origin.tolist(), 'display_rotation': rotation.tolist(),
            'bounds': {'min': (all_vertices.min(0)-origin).tolist(), 'max': (all_vertices.max(0)-origin).tolist()},
            'sample_counts': [len(p) for p, n in samples], 'core_count': len(queries),
            'parameters': {'surface_voxel_m': .035, 'core_voxel_m': .08, 'normal_radii_m': [.16, .24],
                'm3c2_radius_m': .10, 'm3c2_half_depth_m': .65, 'registration_sigma_m': .02,
                'gls_radii_m': SCALES.tolist(), 'gls_grid_m': .30, 'gls_strong_score': .08, 'gls_suspect_score': .025},
            'methods': {'m3c2': 'Own implementation of cylinder mean distance / nominal LoD95. Mesh sample correlation and alignment uncertainty are not calibrated.',
                'gls': 'Palma Eq2-4 scoring core reproduction: APSS, 20 linear scales, minimum containing scale per query, distance-weighted mapping. Fixed .30m occupied grid instead of density-derived spacing. Not full detector: no LoOP, Gabriel-radius preprocessing, consistency propagation, graph consolidation or visibility inference.'},
            'rules': {'m3c2': 'Strong: |d|>=.06m, |d|/LoD>=1, PCA planarity>=.25. Suspect: |d|>=.035m and (ratio>=.5 or weak normal). Support<6: undefined.',
                'gls': 'Strong: raw ch_p>=.08, delta>.05 at >=65% query weight, >=6 supported query votes, nearest other surface>=.035m. Suspect: raw ch_p>=.025. Fewer than 3 supported queries: undefined.'},
            'caveat': 'Display thresholds are demonstration rules, not calibrated probabilities, accuracy claims or ground-truth change labels. Missing geometry alone cannot establish removal. Mesh artifacts may also be highlighted.',
            'region_grouping': 'Connected strong-evidence cores (.14m grid), split at 1.6m horizontal tiles when extent exceeds 2m; navigation aids, not object instances or event counts.',
            'displays': displays, 'regions': regions(queries, (mc, gc), (m, g)),
            'counts': {'m3c2': np.bincount(mc, minlength=4).tolist(), 'gls': np.bincount(gc, minlength=4).tolist()},
            'elapsed_s': round(time.time()-started, 2)}
    (output/'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'counts': data['counts'], 'regions': len(data['regions']), 'elapsed_s': data['elapsed_s']}), flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m3c2-cache', type=Path, help='Previous run directory; reuse only after input hashes, core coordinates and M3C2 parameters match')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory; source and previous runs are preserved.')
    build(args.before, args.after, args.output, args.m3c2_cache)
