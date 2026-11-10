"""Fresh mesh-only global alignment, then frozen-region rigid correction and bias.

Uses the original robust wall/floor ICP and residual estimator. This generic
adapter uses FPFH/RANSAC/ICP globally; it does not replay the experiment's saved
GeoTransformer transforms or its historical 13-room partition.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tools.mesh_simsac.core import transform, similarity
from tools.mesh_simsac.structural import rigid_icp, residuals, error_summary


def load_mesh(spec, count, seed):
    import open3d as o3d
    path = Path(spec['mesh'])
    labels = None
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            p = d['p'] if 'p' in d else d['vertices']
            f = d['faces']
            for key in ['face_semantic', 'face_class_id']:
                if key in d:
                    labels = d[key].copy()
                    break
    else:
        mesh = o3d.io.read_triangle_mesh(str(path))
        p, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    p = np.asarray(p, dtype=float)
    f = np.asarray(f)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Mesh vertices must be finite Nx3')
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(p):
        raise ValueError('Mesh requires valid triangle indices')
    p = transform(p, spec['world_from_mesh'])
    tri = p[f]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    if area.sum() <= 0:
        raise ValueError('Mesh has no nondegenerate triangles')
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(f), count, p=area/area.sum())
    uv = rng.random((count, 2)); a = np.sqrt(uv[:, 0])
    weights = np.c_[1-a, a*(1-uv[:, 1]), a*uv[:, 1]]
    q = np.einsum('ni,nij->nj', weights, tri[ix])
    n = cross[ix] / np.maximum(area[ix, None], 1e-12)
    if labels is not None:
        if labels.shape != (len(f),):
            raise ValueError('Face labels and faces differ')
        classes = labels[ix]
        # Existing MTPano mesh label convention: wall=0, floor=3.
        structural = np.isin(classes, [0, 3])
        origin = 'mesh semantic wall=0/floor=3'
    else:
        low, high = np.quantile(q[:, 1], [.02, .98]); h = high-low
        floor = (abs(n[:, 1]) > .9) & (q[:, 1] < low+.2*h)
        wall = (abs(n[:, 1]) < .2) & (q[:, 1] > low+.15*h) & (q[:, 1] < low+.85*h)
        classes = np.where(floor, 3, np.where(wall, 0, 255))
        structural = floor | wall
        origin = 'geometric normal/height proxy; semantic structure not supplied'
    if structural.sum() < 1000:
        raise ValueError('Too few wall/floor mesh samples for registration')
    return {'p': q, 'n': n, 'classes': classes, 'structural': structural,
            'structure_source': origin, 'vertices': len(p), 'triangles': len(f)}


def assign(points, centers):
    return cKDTree(np.asarray(centers)[:, [0, 2]]).query(np.asarray(points)[:, [0, 2]], workers=1)[1]


def scores(a, b, threshold):
    if min(len(a), len(b)) < 1:
        return {'overlap': 0., 'cost': 1., 'rmse': None}
    ds = [cKDTree(b).query(a, workers=1)[0], cKDTree(a).query(b, workers=1)[0]]
    d = np.concatenate(ds); ok = d < threshold
    return {'overlap': float(min(np.mean(x < threshold) for x in ds)),
            'cost': float(np.mean(np.minimum(d, threshold*3)**2)),
            'rmse': float(np.sqrt(np.mean(d[ok]**2))) if ok.any() else None}


def global_fit(a, b, height, minimum):
    import open3d as o3d
    reg = o3d.pipelines.registration
    voxel = .035*height
    clouds, features = [], []
    o3d.utility.random.seed(0)
    for p in [a, b]:
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(voxel)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=40))
        features.append(reg.compute_fpfh_feature(c, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*7, max_nn=100)))
        clouds.append(c)
    seeds = [np.eye(4)]
    for _ in range(2):
        r = reg.registration_ransac_based_on_feature_matching(
            clouds[1], clouds[0], features[1], features[0], True, voxel*3,
            reg.TransformationEstimationPointToPoint(True), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(.9), reg.CorrespondenceCheckerBasedOnDistance(voxel*3)],
            reg.RANSACConvergenceCriteria(50000, .999))
        seeds.append(r.transformation)
    candidates = []
    for seed in seeds:
        scaled = reg.registration_icp(clouds[1], clouds[0], .3*height, seed,
                                     reg.TransformationEstimationPointToPoint(True),
                                     reg.ICPConvergenceCriteria(max_iteration=80))
        result = reg.registration_icp(clouds[1], clouds[0], .18*height, scaled.transformation,
                                     reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=80))
        matrix = similarity(result.transformation)[0]
        quality = scores(a, transform(b, matrix), .05*height)
        candidates.append({'matrix': matrix.tolist(), **quality})
    candidates.sort(key=lambda c: (-c['overlap'], c['cost']))
    selected = candidates[0]
    ambiguous = any(c['overlap'] >= selected['overlap']-.02 and
                    np.median(np.linalg.norm(transform(b[::20], c['matrix'])-transform(b[::20], selected['matrix']), axis=1)) > .5*height
                    for c in candidates[1:])
    if selected['overlap'] < minimum or ambiguous:
        raise ValueError(f'Global mesh registration failed overlap/ambiguity gate: {selected["overlap"]:.3f}, ambiguous={ambiguous}')
    return np.array(selected['matrix']), {'method': 'mesh FPFH/RANSAC + similarity ICP + point-to-plane ICP',
                                         'candidates': candidates, 'ambiguous': ambiguous,
                                         'passed': True, 'ground_truth_accuracy_verified': False}


def register(meshes, settings, progress=lambda **_: None):
    h = settings['height_u']
    if h is None:
        h = float(np.diff(np.quantile(meshes[0]['p'][:, 1], [.02, .98]))[0])
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Invalid reference mesh height')
    # Three disjoint sample indices: fitting / bias calibration / validation.
    structural = []
    for d in meshes:
        mask = d['structural']
        structural.append({k: d[k][mask] for k in ['p', 'n', 'classes']})
    a, b = structural
    split = [np.arange(len(d['p'])) % 3 for d in structural]
    progress(stage='global_registration')
    g, global_report = global_fit(a['p'][split[0] == 0], b['p'][split[1] == 0], h, settings['min_overlap'])
    moved = transform(b['p'], g); rotation = g[:3, :3]/similarity(g)[1]
    bn = b['n'] @ rotation.T
    # A geometry-only frozen Voronoi partition. No candidate/keypoint locations.
    spacing = settings['region_width_heights']*h
    grid = np.floor(a['p'][:, [0, 2]]/spacing).astype(int)
    cells, counts = np.unique(grid, axis=0, return_counts=True)
    cells = cells[counts >= 100]
    if not len(cells):
        raise ValueError('No supported mesh partition cells')
    if len(cells) > settings['max_regions']:
        raise ValueError('Too many regions; increase region_width_heights or max_regions')
    centers = np.c_[(cells[:, 0]+.5)*spacing, np.zeros(len(cells)), (cells[:, 1]+.5)*spacing]
    z0, z1 = assign(a['p'], centers), assign(moved, centers)
    regions = []
    for z in range(len(centers)):
        progress(stage='regional_correction', completed=z, total=len(centers))
        ids = [np.flatnonzero((lab == z) & (sp == 0)) for lab, sp in [(z0, split[0]), (z1, split[1])]]
        aa, bb = a['p'][ids[0]], moved[ids[1]]
        baseline = scores(aa, bb, .05*h); local = np.eye(4)
        if min(len(aa), len(bb)) >= 150:
            best = baseline
            for weight in [.02, .12]:
                candidate, _ = rigid_icp(bb, aa, bn[ids[1]], a['n'][ids[0]], np.eye(4), h, weight)
                trial = scores(aa, transform(bb, candidate), .05*h)
                if trial['overlap'] >= baseline['overlap'] and trial['cost'] <= best['cost']:
                    local, best = candidate, trial
        def residual_field(part, matrix):
            ix = [np.flatnonzero((lab == z) & (sp == part)) for lab, sp in [(z0, split[0]), (z1, split[1])]]
            rows = residuals(a['p'][ix[0]], transform(b['p'][ix[1]], matrix), a['n'][ix[0]],
                             b['n'][ix[1]] @ (matrix[:3, :3]/similarity(matrix)[1]).T, a['classes'][ix[0]], b['classes'][ix[1]])
            return error_summary(rows, h)
        total = local @ g
        calibration = residual_field(1, total)
        bias = np.array(calibration.get('translation_bias_u', [0., 0., 0.])) if calibration.get('supported') else np.zeros(3)
        with_bias = total.copy(); with_bias[:3, 3] += bias
        # Bias is selected on its calibration split, never on object/change pixels.
        ci = [np.flatnonzero((lab == z) & (sp == 1)) for lab, sp in [(z0, split[0]), (z1, split[1])]]
        if scores(a['p'][ci[0]], transform(b['p'][ci[1]], with_bias), .05*h)['cost'] > scores(a['p'][ci[0]], transform(b['p'][ci[1]], total), .05*h)['cost']:
            bias[:] = 0; with_bias = total
        validation = residual_field(2, with_bias)
        vi = [np.flatnonzero((lab == z) & (sp == 2)) for lab, sp in [(z0, split[0]), (z1, split[1])]]
        val = scores(a['p'][vi[0]], transform(b['p'][vi[1]], with_bias), .05*h)
        supported = bool(validation.get('supported') and val['overlap'] >= settings['min_overlap'])
        regions.append({'id': f'R{z+1:02d}', 'matrix_after_to_common': with_bias.tolist(),
                        'local_matrix': local.tolist(), 'translation_bias_u': bias.tolist(),
                        'calibration': calibration, 'validation': validation, 'validation_overlap': val,
                        'supported': supported, 'fit_before': baseline,
                        'fit_after': scores(aa, transform(bb, local), .05*h)})
    return {'schema': 'mesh-regional-registration-v1', 'height_u': h, 'global': global_report,
            'global_after_to_before': g.tolist(), 'centers': centers.tolist(), 'regions': regions,
            'partition': 'mesh-derived XZ Voronoi cells frozen after global fit',
            'input_geometry': 'triangle meshes only; Gaussian means never used for fitting',
            'units': 'declared input world units; metric scale not inferred',
            'split': 'disjoint sample indices for fit, bias calibration, residual validation'}
