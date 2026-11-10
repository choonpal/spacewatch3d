"""Surface exclusions, independent of object instances and change decisions.

Large common vertical planes are geometric. Reflective pane polygons are
explicit human material annotations, not an automatic reflection detector.
"""
import numpy as np
from .surface_geometry import BARYCENTRIC, face_geometry


def large_wall_planes(points, faces, height):
    centers, normals, area = face_geometry(points, faces)
    remaining = (np.abs(normals[:, 1]) < .25) & (area > 1e-10)
    rng = np.random.default_rng(931); result = []
    for _ in range(10):
        ids = np.flatnonzero(remaining)
        if len(ids) < 60: break
        proposals = rng.choice(ids, min(90, len(ids)), replace=False, p=area[ids]/area[ids].sum())
        best = None; score = 0.
        for i in proposals:
            mask = remaining & (np.abs((centers-centers[i])@normals[i]) < .025*height) & (np.abs(normals@normals[i]) > .94)
            value = area[mask].sum()
            if value > score: score, best = value, mask
        if best is None or score < .20*height**2: break
        remaining[best] = False
        q = centers[best]; weights = area[best]; mean = np.average(q, axis=0, weights=weights)
        _, vectors = np.linalg.eigh(((q-mean)*weights[:, None]).T@(q-mean))
        n = vectors[:, 0]
        if abs(n[1]) > .25: continue
        u = np.array([n[2], 0., -n[0]]); u /= np.linalg.norm(u)
        xy = np.c_[q@u, q[:, 1]]; lo, hi = np.quantile(xy, [.01, .99], axis=0)
        if hi[0]-lo[0] < .35*height or hi[1]-lo[1] < .45*height: continue
        result.append({'normal': n, 'offset': float(-mean@n), 'axis': u,
                       'low': lo, 'high': hi, 'area_u2': float(score),
                       'band_u': float(max(.012*height, min(.03*height, np.quantile(np.abs((q-mean)@n), .95)*2)))})
    return result


def common_wall_exclusions(points, faces, height, error):
    planes = [large_wall_planes(points[e], faces[e], height) for e in [0, 1]]
    anchor = np.mean([np.median(points[e][np.unique(faces[e])], axis=0) for e in [0, 1]], axis=0)
    matches = []; masks = [np.zeros(len(f), bool) for f in faces]
    for a in planes[0]:
        for b in planes[1]:
            cosine = float(a['normal']@b['normal']); sign = 1 if cosine >= 0 else -1
            # Compare planes near this room, not their world-origin intercepts.
            # Tiny normal differences otherwise become huge offsets in R12.
            separation = abs(a['offset']-sign*b['offset']+anchor@(a['normal']-sign*b['normal']))
            if abs(cosine) < .98 or separation > 1.5*error: continue
            # Check overlap in the first plane's horizontal coordinate.
            frame = np.vstack([b['normal'], b['axis'], [0., 1., 0.]])
            rhs = np.array([[-b['offset'], u, y] for u in [b['low'][0], b['high'][0]] for y in [b['low'][1], b['high'][1]]])
            projected = np.linalg.solve(frame, rhs.T).T@a['axis']
            blo, bhi = projected.min(), projected.max()
            if min(a['high'][0], bhi)-max(a['low'][0], blo) < .20*height: continue
            if min(a['high'][1], b['high'][1])-max(a['low'][1], b['low'][1]) < .30*height: continue
            matches.append((a, b))
    for pair in matches:
        for e, plane in enumerate(pair):
            centers, normals, _ = face_geometry(points[e], faces[e])
            n, d = plane['normal'], plane['offset']
            alignment = np.abs(normals@n)
            # Upper wall faces can retain smooth low-frequency reconstruction
            # bias. Use the room error there; low furniture keeps the tight band.
            band = np.where((centers[:, 1] > .72*height) & (alignment > .97),
                            max(plane['band_u'], error), plane['band_u'])
            uv = np.c_[centers@plane['axis'], centers[:, 1]]
            inside = ((uv >= plane['low']-.04*height) & (uv <= plane['high']+.04*height)).all(1)
            on_plane = (np.abs(centers@n+d) <= band) & (alignment >= .93)
            vertices_on = (np.abs(points[e][faces[e]]@n+d) <= 1.5*band[:, None]).all(1)
            masks[e] |= inside & on_plane & vertices_on
    serial = [[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()} for p in pair] for pair in matches]
    return masks, serial


def inside_convex_polygon(pixels, polygon):
    p = np.asarray(polygon, float); edge = np.roll(p, -1, axis=0)-p
    delta = pixels[:, None, :]-p[None, :, :]
    cross = edge[None, :, 0]*delta[:, :, 1]-edge[None, :, 1]*delta[:, :, 0]
    return (cross >= -1e-6).all(1) | (cross <= 1e-6).all(1)


def project(points, pose, transform):
    inv = np.linalg.inv(transform); raw = points@inv[:3, :3].T+inv[:3, 3]
    camera = (raw-pose[:3, 3])@pose[:3, :3]
    radius = np.linalg.norm(camera, axis=1)
    return np.c_[(np.arctan2(camera[:, 0], camera[:, 2])/(2*np.pi)+.5)*1036-.5,
                 (np.arcsin(np.clip(camera[:, 1]/np.maximum(radius, 1e-12), -1, 1))/np.pi+.5)*518-.5]


def pane_plane_from_border(scene, polygon, pose, transform, height):
    """Fit the physical frame/wall border, not a possibly reflected center."""
    import open3d as o3d
    polygon = np.array(polygon, float); middle = polygon.mean(0)
    enlarged = middle+(polygon-middle)*1.16
    uv = np.concatenate([enlarged[i]+np.linspace(0, 1, 24)[:, None]*(enlarged[(i+1)%len(enlarged)]-enlarged[i]) for i in range(len(enlarged))])
    yaw = ((uv[:, 0]+.5)/1036-.5)*2*np.pi; lat = ((uv[:, 1]+.5)/518-.5)*np.pi
    direction = np.c_[np.sin(yaw)*np.cos(lat), np.sin(lat), np.cos(yaw)*np.cos(lat)]@pose[:3, :3].T@transform[:3, :3].T
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    origin = pose[:3, 3]@transform[:3, :3].T+transform[:3, 3]
    hits = scene.cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(origin, direction.shape), direction].astype('f4')), nthreads=2)['t_hit'].numpy()
    good = np.isfinite(hits) & (hits > .1*height) & (hits < 5*height)
    q = origin+hits[good, None]*direction[good]
    if len(q) < 24: raise ValueError('Too little frame geometry for reflective exclusion')
    rng = np.random.default_rng(125); best = None
    for _ in range(180):
        a, b, c = q[rng.choice(len(q), 3, replace=False)]; n = np.cross(b-a, c-a); n /= max(np.linalg.norm(n), 1e-12)
        if abs(n[1]) > .25: continue
        mask = np.abs((q-a)@n) < .025*height
        if best is None or mask.sum() > best.sum(): best = mask
    if best is None or best.sum() < max(24, len(q)*.50): raise ValueError('Reflective frame plane is not supported')
    center = q[best].mean(0); _, _, vt = np.linalg.svd(q[best]-center, full_matrices=False); n = vt[-1]
    if (origin-center)@n < 0: n = -n
    return n, float(-center@n), int(best.sum())


def reflective_pane_exclusion(points, faces, polygon, pose, transform, plane, height, mode='reflective'):
    n, d = plane
    samples = np.einsum('sj,fjk->fsk', BARYCENTRIC, points[faces])
    pixels = project(samples.reshape(-1, 3), pose, transform)
    within = inside_convex_polygon(pixels, polygon).reshape(len(faces), -1)
    # Geometry materially in front of the pane (e.g. a chair) must survive.
    if mode == 'static_wall':
        _, normals, _ = face_geometry(points, faces)
        behind = (np.abs(samples@n+d) <= .075*height) & (np.abs(normals@n)[:, None] >= .90)
    else:
        behind = (samples@n+d) <= .025*height
    return (within & behind).mean(1) >= .75
