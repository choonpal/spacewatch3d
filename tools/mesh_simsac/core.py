"""Image/geometry operations shared by preparation, inference and CPU checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def transform(points, matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def similarity(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1]):
        raise ValueError("Expected a finite 4x4 homogeneous transform")
    scale = np.cbrt(np.linalg.det(m[:3, :3]))
    if scale <= 0 or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3) * scale**2, atol=1e-5):
        raise ValueError("Only orientation-preserving rigid/uniform-scale transforms are supported")
    return m, scale


def camera_transform(pose, matrix):
    pose, scale_pose = similarity(pose)
    if not np.isclose(scale_pose, 1):
        raise ValueError("world_from_camera must contain a rotation, without scale")
    matrix, scale = similarity(matrix)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3] @ pose[:3, :3] / scale
    result[:3, 3] = transform(pose[None, :3, 3], matrix)[0]
    return result


def cube_rotations():
    # OpenCV camera basis: x right, y down, z forward. ERP latitude is positive down.
    specs = {"front": ([0, 0, 1], [0, 1, 0]), "right": ([1, 0, 0], [0, 1, 0]),
             "back": ([0, 0, -1], [0, 1, 0]), "left": ([-1, 0, 0], [0, 1, 0]),
             "up": ([0, -1, 0], [0, 0, 1]), "down": ([0, 1, 0], [0, 0, -1])}
    return {key: np.column_stack([np.cross(down, forward), down, forward]).astype(float)
            for key, (forward, down) in specs.items()}


def intrinsics(width, height, fov=100):
    focal = (width - 1) / (2 * np.tan(np.deg2rad(fov) / 2))
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]], float)


def rays(K, shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T


def erp_project(image, directions, rotation, mask=False):
    d = directions @ rotation.T
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    h, w = image.shape[:2]
    u = (np.arctan2(d[..., 0], d[..., 2]) / (2 * np.pi) + .5) * w - .5
    v = (np.arcsin(np.clip(d[..., 1], -1, 1)) / np.pi + .5) * h - .5
    padded = np.pad(image, ((0, 0), (1, 1)) + ((0, 0),) * (image.ndim - 2), mode="wrap")
    coords = [np.clip(v, 0, h - 1), np.mod(u, w) + 1]
    if image.ndim == 2:
        return map_coordinates(padded, coords, order=0 if mask else 1, mode="nearest")
    return np.stack([map_coordinates(padded[..., k], coords, order=1, mode="nearest")
                     for k in range(image.shape[-1])], -1).astype("u1")


def project(points, pose, K, shape):
    xyz = (points - pose[:3, 3]) @ pose[:3, :3]
    z = xyz[..., 2]
    pixel = xyz @ K.T
    uv = pixel[..., :2] / np.maximum(z[..., None], 1e-10)
    finite = np.isfinite(uv).all(-1) & np.isfinite(z)
    safe = np.where(finite[..., None], uv, -1)
    # Check continuous coordinates before rounding to avoid a false image-edge hit.
    inside = finite & (z > 0) & (safe[..., 0] >= 0) & (safe[..., 0] <= shape[1] - 1)
    inside &= (safe[..., 1] >= 0) & (safe[..., 1] <= shape[0] - 1)
    ij = np.rint(np.clip(safe, -1, max(shape))).astype(int)
    return ij, z, inside


def project_in_view(points, view, regions=None):
    shape = view['depth'].shape
    if 'region_poses' not in view:
        return project(points, view['pose'], view['K'], shape)
    if regions is None:
        raise ValueError('Regional camera projection requires point region IDs')
    points, regions = np.asarray(points), np.asarray(regions)
    if regions.shape != points.shape[:-1]:
        raise ValueError('One region is required for every projected point')
    ij = np.zeros((*regions.shape, 2), int)
    z, inside = np.zeros(regions.shape), np.zeros(regions.shape, bool)
    for region in np.unique(regions):
        if region <= 0 or region >= len(view['region_poses']):
            continue
        mask = regions == region
        ij[mask], z[mask], inside[mask] = project(points[mask], view['region_poses'][region], view['K'], shape)
    return ij, z, inside


def counterpart_support(points, reference, tolerance, person=None, regions=None):
    """A nearer reference surface occludes; a farther one supports free-space evidence."""
    shape = reference["depth"].shape
    ij, z, inside = project_in_view(points, reference, regions)
    x = np.clip(ij[..., 0], 0, shape[1] - 1)
    y = np.clip(ij[..., 1], 0, shape[0] - 1)
    observed = inside & reference["valid"][y, x]
    if 'point_region' in reference:
        observed &= reference['point_region'][y, x] == regions
    if person is not None:
        observed &= ~person[y, x]
    occluded = observed & (z > reference["depth"][y, x] + tolerance)
    return observed & ~occluded, occluded


def choose_pairs(views, tolerance, min_overlap=.08, max_refs=2):
    result = []
    for query in views:
        mask = query["valid"] & ~query["person"]
        points = query["points"][mask]
        stride = max(1, int(np.ceil(len(points) / 768)))
        regions = query['point_region'][mask][::stride] if 'point_region' in query else None
        points = points[::stride]
        choices = []
        if len(points):
            for ref in views:
                if ref["epoch"] == query["epoch"]:
                    continue
                dot = np.clip(query["pose"][:3, 2] @ ref["pose"][:3, 2], -1, 1)
                if dot < -.2 and 'region_poses' not in ref:
                    continue
                supported, _ = counterpart_support(points, ref, tolerance, ref["person"], regions)
                overlap = float(supported.mean())
                if overlap >= min_overlap:
                    distance = float(np.linalg.norm(query["pose"][:3, 3] - ref["pose"][:3, 3]))
                    choices.append((overlap + .05 * dot - .001 * distance, ref, overlap, distance))
        choices.sort(key=lambda row: (-row[0], row[1]["id"]))
        refs, frames = [], set()
        for _, ref, overlap, distance in choices:
            # Multiple cube faces from one input frame are one observation.
            if ref["frame_id"] in frames:
                continue
            frames.add(ref["frame_id"])
            refs.append(dict(id=ref["id"], overlap=overlap, camera_distance_u=distance))
            if len(refs) == max_refs:
                break
        result.append(dict(query=query["id"], refs=refs,
                           state="ready" if refs else "unobserved_counterpart"))
    return result


def evidence_masks(query, reference, probability, query_person, reference_person, tolerance, threshold=.5):
    supported, occluded = counterpart_support(query["points"], reference, tolerance, reference_person, query.get('point_region'))
    raw = probability > threshold
    own = query["valid"] & query.get('segmentation_supported', True)
    people = raw & own & query_person
    missing = raw & ~own
    hidden = raw & own & ~query_person & occluded
    unobserved = raw & own & ~query_person & ~occluded & ~supported
    candidate = raw & own & ~query_person & supported
    assert int(raw.sum()) == sum(int(x.sum()) for x in [people, missing, hidden, unobserved, candidate])
    return dict(raw=raw, candidate=candidate, observable=own & ~query_person & supported,
                excluded_person=people, unsupported_geometry=missing, occluded=hidden,
                unobserved_counterpart=unobserved)


def vote_state(observed_frames, changed_frames, min_frames=2, ratio=.5):
    observed, changed = set(observed_frames), set(changed_frames)
    if not changed.issubset(observed):
        raise ValueError("Change votes must be a subset of observed frames")
    if len(observed) < min_frames:
        return "insufficient_observations"
    return "change_candidate" if len(changed) >= min_frames and len(changed) / len(observed) >= ratio else "no_supported_change"


def voxel_components(keys):
    if not len(keys):
        return []
    # Sparse hash-grid traversal avoids allocating a dense scene-sized volume.
    remaining = {tuple(key) for key in keys}
    groups = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            key = stack.pop()
            group.append(key)
            for axis in range(3):
                for sign in [-1, 1]:
                    neighbor = list(key)
                    neighbor[axis] += sign
                    neighbor = tuple(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        groups.append(group)
    return groups
