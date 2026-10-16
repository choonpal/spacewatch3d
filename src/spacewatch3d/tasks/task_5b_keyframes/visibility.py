"""Area-weighted surface sampling and two-sided segment/triangle occlusion.

The CPU BVH avoids optional rtree/Embree requirements. Large-scene throughput
has not been established; resource limits intentionally fail instead of truncating.
"""

from dataclasses import dataclass
import numpy as np
import shapely

from .models import Config, InputError, Scene


class BVH:
    def __init__(self, triangles):
        self.triangles = np.asarray(triangles)
        self.low = self.triangles.min(axis=1)
        self.high = self.triangles.max(axis=1)
        self.root = self._build(np.arange(len(triangles)))

    def _build(self, indices):
        low, high = self.low[indices].min(axis=0), self.high[indices].max(axis=0)
        if len(indices) <= 24:
            return low, high, indices, None, None
        axis = int(np.argmax(high - low))
        order = np.argsort((self.low[indices, axis] + self.high[indices, axis]), kind="stable")
        split = len(indices) // 2
        return low, high, None, self._build(indices[order[:split]]), self._build(indices[order[split:]])

    def blocked(self, origin, target):
        delta = target - origin
        distance = np.linalg.norm(delta)
        if distance <= 1e-8:
            return False
        direction = delta / distance
        # Scale-independent meter tolerance, excludes the target face, not nearby occluders.
        end = distance - 1e-6
        stack = [self.root]
        while stack:
            low, high, indices, left, right = stack.pop()
            near, far = 1e-7, end
            for axis in range(3):
                if abs(direction[axis]) < 1e-14:
                    if origin[axis] < low[axis] - 1e-10 or origin[axis] > high[axis] + 1e-10:
                        far = -1
                        break
                else:
                    a = (low[axis] - origin[axis]) / direction[axis]
                    b = (high[axis] - origin[axis]) / direction[axis]
                    near, far = max(near, min(a, b)), min(far, max(a, b))
            if near > far:
                continue
            if indices is None:
                stack.extend((left, right))
                continue
            tri = self.triangles[indices]
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            h = np.cross(direction, e2)
            determinant = np.einsum("ij,ij->i", e1, h)
            valid = np.abs(determinant) > 1e-12
            inverse = np.zeros(len(tri))
            inverse[valid] = 1 / determinant[valid]
            s = origin - tri[:, 0]
            u = inverse * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = inverse * (q @ direction)
            t = inverse * np.einsum("ij,ij->i", e2, q)
            if np.any(valid & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
                      & (t > 1e-7) & (t < end)):
                return True
        return False


@dataclass
class Surface:
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    object_ids: list
    regions: list[str]
    bvh: BVH
    object_areas: dict


def sample_surface(scene: Scene, config: Config) -> Surface:
    geometries = [g for g in scene.geometries if len(g.triangles)]
    areas = [np.linalg.norm(np.cross(g.triangles[:, 1] - g.triangles[:, 0],
                                     g.triangles[:, 2] - g.triangles[:, 0]), axis=1) / 2 for g in geometries]
    totals = np.asarray([a.sum() for a in areas])
    minimum = np.asarray([config.min_object_samples if g.object_id is not None else 1 for g in geometries])
    if minimum.sum() > config.surface_samples:
        raise InputError("surface_samples too small for min_object_samples and structure count")
    fractional = (config.surface_samples - minimum.sum()) * totals / totals.sum()
    counts = minimum + np.floor(fractional).astype(int)
    for index in np.argsort(-(fractional - np.floor(fractional)), kind="stable")[:config.surface_samples - counts.sum()]:
        counts[index] += 1
    rng = np.random.default_rng(config.seed)
    points, normals, weights, object_ids, object_areas = [], [], [], [], {}
    for g, area, total, count in zip(geometries, areas, totals, counts):
        chosen = rng.choice(len(area), size=int(count), p=area / total)
        tri = g.triangles[chosen]
        u, v = np.sqrt(rng.random(int(count))), rng.random(int(count))
        points.extend((1 - u[:, None]) * tri[:, 0] + (u * (1 - v))[:, None] * tri[:, 1] + (u * v)[:, None] * tri[:, 2])
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals.extend(normal / np.linalg.norm(normal, axis=1)[:, None])
        weights.extend([float(total / count)] * int(count))
        object_ids.extend([g.object_id] * int(count))
        if g.object_id is not None:
            object_areas[g.object_id] = object_areas.get(g.object_id, 0) + float(total)
    points = np.asarray(points)
    regions = []
    for p in points:
        assigned = next((rid for rid, poly in scene.regions if shapely.intersects_xy(poly, p[0], p[1])), None)
        if assigned is None:
            # Remaining surfaces always have a denominator, even outside named rooms.
            assigned = "tile_%d_%d" % tuple(np.floor(p[:2] / config.tile_m).astype(int))
        regions.append(assigned)
    return Surface(points, np.asarray(normals), np.asarray(weights), object_ids, regions,
                   BVH(np.concatenate([g.triangles for g in geometries])), object_areas)


def evaluate_positions(positions, surface: Surface, objects: list[dict], config: Config):
    visible = np.zeros((len(positions), len(surface.points)), dtype=bool)
    quality = np.zeros((len(positions), len(objects)))
    masks = [np.asarray([oid == obj["instance_id"] for oid in surface.object_ids]) for obj in objects]
    for index, position in enumerate(positions):
        delta = surface.points - position
        distances = np.linalg.norm(delta, axis=1)
        incidence = np.abs(np.einsum("ij,ij->i", surface.normals, delta / np.maximum(distances[:, None], 1e-12)))
        eligible = (distances > 1e-6) & (distances <= config.max_view_distance_m) & (incidence > 1e-5)
        for patch in np.flatnonzero(eligible):
            visible[index, patch] = not surface.bvh.blocked(position, surface.points[patch])
        for j, (obj, mask) in enumerate(zip(objects, masks)):
            seen = mask & visible[index]
            total = surface.object_areas.get(obj["instance_id"], 0)
            if total == 0:
                continue
            fraction = surface.weights[seen].sum() / total
            solid_angle = np.sum(surface.weights[seen] * incidence[seen] / np.maximum(distances[seen] ** 2, 1e-12))
            quality[index, j] = fraction * min(1, solid_angle / config.solid_angle_target_sr)
    return visible, quality
