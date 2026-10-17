"""Area-weighted surface sampling and two-sided segment/triangle occlusion.

The CPU BVH avoids optional rtree/Embree requirements. Large-scene throughput
has not been established; resource limits intentionally fail instead of truncating.
"""

from dataclasses import dataclass
from collections import deque
import hashlib
import numpy as np
import shapely

from .models import Config, InputError, Scene


# Dimensionless triangle-coordinate grid; a power of two keeps unit vertices
# exact. GEOS overlays share this precision instead of manufacturing 1e-17 m2
# residual regions. This is not an absolute physical-area cutoff.
PARTITION_PRECISION = 64 * np.finfo(float).eps
# Bound temporary ray/triangle arrays independently of scene and sample size.
# This is a computation chunk, never a sampling or visibility truncation limit.
RAY_BATCH_SIZE = 128


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

    def blocked_many(self, origin, targets):
        """Query all targets with bounded CPU batches and scalar tolerances.

        Each live ray traverses the same tree; only array arithmetic is shared.
        No visibility result is cached across positions or surface draws.
        """
        origin, targets = np.asarray(origin, dtype=float), np.asarray(targets, dtype=float)
        if (origin.shape != (3,) or targets.ndim != 2 or targets.shape[1] != 3
                or not np.isfinite(origin).all() or not np.isfinite(targets).all()):
            raise InputError("Occlusion queries require a finite origin 3-vector and target Nx3 array")
        result = np.zeros(len(targets), dtype=bool)
        for start in range(0, len(targets), RAY_BATCH_SIZE):
            result[start:start + RAY_BATCH_SIZE] = self._blocked_batch(origin, targets[start:start + RAY_BATCH_SIZE])
        return result

    def _blocked_batch(self, origin, targets):
        delta = targets - origin
        distances = np.linalg.norm(delta, axis=1)
        directions = np.divide(delta, distances[:, None], out=np.zeros_like(delta), where=distances[:, None] > 1e-8)
        end = distances - 1e-6
        blocked = np.zeros(len(targets), dtype=bool)
        stack = [(self.root, np.flatnonzero(distances > 1e-8))]
        while stack:
            (low, high, indices, left, right), rays = stack.pop()
            rays = rays[~blocked[rays]]
            if not len(rays):
                continue
            near, far = np.full(len(rays), 1e-7), end[rays].copy()
            for axis in range(3):
                direction = directions[rays, axis]
                parallel = np.abs(direction) < 1e-14
                if origin[axis] < low[axis] - 1e-10 or origin[axis] > high[axis] + 1e-10:
                    far[parallel] = -1
                moving = ~parallel
                a = (low[axis] - origin[axis]) / direction[moving]
                b = (high[axis] - origin[axis]) / direction[moving]
                near[moving] = np.maximum(near[moving], np.minimum(a, b))
                far[moving] = np.minimum(far[moving], np.maximum(a, b))
            rays = rays[near <= far]
            if not len(rays):
                continue
            if indices is None:
                stack.extend(((left, rays), (right, rays)))
                continue
            # At most RAY_BATCH_SIZE rays by 24 leaf triangles. Avoid a
            # full samples-by-scene-triangles tensor and retain two-sided hits.
            tri = self.triangles[indices]
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            h = np.cross(directions[rays, None, :], e2[None, :, :])
            determinant = np.einsum("tj,rtj->rt", e1, h)
            valid = np.abs(determinant) > 1e-12
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            s = origin - tri[:, 0]
            u = inverse * np.einsum("tj,rtj->rt", s, h)
            q = np.cross(s, e1)
            v = inverse * np.einsum("tj,rj->rt", q, directions[rays])
            t = inverse * np.einsum("ij,ij->i", e2, q)[None, :]
            blocked[rays] = np.any(valid & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
                                   & (t > 1e-7) & (t < end[rays, None]), axis=1)
        return blocked


@dataclass
class Surface:
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    object_ids: list
    regions: list[str]
    bvh: BVH
    object_areas: dict
    sampling_report: dict | None = None
    prepared: "PreparedSurface | None" = None


@dataclass(frozen=True)
class _SamplingGeometry:
    object_id: str | None


@dataclass(frozen=True)
class PreparedSurface:
    """Plan-local read-only geometry; no random samples or visibility scores."""

    scene: Scene
    signature: bytes
    geometries: tuple
    regions: tuple
    strata: tuple
    totals: np.ndarray
    bvh: BVH
    input_triangle_count: int
    partition_triangle_count: int

    def validate(self, scene, config):
        if scene is not self.scene:
            raise InputError("prepared surface belongs to a different scene")
        if _geometry_signature(scene) != self.signature:
            raise InputError("prepared surface input geometry, objects or regions changed; prepare again")
        if self.input_triangle_count > config.max_triangles:
            raise InputError("max_triangles exceeded by prepared input geometry")
        if self.partition_triangle_count > config.max_triangles:
            raise InputError("max_triangles exceeded by prepared regional surface partition")


def _geometry_signature(scene):
    digest = hashlib.sha256()
    for geometry in scene.geometries:
        triangles = np.ascontiguousarray(geometry.triangles)
        digest.update(repr((geometry.geometry_id, geometry.object_id, triangles.shape, triangles.dtype.str)).encode())
        digest.update(triangles.view(np.uint8))
    for rid, region in scene.regions:
        digest.update(repr(rid).encode())
        digest.update(region.wkb)
    return digest.digest()


def prepare_surface(scene, config):
    """Prepare fixed geometry once without advancing any random generator."""
    geometries = [g for g in scene.geometries if len(g.triangles)]
    triangle_count = sum(len(g.triangles) for g in geometries)
    if triangle_count > config.max_triangles:
        raise InputError("max_triangles exceeded by input surface geometry")
    strata = tuple(_surface_strata(geometries, scene.regions, config))
    totals = np.asarray([s[4] for s in strata])
    # Keep invalid-budget failures ahead of BVH construction, as before.
    # Every actual draw still recomputes its own minimum allocation.
    _minimum_counts(strata, geometries, scene.regions, totals, config)
    bvh = BVH(np.concatenate([g.triangles for g in geometries]))
    for array in (totals, bvh.triangles, bvh.low, bvh.high):
        array.setflags(write=False)
    for _, _, triangles, areas, _ in strata:
        triangles.setflags(write=False)
        areas.setflags(write=False)
    stack = [bvh.root]
    while stack:
        low, high, indices, left, right = stack.pop()
        low.setflags(write=False)
        high.setflags(write=False)
        if indices is None:
            stack.extend((left, right))
        else:
            indices.setflags(write=False)
    return PreparedSurface(scene, _geometry_signature(scene), tuple(_SamplingGeometry(g.object_id) for g in geometries),
                           tuple(scene.regions), strata, totals, bvh, triangle_count,
                           sum(len(s[2]) for s in strata))


def _apportion(count, areas):
    """Deterministic largest-remainder allocation, including count=0."""
    fractional = count * np.asarray(areas) / np.sum(areas)
    result = np.floor(fractional).astype(int)
    for index in np.argsort(-(fractional - result), kind="stable")[:int(count - result.sum())]:
        result[index] += 1
    return result


def _clip_parameter_triangle(xy, region_triangle):
    """Pull back three XY half planes to barycentric (u,v) space.

    Unlike inverse XY projection this also handles vertical mesh faces. A face
    on a shared region/triangulation boundary may appear in several pieces;
    the caller unions them and assigns shared boundaries to the first region.
    """
    polygon = np.array([[0., 0.], [1., 0.], [0., 1.]])
    orientation = np.linalg.det(np.array([region_triangle[1] - region_triangle[0],
                                          region_triangle[2] - region_triangle[0]]))
    for a, b in zip(region_triangle, np.roll(region_triangle, -1, axis=0)):
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) * (1 if orientation > 0 else -1)
        normal /= np.linalg.norm(normal)
        values = (xy - a) @ normal
        values[np.abs(values) < 1e-12] = 0
        coefficient = values[1:] - values[0]
        distances = polygon @ coefficient + values[0]
        clipped = []
        for i in range(len(polygon)):
            p, q = polygon[i - 1], polygon[i]
            dp, dq = distances[i - 1], distances[i]
            if (dp >= 0) != (dq >= 0):
                clipped.append(p + (q - p) * dp / (dp - dq))
            if dq >= 0:
                clipped.append(q)
        if len(clipped) < 3:
            return None
        polygon = np.asarray(clipped)
    result = shapely.Polygon(polygon)
    return result if result.area > 0 else None


def _surface_strata(geometries, regions, config):
    """Partition physical surface area by geometry and named XY region.

    All operations happen in a nondegenerate triangle's barycentric plane, so
    vertical walls, sloped faces, concave regions and holes retain their area.
    Unnamed residual surfaces keep the existing sampled-tile interpretation.
    """
    region_meshes = []
    for rid, polygon in regions:
        triangles = np.asarray([np.asarray(p.exterior.coords)[:3, :2]
                                for p in shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))])
        region_meshes.append((rid, triangles, triangles.min(axis=1), triangles.max(axis=1)))
    region_index = shapely.STRtree([polygon for _, polygon in regions]) if regions else None
    strata, split_count = [], 0
    unit = shapely.set_precision(shapely.Polygon([[0, 0], [1, 0], [0, 1]]), PARTITION_PRECISION)
    for gi, geometry in enumerate(geometries):
        buckets = {}
        if not regions:
            buckets[None] = list(geometry.triangles)
        else:
            for triangle in geometry.triangles:
                xy = triangle[:, :2]
                low_xy, high_xy = xy.min(axis=0), xy.max(axis=0)
                # Query bounding boxes only (no geometric predicate). Vertical
                # faces, shared boundaries and concave/holey false positives
                # still enter the original exact clipping loop. STRtree order
                # is unspecified, so restore input order for boundary ownership.
                nearby = np.sort(region_index.query(shapely.box(*low_xy, *high_xy)))
                remaining = unit
                for region_number in nearby:
                    rid, region_triangles, low, high = region_meshes[region_number]
                    hits = np.flatnonzero(np.all(high >= low_xy, axis=1)
                                          & np.all(low <= high_xy, axis=1))
                    pieces = [_clip_parameter_triangle(xy, region_triangles[i]) for i in hits]
                    pieces = [shapely.set_precision(p, PARTITION_PRECISION) for p in pieces if p is not None]
                    if not pieces:
                        continue
                    assigned = remaining.intersection(shapely.union_all(pieces, grid_size=PARTITION_PRECISION))
                    if assigned.area:
                        buckets.setdefault(rid, []).extend(_lift_triangles(assigned, triangle))
                        remaining = remaining.difference(assigned)
                    if remaining.is_empty:
                        break
                if remaining.area:
                    buckets.setdefault(None, []).extend(_lift_triangles(remaining, triangle))
        for rid, pieces in buckets.items():
            triangles = np.asarray(pieces)
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                            triangles[:, 2] - triangles[:, 0]), axis=1) / 2
            positive = areas > 0
            if not positive.any():
                continue
            triangles, areas = triangles[positive], areas[positive]
            split_count += len(triangles)
            if split_count > config.max_triangles:
                raise InputError("max_triangles exceeded by regional surface partition; crop or explicitly raise limit")
            strata.append((gi, rid, triangles, areas, float(areas.sum())))
        original = np.linalg.norm(np.cross(geometry.triangles[:, 1] - geometry.triangles[:, 0],
                                           geometry.triangles[:, 2] - geometry.triangles[:, 0]), axis=1).sum() / 2
        partitioned = sum(s[4] for s in strata if s[0] == gi)
        if not np.isclose(original, partitioned, rtol=1e-9, atol=1e-10):
            raise InputError("Regional surface partition failed area conservation")
    return strata


def _lift_triangles(domain, triangle):
    for part in shapely.get_parts(domain):
        if part.geom_type != "Polygon" or not part.area:
            continue
        for face in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
            uv = np.asarray(face.exterior.coords)[:3, :2]
            yield triangle[0] + uv[:, :1] * (triangle[1] - triangle[0]) + uv[:, 1:] * (triangle[2] - triangle[0])


def _minimum_counts(strata, geometries, regions, totals, config):
    """Share sample minima between objects and regions via integral max flow.

    Each positive-area stratum needs one sample. Additional samples can satisfy
    a geometry and region deficit simultaneously; independent allocation can
    incorrectly reject a sufficient sample budget. A tiny bipartite residual
    graph maximizes this overlap, then area-apportions the unmatched deficits.
    """
    counts = np.ones(len(strata), dtype=int)
    rows = {gi: [i for i, s in enumerate(strata) if s[0] == gi] for gi in range(len(geometries))}
    columns = {rid: [i for i, s in enumerate(strata) if s[1] == rid] for rid, _ in regions}
    row_deficit = {gi: max(0, (config.min_object_samples if g.object_id is not None else 1) - len(rows[gi]))
                   for gi, g in enumerate(geometries)}
    col_deficit = {rid: max(0, config.min_region_samples - len(indices)) if indices else 0
                   for rid, indices in columns.items()}
    error = "surface_samples too small for min_object_samples, min_region_samples and surface strata"
    if len(strata) + max(sum(row_deficit.values()), sum(col_deficit.values())) > config.surface_samples:
        raise InputError(error)
    source, sink = ("source",), ("sink",)
    residual = {source: {}, sink: {}}

    def edge(a, b, capacity):
        residual.setdefault(a, {})[b] = int(capacity)
        residual.setdefault(b, {})[a] = 0

    for gi, deficit in row_deficit.items():
        if deficit:
            edge(source, ("geometry", gi), deficit)
    for rid, deficit in col_deficit.items():
        if deficit:
            edge(("region", rid), sink, deficit)
    capacities = {}
    for i, (gi, rid, *_) in enumerate(strata):
        capacity = min(row_deficit[gi], col_deficit.get(rid, 0))
        if capacity:
            a, b = ("geometry", gi), ("region", rid)
            edge(a, b, capacity)
            capacities[i] = (a, b, capacity)
    while True:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            a = queue.popleft()
            for b, capacity in residual[a].items():
                if capacity > 0 and b not in previous:
                    previous[b] = a
                    queue.append(b)
        if sink not in previous:
            break
        path, node = [], sink
        while node != source:
            path.append((previous[node], node))
            node = previous[node]
        amount = min(residual[a][b] for a, b in path)
        for a, b in path:
            residual[a][b] -= amount
            residual[b][a] += amount
    for i, (a, b, capacity) in capacities.items():
        counts[i] += capacity - residual[a][b]
    for gi, indices in rows.items():
        minimum = config.min_object_samples if geometries[gi].object_id is not None else 1
        counts[indices] += _apportion(max(0, minimum - counts[indices].sum()), totals[indices])
    for rid, indices in columns.items():
        if indices:
            counts[indices] += _apportion(max(0, config.min_region_samples - counts[indices].sum()), totals[indices])
    if counts.sum() > config.surface_samples:
        raise InputError(error)
    return counts


def _stratified_triangle_indices(areas, count, rng):
    """One independent draw per equal-area CDF bin, without extra samples.

    This stratifies triangle choice, not points within a triangle or occlusion
    boundaries. Each draw retains the stratum's original total/count weight;
    triangle count is never used as a substitute for physical area.
    """
    cdf = np.cumsum(areas, dtype=float)
    cdf /= cdf[-1]
    bins = np.arange(count, dtype=float)
    probability = (bins + rng.random(count)) / count
    # Addition can round a jitter just below 1 onto the next bin (including
    # probability=1 in the last bin). Keep every draw inside its own bin.
    probability = np.minimum(probability, np.nextafter((bins + 1) / count, 0.))
    return np.searchsorted(cdf, probability, side="right")


def sample_surface(scene: Scene, config: Config, prepared: PreparedSurface | None = None) -> Surface:
    if prepared is None:
        prepared = prepare_surface(scene, config)
    else:
        if not isinstance(prepared, PreparedSurface):
            raise InputError("prepared surface must be created by prepare_surface")
        prepared.validate(scene, config)
    geometries, strata, totals = prepared.geometries, prepared.strata, prepared.totals
    # At least one sample per positive-area geometry/region stratum, plus both
    # object and named-region minima. Extra samples never duplicate area weight.
    counts = _minimum_counts(strata, geometries, scene.regions, totals, config)
    counts += _apportion(config.surface_samples - counts.sum(), totals)
    rng = np.random.default_rng(config.seed)
    points, normals, weights, object_ids, object_areas = [], [], [], [], {}
    regions = []
    for (gi, rid, triangles, area, total), count in zip(strata, counts):
        g = geometries[gi]
        chosen = (_stratified_triangle_indices(area, int(count), rng) if g.object_id is not None
                  else rng.choice(len(area), size=int(count), p=area / total))
        tri = triangles[chosen]
        u, v = np.sqrt(rng.random(int(count))), rng.random(int(count))
        points.extend((1 - u[:, None]) * tri[:, 0] + (u * (1 - v))[:, None] * tri[:, 1] + (u * v)[:, None] * tri[:, 2])
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals.extend(normal / np.linalg.norm(normal, axis=1)[:, None])
        weights.extend([float(total / count)] * int(count))
        object_ids.extend([g.object_id] * int(count))
        regions.extend([rid] * int(count))
        if g.object_id is not None:
            object_areas[g.object_id] = object_areas.get(g.object_id, 0) + float(total)
    points = np.asarray(points)
    declared_ids = {rid for rid, _ in scene.regions}
    for i, p in enumerate(points):
        if regions[i] is None:
            # Remaining surfaces always have a denominator, even outside named rooms.
            tile = "tile_%d_%d" % tuple(np.floor(p[:2] / config.tile_m).astype(int))
            while tile in declared_ids:
                tile = "unnamed_" + tile
            regions[i] = tile
    audit = {"method": "geometry_named_region_area_stratified", "sample_count": len(points),
             "object_triangle_sampling": "equal_area_cdf_stratified",
             "structure_triangle_sampling": "iid_area_weighted",
             "partition_precision_barycentric": float(PARTITION_PRECISION),
             "stratum_count": len(strata), "min_region_samples": config.min_region_samples,
             "input_area_m2": float(totals.sum()), "declared_regions": [
                 {"region_id": rid, "sample_count": sum(int(n) for s, n in zip(strata, counts) if s[1] == rid),
                  "input_area_m2": float(sum(s[4] for s in strata if s[1] == rid)),
                  "status": "sampled" if any(s[1] == rid for s in strata) else "no_surface"}
                 for rid, _ in scene.regions]}
    return Surface(points, np.asarray(normals), np.asarray(weights), object_ids, regions,
                   prepared.bvh, object_areas, audit, prepared)


def evaluate_positions(positions, surface: Surface, objects: list[dict], config: Config):
    visible = np.zeros((len(positions), len(surface.points)), dtype=bool)
    quality = np.zeros((len(positions), len(objects)))
    masks = [np.asarray([oid == obj["instance_id"] for oid in surface.object_ids]) for obj in objects]
    for index, position in enumerate(positions):
        delta = surface.points - position
        distances = np.linalg.norm(delta, axis=1)
        incidence = np.abs(np.einsum("ij,ij->i", surface.normals, delta / np.maximum(distances[:, None], 1e-12)))
        eligible = (distances > 1e-6) & (distances <= config.max_view_distance_m) & (incidence > 1e-5)
        patches = np.flatnonzero(eligible)
        visible[index, patches] = ~surface.bvh.blocked_many(position, surface.points[patches])
        for j, (obj, mask) in enumerate(zip(objects, masks)):
            seen = mask & visible[index]
            total = surface.object_areas.get(obj["instance_id"], 0)
            if total == 0:
                continue
            fraction = surface.weights[seen].sum() / total
            solid_angle = np.sum(surface.weights[seen] * incidence[seen] / np.maximum(distances[seen] ** 2, 1e-12))
            quality[index, j] = fraction * min(1, solid_angle / config.solid_angle_target_sr)
    return visible, quality
