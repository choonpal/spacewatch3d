"""Bounded object-only precision, independent of spatial coverage samples.

Empirical two-draw lower estimate, NOT a statistical confidence interval or an
image-legibility guarantee. Decisions and points depend on pose/object/draw,
never on batch order, the final holdout, or the currently selected image set.
"""

from functools import lru_cache
import hashlib
import numpy as np

from .visibility import BVH, Surface, _stratified_triangle_indices, evaluate_positions


class ObjectQualityRefiner:
    BASE_SAMPLES = 64
    PRIORITY_SAMPLES = 256
    FINE_SAMPLES = 1024

    def __init__(self, scene, bvh, config, object_bvhs=None):
        self.bvh, self.config = bvh, config
        # Plan-local bounded caches; a global method cache would retain old
        # scenes/BVHs after subsequent captures are processed.
        self.sample = lru_cache(maxsize=64)(self.sample)
        self._score = lru_cache(maxsize=4096)(self._score)
        self.triangles = {}
        self.signatures = {}
        self.object_bvhs = {}
        pieces = {}
        for geometry in scene.geometries:
            owners = geometry.object_ids or ((geometry.object_id,) if geometry.object_id is not None else ())
            for oid in owners:
                if len(geometry.triangles):
                    pieces.setdefault(oid, []).append(geometry.triangles)
        for oid, items in pieces.items():
            triangles = np.concatenate(items)
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1) / 2
            triangles, area = triangles[area > 0], area[area > 0]
            self.triangles[oid] = triangles, area
            self.object_bvhs[oid] = ((object_bvhs or {}).get(oid)
                                     or BVH(triangles))
            self.signatures[oid] = hashlib.sha256(triangles.tobytes()).hexdigest()

    def sample(self, oid, count, stage):
        triangles, area = self.triangles[oid]
        total = float(area.sum())
        seed_data = repr((self.config.seed, self.signatures[oid], count, stage)).encode()
        seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        chosen = _stratified_triangle_indices(area, count, rng)
        tri = triangles[chosen]
        u, v = np.sqrt(rng.random(count)), rng.random(count)
        points = (1 - u[:, None]) * tri[:, 0] + (u * (1 - v))[:, None] * tri[:, 1] + (u * v)[:, None] * tri[:, 2]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        # Only object quality uses these weights. They never enter spatial
        # area, region allocation, or the shared coverage denominator.
        return Surface(points, normals, np.full(count, total / count), [oid] * count,
                       ["object_quality_only"] * count, self.bvh, {oid: total},
                       object_bvhs={oid: self.object_bvhs[oid]})

    def _quality(self, position, oid, count, stage):
        return float(evaluate_positions(np.asarray([position]), self.sample(oid, count, stage),
                                        [{"instance_id": oid}], self.config)[1][0, 0])

    def _near_boundary(self, value):
        low = self.config.object_quality_threshold - .1
        high = self.config.object_quality_threshold + self.config.object_selection_margin + .1
        return low - 1e-12 <= value <= high + 1e-12

    def _score(self, position, oid, priority):
        if oid not in self.triangles or not len(self.triangles[oid][0]):
            return 0., (0.,), 0
        coarse = self._quality(position, oid, self.BASE_SAMPLES, 0)
        if not priority and not self._near_boundary(coarse):
            return coarse, (coarse,), self.BASE_SAMPLES
        intermediate = self._quality(position, oid, self.PRIORITY_SAMPLES, 1)
        if self._near_boundary(coarse) or self._near_boundary(intermediate) or abs(coarse - intermediate) > .025:
            estimates = tuple(self._quality(position, oid, self.FINE_SAMPLES, stage) for stage in (2, 3))
            return min(estimates), estimates, self.FINE_SAMPLES
        return intermediate, (intermediate,), self.PRIORITY_SAMPLES

    def evaluate(self, positions, objects):
        result = np.zeros((len(positions), len(objects)))
        for i, position in enumerate(positions):
            pose = tuple(float(v) for v in position)
            for j, obj in enumerate(objects):
                priority = bool(obj.get("must_see", False) or obj.get("importance", 0) >= .75)
                result[i, j] = self._score(pose, obj["instance_id"], priority)[0]
        return result

    def report(self, positions, objects):
        records = []
        for obj in objects:
            priority = bool(obj.get("must_see", False) or obj.get("importance", 0) >= .75)
            views = []
            for position in positions:
                pose = tuple(float(v) for v in position)
                quality, estimates, count = self._score(pose, obj["instance_id"], priority)
                views.append({"position_m": list(pose), "quality": quality,
                              "samples_per_estimate": count, "estimate_count": len(estimates),
                              "estimates": list(estimates),
                              "nominal_decision_disagrees": min(estimates) < self.config.object_quality_threshold <= max(estimates)})
            records.append({"instance_id": obj["instance_id"], "views": views})
        return {"method": "object_only_adaptive_precision_lower_of_two_fine_draws",
                "observation_model": "self_visible_exterior" if self.config.shape_normalized_object_quality else "whole_mesh_surface",
                "seed": self.config.seed, "space_surface_unchanged": True,
                "statistical_confidence_guarantee": False, "objects": records}


def make_quality_refiner(scene, surface, objects, config):
    if not config.object_quality_refinement or not objects:
        return None
    return ObjectQualityRefiner(scene, surface.bvh, config, surface.object_bvhs)
