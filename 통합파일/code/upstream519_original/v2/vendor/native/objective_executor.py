"""Same Objective.measure report, with validated static-region memoization."""
import numpy as np
from spacewatch3d.tasks.task_5b_keyframes.selection import Objective

ORIGINAL = Objective.measure


def static_data(self):
    # Fingerprints also preserve reference behavior if a caller mutates arrays.
    signature = (tuple(self.region_ids), frozenset(self.declared_regions),
                 tuple((a.shape, a.dtype.str, a.tobytes()) for a in
                       (self.weights, self.regions, self.observable, self.importance, self.possible_objects)))
    if getattr(self, "_execution_static_signature", None) != signature:
        regions = []
        for rid in self.region_ids:
            mask = self.regions == rid
            indices = np.flatnonzero(mask)
            regions.append({"rid": rid, "indices": indices, "weights": self.weights[indices],
                            "observable": self.observable[indices], "count": int(mask.sum()),
                            "declared": rid in self.declared_regions,
                            "possible": float(self.weights[mask & self.observable].sum()),
                            "total": float(self.weights[mask].sum())})
        self._execution_static = {"regions": regions, "importance_sum": float(self.importance.sum()),
                "feasible_weight": float(self.importance[self.possible_objects].sum()),
                "possible_area": float(self.weights[self.observable].sum()),
                "total_area": float(self.weights.sum())}
        self._execution_static_signature = signature
    return self._execution_static


def measure(self, selected):
    fixed = static_data(self)
    covered = np.any(self.visible[selected], axis=0) if selected else np.zeros(len(self.weights), dtype=bool)
    best = np.max(self.quality[selected], axis=0) if selected else np.zeros(len(self.objects))
    per_region = []
    for region in fixed["regions"]:
        mask = covered[region["indices"]]
        possible, total = region["possible"], region["total"]
        seen = float(region["weights"][mask].sum())
        feasible_seen = float(region["weights"][mask & region["observable"]].sum())
        per_region.append({"region_id": region["rid"], "input_area_m2": total,
                "sample_count": region["count"], "declared": region["declared"],
                "assessment_status": "no_surface" if not region["count"] else "sampled",
                "observable_area_m2": possible, "covered_area_m2": seen,
                "coverage_observable": feasible_seen / possible if possible else None,
                "coverage_total": seen / total if total else None})
    feasible_regions = [r["coverage_observable"] for r in per_region if r["coverage_observable"] is not None]
    c_space = float(np.mean(feasible_regions)) if feasible_regions else 0.0
    importance_sum = fixed["importance_sum"]
    c_object = float(best @ self.importance / importance_sum) if importance_sum else 0.0
    alpha, beta = self.config.space_weight, self.config.object_weight if importance_sum else 0.0
    if alpha + beta == 0:
        alpha = 1.0
    utility = (alpha * c_space + beta * c_object) / (alpha + beta) if alpha + beta else 0.0
    observed = best >= self.object_thresholds
    feasible_weight = fixed["feasible_weight"]
    observed_weight = float(self.importance[observed].sum())
    object_feasible = float(self.importance[observed & self.possible_objects].sum()) / feasible_weight if feasible_weight else None
    unmet = []
    if not self.observable.any():
        unmet.append("no_observable_surface")
    if c_space + 1e-10 < self.config.space_target:
        unmet.append("space_target")
    if any(value + 1e-10 < self.config.region_target for value in feasible_regions):
        unmet.append("region_target")
    unmet.extend("region_unassessed:" + r["region_id"] for r in per_region if not r["sample_count"])
    if object_feasible is not None and object_feasible + 1e-10 < self.config.object_target:
        unmet.append("object_target")
    for j, obj in enumerate(self.objects):
        if (self.config.require_all_objects or obj["must_see"]) and not observed[j]:
            prefix = "must_see:" if obj["must_see"] else "object_unobserved:"
            unmet.append(prefix + obj["instance_id"])
    possible_area = fixed["possible_area"]
    seen_area = float(self.weights[covered].sum())
    report = {"utility": utility, "balanced_space_coverage": c_space,
            "weighted_object_quality": c_object, "input_area_m2": fixed["total_area"],
            "observable_area_m2": possible_area, "covered_area_m2": seen_area,
            "coverage_observable": float(self.weights[covered & self.observable].sum()) / possible_area if possible_area else 0,
            "coverage_total": seen_area / fixed["total_area"],
            "object_importance_coverage_observable": object_feasible,
            "object_importance_coverage_total": observed_weight / importance_sum if importance_sum else None,
            "object_observation_requirements": {
                "policy": "all_input_objects" if self.config.require_all_objects else "explicit_must_see_only",
                "required_object_count": sum(self.config.require_all_objects or obj["must_see"] for obj in self.objects),
                "unobserved_required_object_ids": [obj["instance_id"] for j, obj in enumerate(self.objects)
                    if (self.config.require_all_objects or obj["must_see"]) and not observed[j]],
                "observed_object_count": int(observed.sum()), "input_object_count": len(self.objects)},
            "regions": per_region, "unmet_targets": unmet,
            "object_best_quality": {obj["instance_id"]: float(best[j]) for j, obj in enumerate(self.objects)}}
    return utility, report


def install():
    Objective.measure = measure


def restore():
    Objective.measure = ORIGINAL
