"""Untouched final holdout samples, never reused to optimize this plan."""

from dataclasses import replace
from .models import InputError
from .selection import Objective
from .visibility import evaluate_positions, sample_surface


def validate_coverage(scene, route, selected_nodes, objects, config, banned=()):
    if not config.independent_coverage_validation:
        return {"status": "disabled", "used_for_selection": False, "runs": []}
    banned = set(banned)
    eligible = [n for n in range(len(route.positions)) if n not in banned]
    scored = set(route.candidates) - banned
    if len(set(eligible) - scored) > config.max_visibility_probes:
        raise InputError("max_visibility_probes exceeded by independent coverage validation; crop or explicitly raise limit")
    lookup = {node: i for i, node in enumerate(eligible)}
    if any(node not in lookup for node in selected_nodes):
        raise InputError("Independent validation received an excluded or unknown camera")
    selected = [lookup[node] for node in selected_nodes]
    sample_count = config.surface_samples * config.validation_sample_multiplier
    runs = []
    for index in range(config.validation_runs):
        # Graph, poses, importance and selected images stay fixed. Only the
        # random surface draw changes, and none of these scores affect selection.
        seed = config.seed + index + 1
        surface = sample_surface(scene, replace(config, seed=seed, surface_samples=sample_count))
        visible, quality = evaluate_positions(route.positions[eligible], surface, objects, config)
        objective = Objective(visible, quality, surface.weights, surface.regions, objects, config,
                              declared_regions=[rid for rid, _ in scene.regions])
        report = objective.measure(selected)[1]
        report["sample_role"] = "holdout"
        runs.append({"seed": seed, "sample_count": sample_count, "coverage_report": report,
                     "surface_sampling": surface.sampling_report,
                     "unobservable_object_ids": [obj["instance_id"] for j, obj in enumerate(objects)
                                                 if not objective.possible_objects[j]]})
    return {"status": "failed" if any(r["coverage_report"]["unmet_targets"] for r in runs) else "passed",
            "used_for_selection": False, "method": "fresh_surface_samples_on_fixed_eligible_route_graph",
            "selected_viewpoint_ids": [route.viewpoint_id(n) for n in selected_nodes],
            "eligible_route_node_count": len(eligible), "sample_count_per_run": sample_count,
            "worst_coverage_observable": min(r["coverage_report"]["coverage_observable"] for r in runs),
            "worst_balanced_space_coverage": min(r["coverage_report"]["balanced_space_coverage"] for r in runs),
            "runs": runs}
