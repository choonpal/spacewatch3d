"""A bounded selection heuristic, not a statistical confidence guarantee."""

from copy import copy
import numpy as np


def guard_thresholds(best_quality, config):
    """Never relax nominal quality; cap extra headroom at known eligible maxima."""
    threshold = config.object_quality_threshold
    return np.maximum(threshold, np.minimum(threshold + config.object_selection_margin, best_quality))


def guard_objectives(objectives):
    guarded = []
    for objective in objectives:
        item = copy(objective)
        best = np.max(item.quality, axis=0) if len(item.quality) else np.zeros(len(item.objects))
        item.object_thresholds = guard_thresholds(best, item.config)
        guarded.append(item)
    return guarded


def finish_guard_report(objectives, selected, seeds):
    config = objectives[0].config
    if not config.object_selection_margin:
        return {"status": "disabled", "used_for_selection": False, "runs": []}
    runs = []
    for objective, seed in zip(objectives, seeds):
        measured = objective.measure(selected)[1]
        best = np.max(objective.quality, axis=0) if len(objective.quality) else np.zeros(len(objective.objects))
        objects = []
        for j, obj in enumerate(objective.objects):
            threshold = float(objective.object_thresholds[j])
            quality = measured["object_best_quality"][obj["instance_id"]]
            objects.append({"instance_id": obj["instance_id"], "best_eligible_quality": float(best[j]),
                            "selection_threshold": threshold, "selected_best_quality": quality,
                            "nominally_observable": bool(objective.possible_objects[j]),
                            "capped_at_best_known": bool(best[j] >= config.object_quality_threshold
                                                        and best[j] < config.object_quality_threshold + config.object_selection_margin),
                            "guard_met": quality >= threshold})
        runs.append({"seed": seed, "sample_role": "selection_guard", "objects": objects,
                     "object_importance_coverage_observable": measured["object_importance_coverage_observable"],
                     "unmet_targets": [t for t in measured["unmet_targets"]
                                       if t == "object_target" or t.startswith(("must_see:", "object_unobserved:"))]})
    return {"status": "targets_unmet" if any(r["unmet_targets"] for r in runs) else "targets_met",
            "used_for_selection": True, "method": "capped_quality_headroom",
            "margin": config.object_selection_margin, "nominal_threshold": config.object_quality_threshold,
            "nominal_observable_denominator_preserved": True, "runs": runs}
