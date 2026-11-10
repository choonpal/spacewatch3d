"""Whole-input diagnostics, independent of feasible-plan stopping criteria.

No object or surface is removed from these denominators. These are geometric
coverage targets, not image-quality validation or a declaration of patrol safety.
"""


def summarize_input_coverage(report, objects, config):
    draws = [("selection", config["seed"], report)]
    for key, role in (("coverage_repair", "optimization_repair"),
                      ("independent_validation", "holdout")):
        draws.extend((role, run["seed"], run["coverage_report"])
                     for run in report.get(key, {}).get("runs", []))
    runs, missing = [], set()
    importance_sum = sum(obj["importance"] for obj in objects)
    for role, seed, measured in draws:
        regions = measured["regions"]
        values = [region["coverage_total"] for region in regions
                  if region["coverage_total"] is not None]
        balanced = sum(values) / len(values) if values else 0.
        observed = {obj["instance_id"] for obj in objects
                    if measured["object_best_quality"][obj["instance_id"]] >= config["object_quality_threshold"]}
        unobserved = sorted(obj["instance_id"] for obj in objects if obj["instance_id"] not in observed)
        missing.update(unobserved)
        object_coverage = (sum(obj["importance"] for obj in objects if obj["instance_id"] in observed)
                           / importance_sum if importance_sum else None)
        unmet = []
        if balanced + 1e-10 < config["space_target"]:
            unmet.append("space_target")
        if any(value + 1e-10 < config["region_target"] for value in values):
            unmet.append("region_target")
        unmet.extend("region_unassessed:" + region["region_id"] for region in regions
                     if region["coverage_total"] is None)
        if object_coverage is not None and object_coverage + 1e-10 < config["object_target"]:
            unmet.append("object_target")
        unmet.extend(("must_see:" if obj["must_see"] else "object_unobserved:") + obj["instance_id"]
                     for obj in objects if (config.get("require_all_objects", False) or obj["must_see"])
                     and obj["instance_id"] not in observed)
        runs.append({"sample_role": role, "seed": seed,
                     "balanced_space_coverage_total": balanced,
                     "surface_coverage_total": measured["coverage_total"],
                     "object_importance_coverage_total": object_coverage,
                     "unobserved_object_ids": unobserved, "unmet_targets": unmet})
    return {"status": "targets_unmet" if any(run["unmet_targets"] for run in runs) else "targets_met",
            "scope": "whole_input_geometry_not_image_quality",
            "does_not_change_selection_or_denominators": True,
            "independent_validation_performed": report.get("independent_validation", {}).get("status") != "disabled"
                                                and bool(report.get("independent_validation", {}).get("runs")),
            "unobserved_object_ids": sorted(missing), "runs": runs}
