"""Predeclared optimization samples, never the final holdout samples."""

from copy import copy, deepcopy
from dataclasses import replace
import numpy as np

from .candidates import audit_visibility_candidates
from .routing import add_candidates
from .selection import Objective
from .visibility import evaluate_positions, sample_surface
from .object_quality import make_quality_refiner


class JointObjective:
    """Meet every draw's original goals; averaging must not hide a failed draw."""

    def __init__(self, objectives):
        self.objectives = objectives
        self.visible = objectives[0].visible
        if any(len(obj.visible) != len(self.visible) for obj in objectives):
            raise ValueError("Joint objectives must have identical candidate ordering")

    def measure(self, selected):
        measured = [objective.measure(selected) for objective in self.objectives]
        utility = float(np.mean([value for value, _ in measured]))
        reports = [report for _, report in measured]
        report = {**reports[0], "utility": utility, "optimization_reports": reports,
                  "unmet_targets": [f"sample_{i}:{target}" for i, item in enumerate(reports)
                                    for target in item["unmet_targets"]]}
        return utility, report

    def constraint_deficit(self, report):
        return sum(obj.constraint_deficit(item) for obj, item in
                   zip(self.objectives, report["optimization_reports"]))


def prepare_repair_samples(scene, route, primary_surface, primary_visible, primary_quality,
                           objects, config, banned=(), primary_refiner=None):
    """Append candidates discovered in fixed training draws, retaining all draws.

    Old row order stays fixed. Every new node is scored on every previous draw,
    so no draw drops its denominator or allows a later pruning to erase its goal.
    The route geometry, banned nodes and primary audit metadata stay unchanged.
    """
    route.candidate_sampling["coverage_repair_added_count"] = 0
    datasets = [{"seed": config.seed, "surface": primary_surface,
                 "visible": primary_visible, "quality": primary_quality, "quality_refiner": primary_refiner}]
    if not config.coverage_repair:
        return datasets, {"status": "disabled", "used_for_selection": False, "runs": []}
    audits = []
    for index in range(config.coverage_repair_runs):
        seed = config.seed + index + 1
        multiplier = config.coverage_repair_sample_multiplier
        draw_config = replace(config, seed=seed, surface_samples=config.surface_samples * multiplier,
                              min_object_samples=config.min_object_samples * multiplier,
                              min_region_samples=config.min_region_samples * multiplier)
        prepared = getattr(primary_surface, "prepared", None)
        surface = (sample_surface(scene, draw_config, prepared=prepared) if prepared is not None
                   else sample_surface(scene, draw_config))
        refiner = make_quality_refiner(scene, surface, objects, draw_config)

        def evaluate(nodes):
            visible, quality = evaluate_positions(route.positions[nodes], surface, objects, config)
            if refiner is not None:
                quality = refiner.evaluate(route.positions[nodes], objects)
            return visible, quality

        visible, quality = evaluate(route.candidates)
        # Reuse the same bounded visibility/joint-object/connectivity audit, but
        # do not overwrite the primary draw's counts and preservation evidence.
        audited_route = copy(route)
        audited_route.candidates = list(route.candidates)
        audited_route.candidate_reasons = deepcopy(route.candidate_reasons)
        audited_route.candidate_sampling = {}
        added, added_visible, added_quality = audit_visibility_candidates(
            audited_route, visible, quality, surface.weights, config,
            evaluate, banned)
        if added:
            actual = add_candidates(route, added, config.grid_m, "coverage_repair", config, force=True)
            assert actual == added
            for node in actual:
                route.candidate_reasons[node].extend(reason for reason in audited_route.candidate_reasons[node]
                                                     if reason not in route.candidate_reasons[node])
            for dataset in datasets:
                new_visible, new_quality = evaluate_positions(route.positions[actual], dataset["surface"], objects, config)
                if dataset["quality_refiner"] is not None:
                    new_quality = dataset["quality_refiner"].evaluate(route.positions[actual], objects)
                dataset["visible"] = np.concatenate([dataset["visible"], new_visible])
                dataset["quality"] = np.concatenate([dataset["quality"], new_quality])
            visible = np.concatenate([visible, added_visible])
            quality = np.concatenate([quality, added_quality])
        route.candidate_sampling["coverage_repair_added_count"] += len(added)
        datasets.append({"seed": seed, "surface": surface, "visible": visible, "quality": quality})
        datasets[-1]["quality_refiner"] = refiner
        audits.append({"seed": seed, "sample_count": draw_config.surface_samples,
                       "min_object_samples": draw_config.min_object_samples,
                       "min_region_samples": draw_config.min_region_samples,
                       "added_candidate_ids": [route.candidate_id(n) for n in added],
                       "candidate_audit": audited_route.candidate_sampling["visibility_audit"],
                       "surface_sampling": surface.sampling_report})
    return datasets, {"status": "prepared", "used_for_selection": True,
                      "method": "predeclared_joint_sample_goals", "runs": audits}


def make_objectives(datasets, allowed_indices, objects, config, declared_regions):
    return [Objective(data["visible"][allowed_indices], data["quality"][allowed_indices],
                      data["surface"].weights, data["surface"].regions, objects, config,
                      declared_regions=declared_regions) for data in datasets]


def finish_repair_report(report, objectives, selected):
    if report["status"] == "disabled":
        return
    for run, objective in zip(report["runs"], objectives[1:]):
        coverage = objective.measure(selected)[1]
        coverage["sample_role"] = "optimization_repair"
        run["coverage_report"] = coverage
    report["status"] = "targets_unmet" if any(run["coverage_report"]["unmet_targets"]
                                                for run in report["runs"]) else "targets_met"
