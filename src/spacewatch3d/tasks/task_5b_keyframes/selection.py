"""Marginal coverage/object utility, optional routed image cost, pruning and one swap."""

import numpy as np

from .models import Config


class Objective:
    def __init__(self, visible, quality, weights, regions, objects, config: Config, declared_regions=()):
        self.visible, self.quality = visible, quality
        self.weights, self.regions, self.objects, self.config = weights, np.asarray(regions), objects, config
        self.observable = np.any(visible, axis=0) if len(visible) else np.zeros(len(weights), dtype=bool)
        self.importance = np.asarray([obj["importance"] for obj in objects])
        self.possible_objects = np.any(quality >= config.object_quality_threshold, axis=0) if len(quality) else np.zeros(len(objects), bool)
        # Selection guards may replace these thresholds on a shallow copy, but
        # must never recompute the nominal observable-object denominator.
        self.object_thresholds = np.full(len(objects), config.object_quality_threshold)
        self.declared_regions = set(declared_regions)
        self.region_ids = sorted(set(regions) | self.declared_regions)

    def measure(self, selected):
        covered = np.any(self.visible[selected], axis=0) if selected else np.zeros(len(self.weights), dtype=bool)
        best = np.max(self.quality[selected], axis=0) if selected else np.zeros(len(self.objects))
        per_region = []
        for rid in self.region_ids:
            mask = self.regions == rid
            possible = float(self.weights[mask & self.observable].sum())
            seen = float(self.weights[mask & covered].sum())
            feasible_seen = float(self.weights[mask & covered & self.observable].sum())
            total = float(self.weights[mask].sum())
            per_region.append({"region_id": rid, "input_area_m2": total,
                               "sample_count": int(mask.sum()), "declared": rid in self.declared_regions,
                               "assessment_status": "no_surface" if not mask.any() else "sampled",
                               "observable_area_m2": possible, "covered_area_m2": seen,
                               "coverage_observable": feasible_seen / possible if possible else None,
                               "coverage_total": seen / total if total else None})
        feasible_regions = [r["coverage_observable"] for r in per_region if r["coverage_observable"] is not None]
        c_space = float(np.mean(feasible_regions)) if feasible_regions else 0.0
        importance_sum = float(self.importance.sum())
        c_object = float(best @ self.importance / importance_sum) if importance_sum else 0.0
        alpha, beta = self.config.space_weight, self.config.object_weight if importance_sum else 0.0
        if alpha + beta == 0:
            alpha = 1.0  # No semantic weights -> space-only, even for object-only config.
        utility = (alpha * c_space + beta * c_object) / (alpha + beta) if alpha + beta else 0.0
        observed = best >= self.object_thresholds
        feasible_weight = float(self.importance[self.possible_objects].sum())
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
            if obj["must_see"] and not observed[j]:
                unmet.append("must_see:" + obj["instance_id"])
        possible_area = float(self.weights[self.observable].sum())
        seen_area = float(self.weights[covered].sum())
        report = {"utility": utility, "balanced_space_coverage": c_space,
                  "weighted_object_quality": c_object, "input_area_m2": float(self.weights.sum()),
                  "observable_area_m2": possible_area, "covered_area_m2": seen_area,
                  "coverage_observable": float(self.weights[covered & self.observable].sum()) / possible_area if possible_area else 0,
                  "coverage_total": seen_area / float(self.weights.sum()),
                  "object_importance_coverage_observable": object_feasible,
                  "object_importance_coverage_total": observed_weight / importance_sum if importance_sum else None,
                  "regions": per_region, "unmet_targets": unmet,
                  "object_best_quality": {obj["instance_id"]: float(best[j]) for j, obj in enumerate(self.objects)}}
        return utility, report

    def constraint_deficit(self, report):
        """Unweighted shortfall of active goals, separate from ranking utility."""
        deficit = float("no_observable_surface" in report["unmet_targets"])
        deficit += sum(target.startswith("region_unassessed:") for target in report["unmet_targets"])
        deficit += max(0., self.config.space_target - report["balanced_space_coverage"])
        deficit += sum(max(0., self.config.region_target - region["coverage_observable"])
                       for region in report["regions"] if region["coverage_observable"] is not None)
        object_coverage = report["object_importance_coverage_observable"]
        if object_coverage is not None:
            deficit += max(0., self.config.object_target - object_coverage)
        deficit += sum(max(0., threshold - report["object_best_quality"][obj["instance_id"]]) / threshold
                       for obj, threshold in zip(self.objects, self.object_thresholds) if obj["must_see"])
        return deficit


class AttainableGoals:
    """Full-pool ceiling for stopping, never a relaxed success definition.

Failure labels and original denominators stay in the returned measurements.
All remaining achievable goals and the best attainable deficit must be met.
This is feasibility under current estimates, NOT proof about physical geometry.
"""

    def __init__(self, objective):
        self.objective = objective
        ceiling = objective.measure(list(range(len(objective.visible))))[1]
        self.unreachable = set(ceiling["unmet_targets"])
        self.minimum_deficit = float(objective.constraint_deficit(ceiling))

    def met(self, report):
        return bool(set(report["unmet_targets"]) <= self.unreachable
                    and self.objective.constraint_deficit(report) <= self.minimum_deficit + 1e-10)

    def report(self, measured):
        return {"method": "full_pool_attainable_goal_stopping_without_success_relaxation",
                "unreachable_targets_under_current_estimates": sorted(self.unreachable),
                "minimum_full_pool_deficit": self.minimum_deficit,
                "all_attainable_goals_met": self.met(measured),
                "original_targets_and_denominators_preserved": True}


def _prune_redundant(selected, measure, required=(), goal_met=None):
    """Recheck routed goals after each deletion until no core view is removable.

    Routed utility is not monotone: deleting a core view can change the order and
    the transition images. A previously rejected deletion may then become valid.
    Every accepted step removes a core view, so this loop is finite.
    """
    selected = list(selected)
    goal_met = goal_met or (lambda report: not report["unmet_targets"])
    while True:
        changed = False
        for chosen in list(reversed(selected)):
            if chosen in required:
                continue
            trial = [i for i in selected if i != chosen]
            current_report = measure(selected)[1]
            trial_report = measure(trial)[1]
            if (goal_met(trial_report) and not trial_report.get("selection_connection_failures")
                    and trial_report.get("selection_image_count", len(trial)) <= current_report.get("selection_image_count", len(selected))):
                selected = trial
                changed = True
        if not changed:
            return selected


def select_keyframes(objective: Objective, evaluate=None, required=()):
    """Use pure gain, or exact routed-set gain per additional unique image.

    The injected evaluator may include transition viewpoints in utility/goals.
    A bounded one-swap search improves cost without relaxing any target.
    """
    measure = evaluate or objective.measure
    attainable = AttainableGoals(objective)
    selected = sorted(set(required))
    history = {i: {"selection_step": None, "marginal_gain_at_selection": None,
                   "selection_method": "reference_viewpoint_reuse",
                   "marginal_image_count_at_selection": None} for i in selected}
    candidates = [i for i in range(len(objective.visible)) if i not in selected]
    while candidates:
        utility, report = measure(selected)
        if attainable.met(report):
            break
        deficit = objective.constraint_deficit(report)
        options = []
        for i in candidates:
            trial_utility, trial_report = measure(selected + [i])
            gain = trial_utility - utility
            added_images = trial_report.get("selection_image_count", len(selected) + 1) - report.get("selection_image_count", len(selected))
            # Prefer a feasible connection among positive-gain alternatives.
            # If every useful alternative is disconnected, retain an explicit
            # incomplete plan instead of pretending the hidden region vanished.
            rank = (gain > 1e-12, not trial_report.get("selection_connection_failures"),
                    gain / max(1, added_images), gain,
                    -trial_report.get("selection_image_count", len(selected) + 1),
                    -trial_report.get("selection_tour_length_m", 0), -i)
            completed = len(report["unmet_targets"]) - len(trial_report["unmet_targets"])
            progress = deficit - objective.constraint_deficit(trial_report)
            constraint_rank = None
            if completed > 0 or (completed == 0 and progress > 1e-12):
                constraint_rank = (not trial_report.get("selection_connection_failures"), completed > 0,
                                   completed / max(1, added_images), progress / max(1, added_images), gain,
                                   -trial_report.get("selection_image_count", len(selected) + 1),
                                   -trial_report.get("selection_tour_length_m", 0), -i)
            options.append((rank, gain, i, added_images, constraint_rank))
        _, gain, chosen, added_images, _ = max(options, key=lambda option: option[0])
        method = "gain_per_added_image" if evaluate else "marginal_gain"
        if gain <= 1e-12:
            # A zero ranking weight does not disable a hard goal. Evaluate the
            # actual routed set, including objects seen only by transition views.
            useful = [option for option in options if option[4] is not None]
            if not useful:
                break
            _, gain, chosen, added_images, _ = max(useful, key=lambda option: option[4])
            method = "constraint_progress"
        history[chosen] = {"selection_step": len(selected) + 1, "marginal_gain_at_selection": float(gain),
                           "selection_method": method,
                           "marginal_image_count_at_selection": added_images}
        selected.append(chosen)
        candidates.remove(chosen)
    if attainable.met(measure(selected)[1]):
        selected = _prune_redundant(selected, measure, required, attainable.met)
        if evaluate:
            # Exactly one best improving exchange, not an unbounded optimizer.
            current_report = measure(selected)[1]
            # A disconnected two-image plan must not defeat a connected
            # three-image plan. Cost optimization is secondary to feasibility.
            best_cost = (bool(current_report["selection_connection_failures"]),
                         current_report["selection_image_count"], current_report["selection_tour_length_m"])
            best_swap = None
            available = [i for i in range(len(objective.visible)) if i not in selected]
            for old in selected:
                if old in required:
                    continue
                for new in available:
                    trial = [new if i == old else i for i in selected]
                    trial_report = measure(trial)[1]
                    if not attainable.met(trial_report) or trial_report["selection_connection_failures"]:
                        continue
                    # Compare the completed exchange, not its temporary two-view
                    # state. A replacement can subsume several old views even
                    # when that temporary route is longer than the current one.
                    trial = _prune_redundant(trial, measure, required, attainable.met)
                    trial_report = measure(trial)[1]
                    cost = (bool(trial_report["selection_connection_failures"]),
                            trial_report["selection_image_count"], trial_report["selection_tour_length_m"])
                    if (attainable.met(trial_report) and not trial_report["selection_connection_failures"]
                            and cost < best_cost):
                        best_cost, best_swap = cost, (old, new, trial)
            if best_swap:
                old, new, selected = best_swap
                history[new] = {"selection_step": None, "marginal_gain_at_selection": None,
                                "selection_method": "route_cost_local_swap", "replaced_index": old,
                                "marginal_image_count_at_selection": None}
                # A replacement can make other core views redundant. Rebuild the
                # routed set for every deletion; never delete transition images
                # independently of their required graph connection.
                selected = _prune_redundant(selected, measure, required, attainable.met)
    report = dict(measure(selected)[1])
    report["attainable_goals"] = attainable.report(report)
    return selected, history, report


def repair_connected_selection(objective, evaluate, selected, history, groups, required=()):
    """Try whole feasible camera components without changing the global targets.

    A greedy stall or one-swap repair can miss a feasible tour requiring several additions/replacements.
    Each eligible component starts with all its views, then removes redundancy.
    This is bounded, not a global minimum-image optimizer.
    """
    current = evaluate(selected)[1]
    attainable = AttainableGoals(objective)
    if not current.get("selection_connection_failures") and attainable.met(current):
        return selected, history, current
    required = set(required)
    best = None
    for group in groups:
        if not required <= set(group):
            continue
        trial = sorted(group)
        report = evaluate(trial)[1]
        if not attainable.met(report) or report["selection_connection_failures"]:
            continue
        trial = _prune_redundant(trial, evaluate, required, attainable.met)
        report = evaluate(trial)[1]
        rank = (report["selection_image_count"], report["selection_tour_length_m"], -report["utility"], tuple(trial))
        if best is None or rank < best[0]:
            best = rank, trial, report
    if best is None:
        return selected, history, current
    _, selected, report = best
    history = {i: history[i] if i in required else {
        "selection_step": None, "marginal_gain_at_selection": None,
        "selection_method": "connected_component_repair", "marginal_image_count_at_selection": None}
        for i in selected}
    return selected, history, report
