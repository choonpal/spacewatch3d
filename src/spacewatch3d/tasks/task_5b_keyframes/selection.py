"""Marginal coverage/object utility, optional routed image cost, pruning and one swap."""

import numpy as np

from .models import Config


class Objective:
    def __init__(self, visible, quality, weights, regions, objects, config: Config):
        self.visible, self.quality = visible, quality
        self.weights, self.regions, self.objects, self.config = weights, np.asarray(regions), objects, config
        self.observable = np.any(visible, axis=0) if len(visible) else np.zeros(len(weights), dtype=bool)
        self.importance = np.asarray([obj["importance"] for obj in objects])
        self.possible_objects = np.any(quality >= config.object_quality_threshold, axis=0) if len(quality) else np.zeros(len(objects), bool)
        self.region_ids = sorted(set(regions))

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
                               "observable_area_m2": possible, "covered_area_m2": seen,
                               "coverage_observable": feasible_seen / possible if possible else None,
                               "coverage_total": seen / total if total else 0})
        feasible_regions = [r["coverage_observable"] for r in per_region if r["coverage_observable"] is not None]
        c_space = float(np.mean(feasible_regions)) if feasible_regions else 0.0
        importance_sum = float(self.importance.sum())
        c_object = float(best @ self.importance / importance_sum) if importance_sum else 0.0
        alpha, beta = self.config.space_weight, self.config.object_weight if importance_sum else 0.0
        if alpha + beta == 0:
            alpha = 1.0  # No semantic weights -> space-only, even for object-only config.
        utility = (alpha * c_space + beta * c_object) / (alpha + beta) if alpha + beta else 0.0
        observed = best >= self.config.object_quality_threshold
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


def _prune_redundant(selected, measure):
    """Recheck routed goals after each deletion until no core view is removable.

    Routed utility is not monotone: deleting a core view can change the order and
    the transition images. A previously rejected deletion may then become valid.
    Every accepted step removes a core view, so this loop is finite.
    """
    selected = list(selected)
    while True:
        changed = False
        for chosen in list(reversed(selected)):
            trial = [i for i in selected if i != chosen]
            current_report = measure(selected)[1]
            trial_report = measure(trial)[1]
            if (not trial_report["unmet_targets"] and not trial_report.get("selection_connection_failures")
                    and trial_report.get("selection_image_count", len(trial)) <= current_report.get("selection_image_count", len(selected))):
                selected = trial
                changed = True
        if not changed:
            return selected


def select_keyframes(objective: Objective, evaluate=None):
    """Use pure gain, or exact routed-set gain per additional unique image.

    The injected evaluator may include transition viewpoints in utility/goals.
    A bounded one-swap search improves cost without relaxing any target.
    """
    measure = evaluate or objective.measure
    selected, history = [], {}
    candidates = list(range(len(objective.visible)))
    while candidates:
        utility, report = measure(selected)
        if not report["unmet_targets"]:
            break
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
            options.append((rank, gain, i, added_images))
        _, gain, chosen, added_images = max(options)
        # Thresholded object goals can need a useful view with tiny continuous gain.
        if gain <= 1e-12:
            # Zero importance must not silently disable explicit must-see constraints.
            pending = [j for j, obj in enumerate(objective.objects)
                       if obj["must_see"] and "must_see:" + obj["instance_id"] in report["unmet_targets"]]
            useful = [i for i in candidates if any(objective.quality[i, j] >= objective.config.object_quality_threshold for j in pending)]
            if not useful:
                break
            chosen = useful[0]
            _, gain, _, added_images = next(option for option in options if option[2] == chosen)
        history[chosen] = {"selection_step": len(selected) + 1, "marginal_gain_at_selection": float(gain),
                           "selection_method": "gain_per_added_image" if evaluate else "marginal_gain",
                           "marginal_image_count_at_selection": added_images}
        selected.append(chosen)
        candidates.remove(chosen)
    if not measure(selected)[1]["unmet_targets"]:
        selected = _prune_redundant(selected, measure)
        if evaluate:
            # Exactly one best improving exchange, not an unbounded optimizer.
            current_report = measure(selected)[1]
            best_cost = (current_report["selection_image_count"], current_report["selection_tour_length_m"])
            best_swap = None
            available = [i for i in range(len(objective.visible)) if i not in selected]
            for old in selected:
                for new in available:
                    trial = [new if i == old else i for i in selected]
                    trial_report = measure(trial)[1]
                    cost = (trial_report["selection_image_count"], trial_report["selection_tour_length_m"])
                    if (not trial_report["unmet_targets"] and not trial_report["selection_connection_failures"]
                            and cost < best_cost):
                        best_cost, best_swap = cost, (old, new)
            if best_swap:
                old, new = best_swap
                selected = [new if i == old else i for i in selected]
                history[new] = {"selection_step": None, "marginal_gain_at_selection": None,
                                "selection_method": "route_cost_local_swap", "replaced_index": old,
                                "marginal_image_count_at_selection": None}
                # A replacement can make other core views redundant. Rebuild the
                # routed set for every deletion; never delete transition images
                # independently of their required graph connection.
                selected = _prune_redundant(selected, measure)
    return selected, history, measure(selected)[1]
