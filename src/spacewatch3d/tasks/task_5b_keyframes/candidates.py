"""Local refinement plus an independent, bounded route-node visibility audit."""

import numpy as np

from .models import InputError
from .routing import add_candidates, nearest_candidates, restore_path, shortest


def refine_visibility_candidates(route, visible, quality, surface_weights, config):
    if not config.adaptive_candidates or not route.candidates:
        return []
    # Freeze the coarse owners for this pass; the planner scores every new node
    # before fixing the final observable denominator and selecting keyframes.
    index = {node: i for i, node in enumerate(route.candidates)}
    _, owners = nearest_candidates(route.adjacency, route.candidates)
    pairs = set()
    for node, edges in route.adjacency.items():
        for other, _ in edges:
            a, b = owners[node], owners[other]
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    refine = set()
    changed = 0
    for a, b in sorted(pairs):
        va, vb = visible[index[a]], visible[index[b]]
        union = float(surface_weights[va | vb].sum())
        area_change = float(surface_weights[va ^ vb].sum()) / union if union else 0.0
        object_change = float(np.max(np.abs(quality[index[a]] - quality[index[b]]))) if quality.shape[1] else 0.0
        if max(area_change, object_change) + 1e-12 < config.candidate_visibility_change_threshold:
            continue
        _, previous = shortest(route.adjacency, a, b)
        refine.update(restore_path(previous, a, b))
        changed += 1
    added = add_candidates(route, refine, max(config.grid_m, config.candidate_min_spacing_m),
                           "visibility_change", config)
    route.candidate_sampling.update(visibility_added_count=len(added), changed_neighbor_pairs=changed,
                                    visibility_refinement_passes=1)
    return added


def audit_visibility_candidates(route, visible, quality, surface_weights, config, evaluate, banned=()):
    """Probe every unscored, eligible graph node, independently of seed scores.

    Keep a subset preserving the dense sampled-surface union and the best quality
    of each object (within 1e-12). Close discoveries bypass distance thinning:
    being near a seed is not evidence that the seed sees through the same gap.
    This is a discrete graph/surface audit, not a continuous-space guarantee.
    """
    empty_visible = np.zeros((0, visible.shape[1]), bool)
    empty_quality = np.zeros((0, quality.shape[1]))
    route.candidate_sampling["visibility_audit_added_count"] = 0
    if not config.adaptive_candidates:
        route.candidate_sampling["visibility_audit"] = {"status": "disabled", "probe_count": 0}
        return [], empty_visible, empty_quality
    banned = set(banned)
    known = set(route.candidates)
    probes = [node for node in range(len(route.positions)) if node not in known and node not in banned]
    if len(probes) > config.max_visibility_probes:
        raise InputError("max_visibility_probes exceeded; crop the scene or explicitly raise the visibility audit limit")
    baseline = [i for i, node in enumerate(route.candidates) if node not in banned]
    seen = np.any(visible[baseline], axis=0) if baseline else np.zeros(visible.shape[1], bool)
    best_quality = np.max(quality[baseline], axis=0) if baseline else np.zeros(quality.shape[1])
    audit = {"status": "complete", "method": "all_eligible_route_graph_nodes",
             "probe_count": len(probes), "eligible_route_node_count": len(route.positions) - len(banned),
             "baseline_candidate_count": len(baseline), "excluded_route_node_count": len(banned),
             "sampled_observable_union_preserved": True, "best_object_quality_preserved": True}
    if not probes:
        route.candidate_sampling["visibility_audit"] = audit
        return [], empty_visible, empty_quality
    probe_visible, probe_quality = evaluate(probes)
    retained = set()
    reasons = {}

    def retain(index, reason):
        nonlocal seen, best_quality
        retained.add(index)
        reasons.setdefault(index, set()).add(reason)
        seen |= probe_visible[index]
        best_quality = np.maximum(best_quality, probe_quality[index])

    # Object maxima are protected even for a tiny/high object whose area would
    # barely affect the surface-area objective. No LLM or object-size filter.
    for j in range(quality.shape[1]):
        index = int(np.argmax(probe_quality[:, j]))
        crosses_threshold = best_quality[j] < config.object_quality_threshold <= probe_quality[index, j]
        if probe_quality[index, j] > best_quality[j] + 1e-12 or crosses_threshold:
            retain(index, "object_quality_discovery")
    dense_union = seen | np.any(probe_visible, axis=0)
    while np.any(dense_union & ~seen):
        missing = dense_union & ~seen
        gains = probe_visible[:, missing] @ surface_weights[missing]
        index = int(np.argmax(gains))
        if gains[index] <= 0:
            # Zero-weight patches still must not invalidate the union claim.
            index = int(np.flatnonzero(np.any(probe_visible[:, missing], axis=1))[0])
        retain(index, "surface_visibility_discovery")
    indices = sorted(retained)
    if len(route.candidates) + len(indices) > config.max_candidates:
        raise InputError("max_candidates exceeded by visibility discoveries; crop the scene or explicitly raise the candidate limit")
    nodes = [probes[i] for i in indices]
    added = add_candidates(route, nodes, config.grid_m, "visibility_audit", config, force=True)
    for index, node in zip(indices, added):
        route.candidate_reasons[node].extend(sorted(reasons[index]))
    route.candidate_sampling.update(visibility_audit_added_count=len(added), visibility_audit=audit)
    return added, probe_visible[indices], probe_quality[indices]
