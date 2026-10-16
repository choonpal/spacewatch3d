"""One-pass visibility-driven refinement of a collision-safe route graph."""

import numpy as np

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
