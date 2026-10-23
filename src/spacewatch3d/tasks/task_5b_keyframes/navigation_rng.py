"""Relative-neighborhood pruning restricted to existing safe image links.

For uv, a witness w must have BOTH safe uw and vw links, each strictly
shorter than uv. Decisions use the original graph, not input order. Every
removed edge has a replacement consisting of shorter edges, recursively;
therefore each original connected component remains connected. This is a
visibility-constrained RNG adaptation, not a bounded-stretch spanner or a
guarantee to preserve every physical corridor cycle.
"""
import copy
import heapq
import math


def _shortest_path(adjacency, start, end):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        if node == end:
            path = [node]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return distance, path[::-1]
        for other, length in sorted(adjacency[node].items()):
            candidate = distance + length
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                previous[other] = node
                heapq.heappush(queue, (candidate, other))
    raise RuntimeError('Relative-neighborhood pruning disconnected a safe link')


def relative_neighborhood_links(links):
    """Return retained links and auditable removal paths; never mutate input.

    Relative tolerance only treats floating-point ties conservatively.
    There is no meter radius, degree cap, angle threshold, or stretch limit.
    Callers supply the undirected geometry-safe links of selected images.
    """
    adjacency = {}
    for link in links:
        a, b = link['from_viewpoint'], link['to_viewpoint']
        adjacency.setdefault(a, {})[b] = link['length_m']
        adjacency.setdefault(b, {})[a] = link['length_m']
    kept, removed = [], []
    for link in links:
        a, b, length = link['from_viewpoint'], link['to_viewpoint'], link['length_m']
        witnesses = []
        for w in sorted(adjacency[a].keys() & adjacency[b].keys()):
            longest_leg = max(adjacency[a][w], adjacency[b][w])
            if longest_leg < length and not math.isclose(longest_leg, length, rel_tol=1e-12, abs_tol=0):
                witnesses.append(w)
        if witnesses:
            removed.append({'link_id': link['link_id'], 'from_viewpoint': a, 'to_viewpoint': b,
                            'length_m': length, 'witness_viewpoint': witnesses[0]})
        else:
            kept.append(link)
    final = {node: {} for node in adjacency}
    for link in kept:
        a, b = link['from_viewpoint'], link['to_viewpoint']
        final[a][b] = final[b][a] = link['length_m']
    for record in removed:
        distance, path = _shortest_path(final, record['from_viewpoint'], record['to_viewpoint'])
        record.update(replacement_path=path, replacement_length_m=distance,
                      detour_ratio=distance / record['length_m'])
    return kept, removed


def apply_relative_neighborhood(navigation):
    """Apply link policy after image optimization; preserve all source state."""
    from .navigation_free import components
    result = copy.deepcopy(navigation)
    original = result['links']
    retained, removed = relative_neighborhood_links(original)
    ids = [v['viewpoint_id'] for v in result['viewpoints']]
    groups = components(ids, [(l['from_viewpoint'], l['to_viewpoint']) for l in retained])
    before = components(ids, [(l['from_viewpoint'], l['to_viewpoint']) for l in original])
    if {frozenset(g) for g in groups} != {frozenset(g) for g in before}:
        raise RuntimeError('Relative-neighborhood pruning changed connected components')
    adjacency = {vid: [] for vid in ids}
    for link in retained:
        adjacency[link['from_viewpoint']].append(link['to_viewpoint'])
        adjacency[link['to_viewpoint']].append(link['from_viewpoint'])
    result.update(schema_version='0.4.0', visibility_links=original, links=retained,
                  adjacency=[{'viewpoint_id': vid, 'neighbors': sorted(adjacency[vid])}
                             for vid in sorted(ids)])
    result['link_pruning'] = {
        'method': 'visibility_constrained_relative_neighborhood',
        'distance_metric': 'safe_straight_chord_xy_m',
        'rule': 'both safe witness legs strictly shorter than the removed edge',
        'removed_links': removed, 'connected_components_preserved': True,
        'max_removed_edge_detour_ratio': max((r['detour_ratio'] for r in removed), default=1.0),
        'stretch_bound_enforced': False, 'every_physical_corridor_cycle_preserved': False,
    }
    result['report'].update(link_count=len(retained), visibility_link_count=len(original),
                            removed_link_count=len(removed), connected_components=len(groups),
                            cycle_rank=len(retained)-len(ids)+len(groups),
                            all_safe_selected_links_exposed=False,
                            link_policy='visibility_constrained_relative_neighborhood')
    result['warnings'] = [w for w in result['warnings']
                          if not w.startswith('All safe links between selected images are exposed;')]
    result['warnings'].append('RNG prunes navigation arrows only; visibility_links retains all safe pairs. '
                              'Connectivity is preserved, but bounded detour, bounded degree and every corridor cycle are not guaranteed.')
    return result
