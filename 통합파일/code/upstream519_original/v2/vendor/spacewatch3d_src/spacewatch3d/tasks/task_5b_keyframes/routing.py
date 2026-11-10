"""Conservative walkable raster, medial-axis graph and collision-safe paths."""

from dataclasses import dataclass, field
import heapq
import math
import numpy as np
import shapely
from shapely.geometry import MultiPoint
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis

from .models import Config, InputError, Scene


def clip_z(points, height, above):
    result = []
    for a, b in zip(points, points[1:] + points[:1]):
        ia, ib = (a[2] >= height, b[2] >= height) if above else (a[2] <= height, b[2] <= height)
        if ia:
            result.append(a)
        if ia != ib:
            result.append(a + (b - a) * ((height - a[2]) / (b[2] - a[2])))
    return result


def collision_height_interval(scene, config):
    """Above-floor union of the configured body and the actual camera lens.

    Body limits describe collision geometry, not permitted viewing heights.
    Including a taller/lower lens can only expand the checked vertical range.
    """
    height = scene.camera_height_m if config.camera_height_m is None else config.camera_height_m
    if not np.isfinite(height) or height <= 0:
        raise InputError("Camera height must be finite and above the floor")
    return min(config.body_min_m, height), max(config.body_max_m, height)


def footprint(scene, config):
    polygons = []
    low, high = collision_height_interval(scene, config)
    low, high = scene.floor_height + low, scene.floor_height + high
    for geometry in scene.geometries:
        triangles = geometry.collision_triangles if geometry.collision_triangles is not None else geometry.triangles
        if not geometry.obstacle or not len(triangles):
            continue
        if triangles[:, :, 2].max() < low or triangles[:, :, 2].min() > high:
            continue
        for triangle in triangles:
            points = list(triangle)
            if not geometry.solid:
                points = clip_z(points, low, True)
                points = clip_z(points, high, False) if points else []
            if points:
                # A vertical face projects to a LINE, not an empty polygon.
                polygons.append(MultiPoint(np.asarray(points)[:, :2]).convex_hull)
    return shapely.union_all(polygons)


def safe_segments(floor, obstacles, segments, clearance_m):
    """Check actual straight camera motions, not raster corner adjacency.

    Units and clearance are physical. No curve smoothing or wall-crossing
    shortcut is licensed by a pair of individually safe endpoints.
    """
    segments = np.asarray(segments, dtype=float).reshape((-1, 2, 2))
    if not len(segments):
        return np.zeros(0, dtype=bool)
    lines = shapely.linestrings(segments)
    boundary_distance = shapely.distance(floor.boundary, lines)
    safe = shapely.covers(floor, lines) & (boundary_distance >= clearance_m) & (boundary_distance > 0)
    if not obstacles.is_empty:
        safe &= ~shapely.intersects(obstacles, lines)
        safe &= shapely.distance(obstacles, lines) >= clearance_m
    return safe


def raster_topology(mask, origin, grid_m, floor, obstacles, clearance_m):
    """Validate every raster edge against continuous physical clearance."""
    pairs = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            nr, nc = r + dr, c + dc
            if nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pairs.append(((r, c), (nr, nc)))
    allowed_edges = set()
    if pairs:
        xy = np.asarray([[[origin[0] + (c + .5) * grid_m,
                           origin[1] + (r + .5) * grid_m] for r, c in pair] for pair in pairs])
        valid = safe_segments(floor, obstacles, xy, clearance_m)
        for pair, accepted in zip(pairs, valid):
            if accepted:
                allowed_edges.add(pair)
    allowed_edges = frozenset(allowed_edges)
    labels, count = np.zeros(mask.shape, dtype=int), 0
    for cell in map(tuple, np.argwhere(mask).tolist()):
        if labels[cell]:
            continue
        count += 1
        labels[cell] = count
        stack = [cell]
        while stack:
            current = stack.pop()
            for other, _ in neighbors(current, mask, allowed_edges):
                if not labels[other]:
                    labels[other] = count
                    stack.append(other)
    return labels, count, allowed_edges


def neighbors(cell, mask, allowed_edges=None):
    r, c = cell
    h, w = mask.shape
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            if allowed_edges is not None:
                if tuple(sorted((cell, (nr, nc)))) not in allowed_edges:
                    continue
            elif dr and dc and not (mask[r, nc] and mask[nr, c]):
                continue
            yield (nr, nc), math.hypot(dr, dc)


def shortest(adjacency, source, target=None, max_distance=math.inf):
    distances, previous, queue = {source: 0.0}, {}, [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == target:
            break
        for other, length in adjacency[node]:
            candidate = cost + length
            if candidate > max_distance + 1e-9:
                continue
            if candidate < distances.get(other, math.inf) - 1e-12:
                distances[other], previous[other] = candidate, node
                heapq.heappush(queue, (candidate, other))
    return distances, previous


def restore_path(previous, source, target):
    path = [target]
    while path[-1] != source:
        if path[-1] not in previous:
            raise InputError("No collision-free graph path")
        path.append(previous[path[-1]])
    return list(reversed(path))


def nearest_candidates(adjacency, candidates):
    """Graph Voronoi ownership; never compare distances through walls."""
    distance, owner = {}, {}
    queue = []
    for node in candidates:
        distance[node], owner[node] = 0.0, node
        heapq.heappush(queue, (0.0, node, node))
    while queue:
        cost, source, node = heapq.heappop(queue)
        if (cost, source) != (distance[node], owner[node]):
            continue
        for other, length in adjacency[node]:
            trial = (cost + length, source)
            if trial < (distance.get(other, math.inf), owner.get(other, math.inf)):
                distance[other], owner[other] = trial
                heapq.heappush(queue, (*trial, other))
    return distance, owner


def add_candidates(route, nodes, spacing, reason, config, force=False):
    # Forced anchors/discoveries bypass spacing, so no distance map is needed.
    distance = None if force else nearest_candidates(route.adjacency, route.candidates)[0]
    known, added = set(route.candidates), []
    for node in sorted(set(nodes)):
        if node in known:
            if reason not in route.candidate_reasons[node]:
                route.candidate_reasons[node].append(reason)
            continue
        if not force and distance.get(node, math.inf) + 1e-12 < spacing:
            continue
        if len(route.candidates) >= config.max_candidates:
            raise InputError("max_candidates exceeded; adjust candidate sampling settings")
        route.candidates.append(node)
        route.candidate_reasons[node] = [reason]
        known.add(node)
        added.append(node)
        if not force:
            # Only distances below spacing can reject a later candidate. Keep
            # graph-distance semantics and the original threshold tolerance;
            # this cap never applies to actual tour/connection paths.
            local, _ = shortest(route.adjacency, node, max_distance=spacing)
            for other, value in local.items():
                distance[other] = min(distance.get(other, math.inf), value)
    return added


@dataclass
class Route:
    origin: np.ndarray
    grid_m: float
    walkable: np.ndarray
    clearance: np.ndarray
    cells: list
    positions: np.ndarray
    components: list[int]
    adjacency: dict
    candidates: list[int]
    obstacles: object
    candidate_reasons: dict = field(default_factory=dict)
    candidate_sampling: dict = field(default_factory=dict)
    node_ids: dict = field(default_factory=dict)
    viewpoint_ids: dict = field(default_factory=dict)
    camera_poses: dict = field(default_factory=dict)
    raster_edges: object = None
    raster_labels: object = None

    def node_id(self, node):
        if node in self.node_ids:
            return self.node_ids[node]
        r, c = self.cells[node]
        return f"n_{r:05d}_{c:05d}"

    def candidate_id(self, node):
        return "candidate_" + self.node_id(node)

    def viewpoint_id(self, node):
        return self.viewpoint_ids.get(node, "vp_" + self.node_id(node))

    def camera_pose(self, node):
        if node in self.camera_poses:
            return self.camera_poses[node]
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        pose[:3, 3] = self.positions[node]
        return pose

    def component_id(self, node):
        return f"component_{self.components[node]:03d}"

    def export(self):
        nodes = [{"node_id": self.node_id(i), "position_m": p.tolist(),
                  "component_id": self.component_id(i)} for i, p in enumerate(self.positions)]
        edges = []
        for node in self.adjacency:
            for other, length in self.adjacency[node]:
                if other > node:
                    edges.append({"from_node": self.node_id(node), "to_node": self.node_id(other),
                                  "length_m": length,
                                  "polyline_m": [self.positions[node].tolist(), self.positions[other].tolist()]})
        result = {"nodes": nodes, "edges": edges, "grid_m": self.grid_m,
                  "walkable_cells": int(self.walkable.sum()), "method": "medial_axis_grid_graph",
                  "grid_origin_xy_m": self.origin.tolist()}
        if self.candidate_sampling.get('lateral_spur_node_count'):
            result['method']='medial_axis_grid_graph_with_safe_observation_spurs'
            result['central_node_ids']=[self.node_id(i) for i in range(self.candidate_sampling['lateral_backbone_node_count'])]
        return result


def build_route(scene: Scene, config: Config) -> Route:
    bounds = np.asarray(scene.floor.bounds)
    # The world origin has no geometric meaning. Center a new lattice on the
    # space, so a straight corridor's center is not missed merely by shifting
    # its coordinates. This does not make finite grids resolution-complete.
    anchor = (bounds[:2] + (bounds[2:] - bounds[:2]) / 2 - config.grid_m / 2
              if scene.route_grid_anchor_xy_m is None else np.asarray(scene.route_grid_anchor_xy_m))
    coordinates = (bounds.reshape(2, 2) - anchor) / config.grid_m
    # Restore only arithmetic-near integer bounds, not a physical clearance
    # allowance. Otherwise translation roundoff can add/remove a padding cell.
    tolerance = 8 * np.finfo(float).eps * (np.abs(bounds.reshape(2, 2)) + np.abs(anchor) + config.grid_m) / config.grid_m
    nearest = np.rint(coordinates)
    coordinates = np.where(np.abs(coordinates - nearest) <= tolerance, nearest, coordinates)
    first, last = np.floor(coordinates[0]).astype(int) - 1, np.ceil(coordinates[1]).astype(int) + 1
    origin = anchor + first * config.grid_m
    size = last - first + 1
    if int(size[0]) * int(size[1]) > config.max_grid_cells:
        raise InputError("max_grid_cells exceeded; crop the scene or increase grid_m")
    yy, xx = np.mgrid[:size[1], :size[0]]
    x, y = origin[0] + (xx + 0.5) * config.grid_m, origin[1] + (yy + 0.5) * config.grid_m
    # Camera/body clearance is checked at positions AND on every full edge.
    # Requiring a whole square cell adds a grid-dependent safety margin and
    # incorrectly erases safe diagonally oriented corridors.
    obstacles = footprint(scene, config)
    centers = shapely.points(x, y)
    walkable = shapely.contains_xy(scene.floor, x, y)
    # Exclude arithmetic-near clearance contacts consistently under translation.
    # This is machine precision, not an additional physical safety allowance.
    roundoff = 8 * np.finfo(float).eps * (np.abs(x) + np.abs(y) + np.max(np.abs(bounds)) + 1)
    walkable &= shapely.distance(scene.floor.boundary, centers) > config.clearance_m + roundoff
    if not obstacles.is_empty:
        walkable &= ~shapely.intersects(obstacles, centers)
        walkable &= shapely.distance(obstacles, centers) > config.clearance_m + roundoff
    clearance = distance_transform_edt(np.pad(walkable, 1), sampling=config.grid_m)[1:-1, 1:-1]
    labels, count, raster_edges = raster_topology(
        walkable, origin, config.grid_m, scene.floor, obstacles, config.clearance_m)
    skeleton = medial_axis(np.pad(walkable, 1), rng=config.seed)[1:-1, 1:-1]
    # Repair rare skeleton disconnects using only confirmed free raster cells.
    for component in range(1, count + 1):
        component_cells = [tuple(map(int, p)) for p in np.argwhere(labels == component)]
        marked = {cell for cell in component_cells if skeleton[cell]}
        if not marked:
            best = max(component_cells, key=lambda cell: (clearance[cell], cell))
            skeleton[best] = True
            marked = {best}
        reached, stack = set(), [min(marked)]
        while stack:
            cell = stack.pop()
            if cell in reached:
                continue
            reached.add(cell)
            stack.extend(other for other, _ in neighbors(cell, walkable, raster_edges) if other in marked and other not in reached)
        while marked - reached:
            queue = [(0.0, cell) for cell in sorted(reached)]
            heapq.heapify(queue)
            distance = {cell: 0.0 for cell in reached}
            previous = {}
            target = None
            while queue:
                cost, cell = heapq.heappop(queue)
                if cost != distance[cell]:
                    continue
                if cell in marked - reached:
                    target = cell
                    break
                for other, length in neighbors(cell, walkable, raster_edges):
                    value = cost + length * (1 + config.grid_m / max(clearance[other], config.grid_m))
                    if value < distance.get(other, math.inf):
                        distance[other], previous[other] = value, cell
                        heapq.heappush(queue, (value, other))
            if target is None:
                raise InputError("Skeleton component repair failed")
            while target not in reached:
                skeleton[target] = True
                marked.add(target)
                target = previous[target]
            stack = list(reached)
            reached = set()
            while stack:
                cell = stack.pop()
                if cell in reached:
                    continue
                reached.add(cell)
                stack.extend(other for other, _ in neighbors(cell, walkable, raster_edges) if other in marked and other not in reached)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(skeleton)]
    if len(cells) > config.max_route_nodes:
        raise InputError("max_route_nodes exceeded")
    lookup = {cell: i for i, cell in enumerate(cells)}
    adjacency = {i: [(lookup[other], length * config.grid_m) for other, length in neighbors(cell, walkable, raster_edges)
                     if other in lookup] for i, cell in enumerate(cells)}
    camera_height = scene.camera_height_m if config.camera_height_m is None else config.camera_height_m
    positions = np.asarray([[origin[0] + (c + 0.5) * config.grid_m,
                             origin[1] + (r + 0.5) * config.grid_m,
                             scene.floor_height + camera_height] for r, c in cells]).reshape((-1, 3))
    components = [int(labels[cell]) for cell in cells]
    # Deterministic graph-distance sampling, not Euclidean thinning across walls.
    nearest, candidates = np.full(len(cells), np.inf), []
    for node in range(len(cells)):
        if nearest[node] + 1e-12 < config.candidate_spacing_m:
            continue
        candidates.append(node)
        if len(candidates) > config.max_candidates:
            raise InputError("max_candidates exceeded; increase candidate_spacing_m")
        # Farther nodes cannot be thinned by this seed; their exact distance
        # is unnecessary. Routing itself still uses unrestricted shortest paths.
        distances, _ = shortest(adjacency, node, max_distance=config.candidate_spacing_m)
        for other, value in distances.items():
            nearest[other] = min(nearest[other], value)
    route = Route(origin, config.grid_m, walkable, clearance, cells, positions, components,
                  adjacency, candidates, obstacles)
    route.raster_edges, route.raster_labels = raster_edges, labels
    route.candidate_reasons = {node: ["uniform_seed"] for node in candidates}
    route.candidate_sampling = {"strategy": "adaptive" if config.adaptive_candidates else "uniform",
                                "uniform_seed_count": len(candidates), "geometry_added_count": 0,
                                "visibility_added_count": 0,
                                "effective_min_spacing_m": max(config.grid_m, config.candidate_min_spacing_m)}
    if config.adaptive_candidates:
        before = len(route.candidates)
        anchors = [node for node, edges in adjacency.items() if len(edges) != 2]
        add_candidates(route, anchors, config.grid_m, "junction_or_endpoint", config, force=True)
        narrow = [node for node, cell in enumerate(cells) if clearance[cell] <= config.candidate_narrow_clearance_m]
        add_candidates(route, narrow, max(config.grid_m, config.candidate_min_spacing_m), "narrow_passage", config)
        route.candidate_sampling["geometry_added_count"] = len(route.candidates) - before
    return route


def _renderable_detour(route, source, target, renderable, config, hop_paths_from=None):
    """Lazy search over scored camera hops, then restore real graph polylines.

    Minimize hop count then distance. Failed camera nodes may be traversed, but
    never used as image endpoints. Geometry obstacles remain absent from the
    walkable graph. No new or unscored camera position is invented here.
    """
    def paths(node):
        return hop_paths_from(node) if hop_paths_from else shortest(
            route.adjacency, node, max_distance=config.max_link_m)

    best, previous = {source: (0, 0.)}, {}
    queue = [(0, 0., source)]
    while queue:
        hops, length, node = heapq.heappop(queue)
        if (hops, length) != best[node]:
            continue
        if node == target:
            anchors = restore_path(previous, source, target)
            return [restore_path(paths(a)[1], a, b) for a, b in zip(anchors, anchors[1:])]
        distances, _ = paths(node)
        for other in sorted(renderable.intersection(distances)):
            if other == node or distances[other] > config.max_link_m + 1e-9:
                continue
            cost = (hops + 1, length + distances[other])
            if cost < best.get(other, (math.inf, math.inf)):
                best[other], previous[other] = cost, node
                heapq.heappush(queue, (*cost, other))
    return None


def camera_components(route, cameras, config, paths_from=None):
    """Connected components of eligible image stops, not physical obstacles."""
    if config.max_link_m is None:
        # Physical component identity is sufficient without an image hop limit.
        # Rejected camera assets do not block traversal through their positions.
        return [{node for node in cameras if route.components[node] == component}
                for component in sorted({route.components[node] for node in cameras})]
    remaining, groups = set(cameras), []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        group, stack = {root}, [root]
        while stack:
            node = stack.pop()
            distances, _ = paths_from(node) if paths_from else shortest(
                route.adjacency, node, max_distance=config.max_link_m)
            reached = {n for n in remaining.intersection(distances)
                       if distances[n] <= config.max_link_m + 1e-9}
            remaining -= reached
            group |= reached
            stack.extend(sorted(reached, reverse=True))
        groups.append(group)
    return groups


def connectivity_bridges(route, cameras, config, banned=()):
    """Preserve dense eligible-camera connectivity when thinning scored nodes.

    A rejected image stop may still be traversed physically. Never bridge a
    camera-network gap that even all eligible graph nodes cannot bridge.
    """
    from functools import lru_cache
    paths = lru_cache(maxsize=256)(lambda node: shortest(
        route.adjacency, node, max_distance=config.max_link_m))
    groups = camera_components(route, cameras, config, paths)
    if len(groups) <= 1:
        return set(), {"camera_components_before": len(groups), "camera_components_after": len(groups)}
    eligible = set(range(len(route.positions))) - set(banned)
    dense_groups = camera_components(route, eligible, config, paths)
    added = set()
    for dense in dense_groups:
        roots = [min(group & dense) for group in groups if group & dense]
        for target in roots[1:]:
            segments = _renderable_detour(route, roots[0], target, dense, config, paths)
            if segments is None:
                raise InputError("Dense camera connectivity repair failed")
            added.update(path[-1] for path in segments)
    added -= set(cameras)
    after = camera_components(route, set(cameras) | added, config, paths)
    return added, {"camera_components_before": len(groups), "camera_components_after": len(after)}


def connect_tour(route: Route, selected: list[int], config: Config, banned: set[int], start=None, paths_from=None, hop_paths_from=None):
    """Nearest-neighbor graph walk. Intermediate assets are unique, visits may repeat."""
    visits, links, transitions, failures = [], [], set(), []
    renderable = set(route.candidates) - banned
    for component in sorted({route.components[node] for node in selected}):
        remaining = set(node for node in selected if route.components[node] == component)
        if start is None:
            current = min(remaining)
        else:
            current = min(remaining, key=lambda n: (float(np.linalg.norm(route.positions[n, :2] - start)), n))
        visits.append(current)
        remaining.remove(current)
        while remaining:
            distances, previous = paths_from(current) if paths_from else shortest(route.adjacency, current)
            target = min(remaining, key=lambda n: (distances.get(n, math.inf), n))
            path = restore_path(previous, current, target)
            lengths = [0.0]
            for a, b in zip(path, path[1:]):
                lengths.append(lengths[-1] + float(np.linalg.norm(route.positions[a] - route.positions[b])))
            anchors, index, gap_node = [0], 0, None
            while config.max_link_m is not None and lengths[-1] - lengths[index] > config.max_link_m + 1e-9:
                options = [j for j in range(index + 1, len(path))
                           if lengths[j] - lengths[index] <= config.max_link_m + 1e-9 and path[j] in renderable]
                if not options:
                    gap_node = path[index]
                    break
                index = max(options)
                anchors.append(index)
            anchors.append(len(path) - 1)
            segments = [(path[a:b + 1], lengths[b] - lengths[a]) for a, b in zip(anchors, anchors[1:])]
            method = "shortest_safe_path" if config.max_link_m is None else "shortest_path_spacing"
            if gap_node is not None:
                detour = _renderable_detour(route, current, target, renderable, config, hop_paths_from)
                if detour is None:
                    failures.append({"code": "transition_gap", "from_node": route.node_id(gap_node),
                                     "to_node": route.node_id(target)})
                else:
                    segments = [(nodes, sum(float(np.linalg.norm(route.positions[a] - route.positions[b]))
                                             for a, b in zip(nodes, nodes[1:]))) for nodes in detour]
                    method = "renderable_candidate_detour"
            for nodes, length in segments:
                a, b = nodes[0], nodes[-1]
                if b not in selected:
                    transitions.add(b)
                links.append({"from_viewpoint": route.viewpoint_id(a),
                              "to_viewpoint": route.viewpoint_id(b),
                              "component_id": route.component_id(a),
                              "length_m": length, "connection_method": method,
                              "route_node_ids": [route.node_id(n) for n in nodes],
                              "polyline_m": route.positions[nodes].tolist()})
                visits.append(b)
            current = target
            remaining.remove(target)
    if len({route.components[node] for node in selected}) > 1:
        failures.append({"code": "disconnected_tour", "message": "Separate tours required; no teleport links created"})
    return visits, links, sorted(transitions), failures
