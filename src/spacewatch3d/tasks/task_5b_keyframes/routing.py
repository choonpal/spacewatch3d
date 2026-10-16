"""Conservative walkable raster, medial-axis graph and collision-safe paths."""

from dataclasses import dataclass
import heapq
import math
import numpy as np
import shapely
from shapely.geometry import MultiPoint
from scipy.ndimage import distance_transform_edt, label
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


def footprint(scene, config):
    polygons = []
    low, high = scene.floor_height + config.body_min_m, scene.floor_height + config.body_max_m
    for geometry in scene.geometries:
        triangles = geometry.triangles
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


def neighbors(cell, mask):
    r, c = cell
    h, w = mask.shape
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            if dr and dc and not (mask[r, nc] and mask[nr, c]):
                continue
            yield (nr, nc), math.hypot(dr, dc)


def shortest(adjacency, source, target=None):
    distances, previous, queue = {source: 0.0}, {}, [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == target:
            break
        for other, length in adjacency[node]:
            candidate = cost + length
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

    def node_id(self, node):
        r, c = self.cells[node]
        return f"n_{r:05d}_{c:05d}"

    def candidate_id(self, node):
        return "candidate_" + self.node_id(node)

    def viewpoint_id(self, node):
        return "vp_" + self.node_id(node)

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
        return {"nodes": nodes, "edges": edges, "grid_m": self.grid_m,
                "walkable_cells": int(self.walkable.sum()), "method": "medial_axis_grid_graph"}


def build_route(scene: Scene, config: Config) -> Route:
    bounds = scene.floor.bounds
    origin = np.floor(np.asarray(bounds[:2]) / config.grid_m) * config.grid_m - config.grid_m
    size = np.ceil((np.asarray(bounds[2:]) - origin) / config.grid_m).astype(int) + 1
    if int(size[0]) * int(size[1]) > config.max_grid_cells:
        raise InputError("max_grid_cells exceeded; crop the scene or increase grid_m")
    yy, xx = np.mgrid[:size[1], :size[0]]
    x, y = origin[0] + (xx + 0.5) * config.grid_m, origin[1] + (yy + 0.5) * config.grid_m
    # Whole cell, not merely center, must fit inside confirmed free floor.
    margin = config.clearance_m + config.grid_m / math.sqrt(2)
    safe_floor = scene.floor.buffer(-margin)
    obstacles = footprint(scene, config)
    walkable = shapely.contains_xy(safe_floor, x, y)
    if not obstacles.is_empty:
        walkable &= ~shapely.intersects_xy(obstacles.buffer(margin), x, y)
    clearance = distance_transform_edt(np.pad(walkable, 1), sampling=config.grid_m)[1:-1, 1:-1]
    labels, count = label(walkable)  # Four-connected: corner touching is not a doorway.
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
            stack.extend(other for other, _ in neighbors(cell, walkable) if other in marked and other not in reached)
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
                for other, length in neighbors(cell, walkable):
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
                stack.extend(other for other, _ in neighbors(cell, walkable) if other in marked and other not in reached)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(skeleton)]
    if len(cells) > config.max_route_nodes:
        raise InputError("max_route_nodes exceeded")
    lookup = {cell: i for i, cell in enumerate(cells)}
    adjacency = {i: [(lookup[other], length * config.grid_m) for other, length in neighbors(cell, walkable)
                     if other in lookup] for i, cell in enumerate(cells)}
    positions = np.asarray([[origin[0] + (c + 0.5) * config.grid_m,
                             origin[1] + (r + 0.5) * config.grid_m,
                             scene.floor_height + config.camera_height_m] for r, c in cells]).reshape((-1, 3))
    components = [int(labels[cell]) for cell in cells]
    # Deterministic graph-distance sampling, not Euclidean thinning across walls.
    nearest, candidates = np.full(len(cells), np.inf), []
    for node in range(len(cells)):
        if nearest[node] + 1e-12 < config.candidate_spacing_m:
            continue
        candidates.append(node)
        if len(candidates) > config.max_candidates:
            raise InputError("max_candidates exceeded; increase candidate_spacing_m")
        distances, _ = shortest(adjacency, node)
        for other, value in distances.items():
            nearest[other] = min(nearest[other], value)
    return Route(origin, config.grid_m, walkable, clearance, cells, positions, components,
                 adjacency, candidates, obstacles)


def connect_tour(route: Route, selected: list[int], config: Config, banned: set[int], start=None):
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
            distances, previous = shortest(route.adjacency, current)
            target = min(remaining, key=lambda n: (distances.get(n, math.inf), n))
            path = restore_path(previous, current, target)
            lengths = [0.0]
            for a, b in zip(path, path[1:]):
                lengths.append(lengths[-1] + float(np.linalg.norm(route.positions[a] - route.positions[b])))
            anchors, index = [0], 0
            while lengths[-1] - lengths[index] > config.max_link_m + 1e-9:
                options = [j for j in range(index + 1, len(path))
                           if lengths[j] - lengths[index] <= config.max_link_m + 1e-9 and path[j] in renderable]
                if not options:
                    failures.append({"code": "transition_gap", "from_node": route.node_id(path[index]),
                                     "to_node": route.node_id(target)})
                    break
                index = max(options)
                anchors.append(index)
            anchors.append(len(path) - 1)
            for a, b in zip(anchors, anchors[1:]):
                if path[b] not in selected:
                    transitions.add(path[b])
                links.append({"from_viewpoint": route.viewpoint_id(path[a]),
                              "to_viewpoint": route.viewpoint_id(path[b]),
                              "component_id": route.component_id(path[a]),
                              "length_m": lengths[b] - lengths[a],
                              "route_node_ids": [route.node_id(n) for n in path[a:b + 1]],
                              "polyline_m": route.positions[path[a:b + 1]].tolist()})
                visits.append(path[b])
            current = target
            remaining.remove(target)
    if len({route.components[node] for node in selected}) > 1:
        failures.append({"code": "disconnected_tour", "message": "Separate tours required; no teleport links created"})
    return visits, links, sorted(transitions), failures
