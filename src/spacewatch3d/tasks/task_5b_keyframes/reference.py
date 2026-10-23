"""Persist canonical geometry plans; reuse exact poses, never estimate alignment."""

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from shapely.affinity import affine_transform

from .loader import geometry_digest, matrix
from .models import InputError, read_json
from .routing import Route, neighbors


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def export_reference(plan):
    """Only completed geometry plans can become a baseline (not quality proof)."""
    if plan["planning_status"] != "complete":
        raise InputError("An incomplete plan cannot become a reference route")
    reuse = plan.get("reference_reuse", {})
    result = {"schema_name": "spacewatch3d.reference_tour_route", "schema_version": "0.1.0",
            "scene_id": plan["scene_id"], "source_capture_id": plan["capture_id"],
            "root_capture_id": reuse.get("root_capture_id", plan["capture_id"]),
            "synthetic": plan["synthetic"], "reference_frame_id": plan["reference_frame_id"],
            "units": "m", "up_axis": "+Z", "route_id": plan["route_id"],
            "route_version": plan["route_version"], "planning_status": "complete",
            "route_graph": plan["route_graph"],
            "viewpoints": [{key: viewpoint[key] for key in
                            ("viewpoint_id", "candidate_id", "route_node_id", "reference_from_camera", "roles")}
                           for viewpoint in plan["viewpoints"]]}
    sampling = plan.get("coverage_report", {}).get("surface_sampling") or {}
    partition = sampling.get("fallback_tiles")
    if partition is not None:
        result["fallback_tiles"] = {key: partition[key] for key in ("origin_xy_m", "tile_m", "boundary_policy")}
    return result


def load_reference(path, scene, config, registration_path=None):
    """Validate a task-local baseline and optionally align current scene into it."""
    path = Path(path).resolve()
    data = read_json(path)
    if (data.get("schema_name") != "spacewatch3d.reference_tour_route"
            or data.get("schema_version") != "0.1.0" or data.get("planning_status") != "complete"):
        raise InputError("Expected a completed spacewatch3d.reference_tour_route / 0.1.0")
    if data.get("scene_id") != scene.scene_id or data.get("synthetic") is not scene.synthetic:
        raise InputError("Reference scene_id/synthetic mismatch")
    for key in ("source_capture_id", "root_capture_id", "reference_frame_id", "route_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise InputError("Invalid reference " + key)
    if data["source_capture_id"] == scene.capture_id:
        raise InputError("Reference reuse requires a different capture_id")
    if data.get("units") != "m" or data.get("up_axis") != "+Z":
        raise InputError("Reference must use meters and +Z")
    if type(data.get("route_version")) is not int or data["route_version"] < 1:
        raise InputError("Invalid reference route_version")
    partition = data.get("fallback_tiles")
    # Older baselines used world zero. Do not silently reinterpret their tiles
    # as floor-local ones. New references persist the resolved anchor.
    tile_origin = [0., 0.]
    boundary_policy = "raw_floor"
    if "fallback_tiles" in data:
        if not isinstance(partition, dict):
            raise InputError("Invalid reference fallback_tiles")
        tile_origin, tile_m = partition.get("origin_xy_m"), partition.get("tile_m")
        boundary_policy = partition.get("boundary_policy", "raw_floor")
        if (not isinstance(tile_origin, list) or len(tile_origin) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in tile_origin)
                or type(tile_m) not in (int, float) or not np.isfinite(tile_m) or tile_m <= 0):
            raise InputError("Invalid reference fallback tile origin/width")
        if boundary_policy not in ("raw_floor", "roundoff_snap"):
            raise InputError("Invalid reference fallback tile boundary policy")
    graph = data.get("route_graph", {})
    if not np.isclose(graph.get("grid_m", 0), config.grid_m, rtol=0, atol=1e-10):
        raise InputError("Reference grid_m differs; reuse the baseline grid resolution")
    # Optional for pre-v27 baselines, which used the world-zero lattice.
    grid_origin = graph.get("grid_origin_xy_m", [0., 0.])
    if (not isinstance(grid_origin, list) or len(grid_origin) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in grid_origin)):
        raise InputError("Invalid reference planning grid origin")
    nodes, views = graph.get("nodes", []), data.get("viewpoints", [])
    if not nodes or not views or len(nodes) > config.max_route_nodes or len(views) > config.max_candidates:
        raise InputError("Empty or oversized reference graph/viewpoints")
    node_map = {}
    for node in nodes:
        nid = node.get("node_id")
        p = np.asarray(node.get("position_m"), float)
        if not isinstance(nid, str) or not nid or nid in node_map or p.shape != (3,) or not np.isfinite(p).all():
            raise InputError("Invalid or duplicate reference node")
        node_map[nid] = node
    positions = np.asarray([node["position_m"] for node in nodes])
    if not np.allclose(positions[:, 2], positions[0, 2], rtol=0, atol=1e-8):
        raise InputError("Reference must be a single-level camera graph")
    seen_positions = set()
    for p in positions:
        grid_xy = (p[:2] - grid_origin) / config.grid_m - .5
        if not np.allclose(grid_xy, np.rint(grid_xy), rtol=0, atol=1e-7):
            raise InputError("Reference nodes must use the declared planning grid")
        cell = tuple(np.rint(grid_xy).astype(int))
        if cell in seen_positions:
            raise InputError("Duplicate reference node position")
        seen_positions.add(cell)
    edges = set()
    for edge in graph.get("edges", []):
        a, b = edge.get("from_node"), edge.get("to_node")
        if a not in node_map or b not in node_map or a == b or frozenset((a, b)) in edges:
            raise InputError("Invalid or duplicate reference edge")
        pair = np.asarray([node_map[a]["position_m"], node_map[b]["position_m"]])
        if (not np.allclose(edge.get("polyline_m"), pair, rtol=0, atol=1e-8)
                or not np.isclose(edge.get("length_m", -1), np.linalg.norm(pair[1] - pair[0]), rtol=0, atol=1e-8)
                or np.any(np.abs(pair[1] - pair[0]) > config.grid_m + 1e-8)):
            raise InputError("Invalid reference edge geometry")
        edges.add(frozenset((a, b)))
    ids, view_nodes = set(), set()
    for view in views:
        vid, nid = view.get("viewpoint_id"), view.get("route_node_id")
        pose = matrix(view.get("reference_from_camera"), "reference camera pose")
        rotation = pose[:3, :3]
        if (not isinstance(vid, str) or not vid or vid in ids or nid not in node_map or nid in view_nodes
                or view.get("candidate_id") != "candidate_" + nid
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-8)
                or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                or not np.allclose(pose[:3, 3], node_map[nid]["position_m"], rtol=0, atol=1e-8)
                or view.get("roles") not in (["content"], ["transition"])):
            raise InputError("Invalid or duplicate reference viewpoint/pose")
        ids.add(vid)
        view_nodes.add(nid)
    # Keep unseen reference IDs from colliding with new deterministic IDs.
    if any(vid != "vp_" + nid for vid, nid in ((v["viewpoint_id"], v["route_node_id"]) for v in views)):
        raise InputError("Reference viewpoint IDs must match their stable route node IDs")
    transform = np.eye(4)
    if registration_path is not None:
        registration = read_json(Path(registration_path))
        expected = {"schema_name": "spacewatch3d.tour_registration", "schema_version": "0.1.0",
                    "scene_id": scene.scene_id, "reference_capture_id": data["source_capture_id"],
                    "current_capture_id": scene.capture_id, "reference_frame_id": data["reference_frame_id"],
                    "current_frame_id": scene.reference_frame_id, "status": "succeeded"}
        if any(registration.get(key) != value for key, value in expected.items()):
            raise InputError("Registration identity/status mismatch; failed or unverified alignment cannot be used")
        transform = matrix(registration.get("reference_from_current"), "reference_from_current")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-8)
                or not np.allclose(rotation[2], [0, 0, 1], rtol=0, atol=1e-8)
                or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
            raise InputError("Registration requires metric SE3 preserving +Z; normalize scale/gravity externally")
    elif scene.reference_frame_id != data["reference_frame_id"]:
        raise InputError("Different coordinate frames require successful explicit registration")
    # All current geometry and floor/regions, not just objects, enter the same frame.
    xy = [transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1], transform[0, 3], transform[1, 3]]
    scene.floor = affine_transform(scene.floor, xy)
    scene.regions = [(rid, affine_transform(region, xy)) for rid, region in scene.regions]
    scene.floor_height += float(transform[2, 3])
    if scene.start is not None:
        scene.start = (transform @ [*scene.start, 0, 1])[:2].tolist()
    for geometry in scene.geometries:
        geometry.triangles = geometry.triangles @ transform[:3, :3].T + transform[:3, 3]
        if geometry.collision_triangles is not None:
            geometry.collision_triangles = geometry.collision_triangles @ transform[:3, :3].T + transform[:3, 3]
    scene.provenance["geometry_sha256"] = geometry_digest(scene.geometries)
    current_frame_id = scene.reference_frame_id
    scene.reference_frame_id = data["reference_frame_id"]
    # This origin already belongs to the baseline frame; do not transform it
    # with the current capture or recompute it from a changed floor extent.
    scene.fallback_tile_origin_xy_m = tuple(float(v) for v in tile_origin)
    scene.fallback_tile_boundary_policy = boundary_policy
    scene.route_grid_anchor_xy_m = tuple(float(v) for v in grid_origin)
    if partition is None:
        scene.provenance.setdefault("input_warnings", []).append({
            "code": "legacy_reference_tile_anchor",
            "message": "Reference predates tile anchors; retaining its original world-zero fallback partition"})
    # A virtual canonical camera keeps its old absolute height, even if the
    # physical camera used a different height in this capture.
    height = float(positions[0, 2] - scene.floor_height)
    if not np.isfinite(height) or height <= 0:
        raise InputError("Reference camera must remain above the current floor")
    scene.provenance["requested_camera_height"] = scene.provenance["camera_height"]
    scene.camera_height_m = height
    scene.provenance["camera_height"] = {"height_above_floor_m": height, "source": "reference_route"}
    scene.provenance["canonical_route_reuse"] = {
        "reference_manifest": str(path), "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "current_frame_id": current_frame_id, "reference_frame_id": scene.reference_frame_id,
        "reference_from_current": transform.tolist(),
        "registration_manifest": str(Path(registration_path).resolve()) if registration_path else None,
        "alignment_source": "external_registration" if registration_path else "declared_common_frame"}
    return data


def reuse_route(scene, route, reference, config):
    """Keep safe content poses, rebuilding their paths from current free space.

    Historical graph positions do not become anchors merely because they are
    still safe. Their IDs can survive where the current graph independently
    needs the same position. Content images are never snapped to a centerline.
    """
    ref_cells, valid_nodes = {}, {}
    for record in reference["route_graph"]["nodes"]:
        p = np.asarray(record["position_m"])
        c, r = np.rint((p[:2] - route.origin) / config.grid_m - .5).astype(int)
        cell = (int(r), int(c))
        if 0 <= r < route.walkable.shape[0] and 0 <= c < route.walkable.shape[1] and route.walkable[cell]:
            ref_cells[cell] = record
            valid_nodes[record["node_id"]] = cell
    anchors = {valid_nodes[view["route_node_id"]] for view in reference["viewpoints"]
               if view["roles"] == ["content"] and view["route_node_id"] in valid_nodes}
    original = set(route.cells)
    marked = original | anchors
    targets = anchors - original
    if targets:
        # Only safe canonical content cameras require attachment. Recompute
        # paths through the current free mask, never through a wall/floor hole.
        queue = [(0., cell) for cell in sorted(original)]
        heapq.heapify(queue)
        distances, previous = {cell: 0. for cell in original}, {}
        while queue and targets:
            cost, cell = heapq.heappop(queue)
            if cost != distances[cell]:
                continue
            targets.discard(cell)
            for other, step in neighbors(cell, route.walkable, route.raster_edges):
                value = cost + step
                if value < distances.get(other, float("inf")) - 1e-12:
                    distances[other], previous[other] = value, cell
                    heapq.heappush(queue, (value, other))
        if targets:
            raise InputError("Cannot attach reference nodes to current free-space graph")
        for cell in sorted(anchors):
            while cell not in original:
                marked.add(cell)
                cell = previous[cell]
    cells = sorted(marked)
    if len(cells) > config.max_route_nodes:
        raise InputError("Reference attachment exceeds max_route_nodes")
    lookup = {cell: i for i, cell in enumerate(cells)}
    labels = route.raster_labels
    if labels is None:
        labels, _ = label(route.walkable)
    positions = np.asarray([[route.origin[0] + (c + .5) * config.grid_m,
                             route.origin[1] + (r + .5) * config.grid_m,
                             scene.floor_height + scene.camera_height_m] for r, c in cells]).reshape((-1, 3))
    adjacency = {i: [(lookup[other], length * config.grid_m) for other, length in neighbors(cell, route.walkable, route.raster_edges)
                     if other in lookup] for i, cell in enumerate(cells)}
    old_candidates = [lookup[route.cells[node]] for node in route.candidates]
    result = Route(route.origin, config.grid_m, route.walkable, route.clearance, cells, positions,
                   [int(labels[cell]) for cell in cells], adjacency, old_candidates, route.obstacles)
    result.raster_edges, result.raster_labels = route.raster_edges, route.raster_labels
    result.candidate_reasons = {lookup[route.cells[node]]: list(route.candidate_reasons[node]) for node in route.candidates}
    result.candidate_sampling = dict(route.candidate_sampling)
    result.candidate_sampling["reference_added_count"] = 0
    reserved = {n["node_id"] for n in reference["route_graph"]["nodes"]}
    for node, cell in enumerate(cells):
        if cell in ref_cells:
            # Stable identity is not a requirement to retain that old position.
            result.node_ids[node] = ref_cells[cell]["node_id"]
        else:
            nid = "n_added_" + digest(positions[node].tolist())[:20]
            if nid in reserved:
                raise InputError("New/reference node ID collision")
            result.node_ids[node] = nid
    report = {"reference_route_id": reference["route_id"], "reference_route_version": reference["route_version"],
              "root_capture_id": reference["root_capture_id"], "source_capture_id": reference["source_capture_id"],
              "reused_viewpoints": [], "unavailable_viewpoints": [], "new_viewpoint_ids": [],
              "viewpoint_reuse_policy": "retain_safe_content_replan_transitions",
              "reference_core_count": sum(v["roles"] == ["content"] for v in reference["viewpoints"]),
              "released_transition_viewpoints": []}
    required = []
    for view in reference["viewpoints"]:
        if view["roles"] == ["transition"]:
            report["released_transition_viewpoints"].append(dict(view))
            continue
        if view["route_node_id"] not in valid_nodes:
            report["unavailable_viewpoints"].append({"viewpoint_id": view["viewpoint_id"],
                "reference_from_camera": view["reference_from_camera"], "reason": "blocked_or_outside_current_walkable_floor"})
            continue
        node = lookup[valid_nodes[view["route_node_id"]]]
        if node not in result.candidates:
            if len(result.candidates) >= config.max_candidates:
                raise InputError("Reference viewpoints exceed max_candidates")
            result.candidates.append(node)
            result.candidate_reasons[node] = []
            result.candidate_sampling["reference_added_count"] += 1
        result.candidate_reasons[node].append("reference_viewpoint")
        result.viewpoint_ids[node] = view["viewpoint_id"]
        result.camera_poses[node] = np.asarray(view["reference_from_camera"], float)
        report["reused_viewpoints"].append(dict(view))
        required.append(node)
    return result, required, report


def finish_reuse(plan, reference, report):
    """Explicit lineage; a new view never impersonates a moved baseline camera."""
    final_ids = {v["viewpoint_id"] for v in plan["viewpoints"]}
    report["reused_viewpoints"] = [v for v in report["reused_viewpoints"] if v["viewpoint_id"] in final_ids]
    reused = {v["viewpoint_id"] for v in report["reused_viewpoints"]}
    report["new_viewpoint_ids"] = sorted(final_ids - reused)
    report["unavailable_count"] = len(report["unavailable_viewpoints"])
    report["reused_count"] = len(reused)
    report["new_count"] = len(report["new_viewpoint_ids"])
    report["released_transition_count"] = len(report["released_transition_viewpoints"])
    report["comparison_status"] = ("reference_views_preserved" if not report["unavailable_count"] else
                                   "partial_reference_views" if reused else "no_reference_views")
    original = {v["viewpoint_id"]: v for v in reference["viewpoints"]}
    current = {v["viewpoint_id"]: {key: v[key] for key in
               ("viewpoint_id", "candidate_id", "route_node_id", "reference_from_camera", "roles")}
               for v in plan["viewpoints"]}
    changed = digest(reference["route_graph"]) != digest(plan["route_graph"]) or digest(original) != digest(current)
    plan["route_id"] = reference["route_id"]
    plan["route_version"] = reference["route_version"] + int(changed)
    report["route_changed"] = changed
    # Coverage-driven replacements are not one-to-one or equivalent views.
    report["replacement_policy"] = "retain_safe_content_then_add_for_current_coverage_and_connections"
    plan["reference_reuse"] = report
