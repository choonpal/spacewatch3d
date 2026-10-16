"""Validate new task-local plans; never feed them into the legacy keyframes kind."""

import math
import numpy as np

from .models import InputError


def validate_outputs(plan, requests):
    def require(condition, message):
        if not condition:
            raise InputError("Output validation: " + message)
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), "non-finite number")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(plan)
    finite(requests)
    require(plan.get("schema_name") == "spacewatch3d.virtual_tour_plan", "plan schema")
    require(requests.get("schema_name") == "spacewatch3d.virtual_tour_render_requests", "request schema")
    require(plan.get("schema_version") == requests.get("schema_version") == "0.1.0", "schema version")
    for field in ("scene_id", "capture_id", "reference_frame_id", "route_id"):
        require(isinstance(plan.get(field), str) and bool(plan[field]), field)
    require(plan["planning_status"] in ("complete", "incomplete"), "planning_status")
    require(plan["quality_status"] in ("unverified", "partial", "failed"), "prototype cannot claim validated imagery")
    require(plan["publication_ready"] is False, "prototype is not a publication validator")
    require(plan["units"] == "m" and plan["up_axis"] == "+Z", "coordinate convention")
    require(requests.get("geometry_sha256") == plan["provenance"]["geometry_sha256"], "request geometry identity")
    require(isinstance(plan["synthetic"], bool), "synthetic")
    require(isinstance(plan["route_version"], int) and plan["route_version"] >= 1, "route version")
    nodes = {n["node_id"]: n for n in plan["route_graph"]["nodes"]}
    require(len(nodes) == len(plan["route_graph"]["nodes"]), "duplicate route node")
    edges = set()
    for edge in plan["route_graph"]["edges"]:
        a, b = edge["from_node"], edge["to_node"]
        require(a in nodes and b in nodes, "edge references missing node")
        require(nodes[a]["component_id"] == nodes[b]["component_id"], "edge crosses components")
        require(np.allclose(edge["polyline_m"], [nodes[a]["position_m"], nodes[b]["position_m"]]), "edge polyline")
        require(math.isclose(edge["length_m"], float(np.linalg.norm(np.asarray(nodes[a]["position_m"]) - nodes[b]["position_m"])), abs_tol=1e-8), "edge length")
        edges.add(frozenset((a, b)))
    viewpoints = {v["viewpoint_id"]: v for v in plan["viewpoints"]}
    require(len(viewpoints) == len(plan["viewpoints"]), "duplicate viewpoint")
    if "candidate_report" in plan:
        sampling = plan["candidate_report"]
        candidates = {c["candidate_id"]: c for c in sampling["candidates"]}
        require(len(candidates) == len(sampling["candidates"]), "duplicate candidate")
        require(len(candidates) == sum(sampling[key] for key in ("uniform_seed_count", "geometry_added_count", "visibility_added_count")), "candidate refinement counts")
        require(len(candidates) == plan["coverage_report"]["candidate_count_before_feedback"], "candidate count/report")
        for candidate in candidates.values():
            require(candidate["route_node_id"] in nodes, "candidate route node")
            require(bool(candidate["reasons"]) and all(isinstance(reason, str) for reason in candidate["reasons"]), "candidate reasons")
        require(all(v["candidate_id"] in candidates for v in viewpoints.values()), "final image must be a scored candidate")
    require(plan["coverage_report"]["unique_image_count"] == len(viewpoints), "final image count")
    require(plan["coverage_report"]["content_count"] == sum("content" in v["roles"] for v in viewpoints.values()), "content count")
    require(plan["coverage_report"]["transition_count"] == sum("transition" in v["roles"] for v in viewpoints.values()), "transition count")
    for viewpoint in viewpoints.values():
        pose = np.asarray(viewpoint["reference_from_camera"], float)
        require(pose.shape == (4, 4) and np.allclose(pose[3], [0, 0, 0, 1]), "pose shape")
        require(np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
                and math.isclose(np.linalg.det(pose[:3, :3]), 1, abs_tol=1e-8), "camera rotation")
        require(bool(viewpoint["roles"]) and set(viewpoint["roles"]) <= {"content", "transition"}, "viewpoint roles")
        require(viewpoint["route_node_id"] in nodes, "viewpoint route node")
        require(np.allclose(pose[:3, 3], nodes[viewpoint["route_node_id"]]["position_m"]), "pose translation")
        require(viewpoint["component_id"] == nodes[viewpoint["route_node_id"]]["component_id"], "viewpoint component")
        require(viewpoint["render_status"] in ("pending", "succeeded"), "render status")
        require((viewpoint["image_ref"] is None) == (viewpoint["render_status"] == "pending"), "image status")
        require(isinstance(viewpoint["image_ref"], (str, type(None))), "image_ref type")
        for observation in viewpoint["object_observations"]:
            require(0 <= observation["importance"] <= 1 and 0 <= observation["geometry_quality"] <= 1 + 1e-8, "object scores in [0,1]")
        require(not {"frame_id", "source_frame_index", "timestamp_s"} & set(viewpoint), "fabricated original-frame fields")
    reqs = {r["viewpoint_id"]: r for r in requests["requests"]}
    require(len(reqs) == len(requests["requests"]) and set(reqs) == set(viewpoints), "request/viewpoint IDs")
    for vid, request in reqs.items():
        require(np.allclose(request["reference_from_camera"], viewpoints[vid]["reference_from_camera"]), "request pose")
        require(request["capture_id"] == plan["capture_id"] and request["reference_frame_id"] == plan["reference_frame_id"], "request frame")
        require(request["projection"] == "equirectangular" and request["media_origin"] == "novel_view", "request projection")
        require(request["render_required"] == (viewpoints[vid]["render_status"] == "pending"), "render_required")
        require(isinstance(request["width"], int) and isinstance(request["height"], int)
                and request["height"] > 0 and request["width"] == 2 * request["height"], "resolution")
    for visit in plan["tour_visits"]:
        require(visit["viewpoint_id"] in viewpoints, "visit references missing viewpoint")
    require(set(v["viewpoint_id"] for v in plan["tour_visits"]) == set(viewpoints), "every final image must be visited")
    actual_pairs = [(a["viewpoint_id"], b["viewpoint_id"]) for a, b in zip(plan["tour_visits"], plan["tour_visits"][1:])
                    if a["component_id"] == b["component_id"]]
    require(actual_pairs == [(link["from_viewpoint"], link["to_viewpoint"]) for link in plan["tour_links"]], "visit/link sequence")
    for link in plan["tour_links"]:
        a, b = link["from_viewpoint"], link["to_viewpoint"]
        require(a in viewpoints and b in viewpoints, "tour link references missing viewpoint")
        ids = link["route_node_ids"]
        require(bool(ids) and ids[0] == viewpoints[a]["route_node_id"] and ids[-1] == viewpoints[b]["route_node_id"], "tour path endpoints")
        require(all(frozenset((u, v)) in edges for u, v in zip(ids, ids[1:])), "tour path must follow graph edges")
        require(np.allclose(link["polyline_m"], [nodes[n]["position_m"] for n in ids]), "tour polyline")
        measured = sum(float(np.linalg.norm(np.asarray(nodes[u]["position_m"]) - nodes[v]["position_m"])) for u, v in zip(ids, ids[1:]))
        require(math.isclose(link["length_m"], measured, abs_tol=1e-8), "tour path length")
        if plan["planning_status"] == "complete":
            require(measured <= plan["provenance"]["config"]["max_link_m"] + 1e-8, "complete tour has oversized jump")
    if plan["planning_status"] == "complete":
        require(not plan["coverage_report"]["unmet_targets"] and not plan["failures"], "complete plan has failures")
        require(bool(viewpoints), "empty plan cannot be complete")
    return True
