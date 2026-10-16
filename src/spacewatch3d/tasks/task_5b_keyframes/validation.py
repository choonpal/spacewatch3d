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
    for field in ("scene_id", "route_id"):
        require(requests.get(field) == plan[field], "request " + field)
    if "route_version" in requests:
        require(type(requests["route_version"]) is int and requests["route_version"] == plan["route_version"], "request route_version")
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
    excluded = {v["viewpoint_id"] for v in plan.get("excluded_viewpoints", [])}
    require(not excluded.intersection(viewpoints), "excluded camera used as final image")
    if "candidate_report" in plan:
        sampling = plan["candidate_report"]
        candidates = {c["candidate_id"]: c for c in sampling["candidates"]}
        require(len(candidates) == len(sampling["candidates"]), "duplicate candidate")
        require(len(candidates) == sum(sampling[key] for key in ("uniform_seed_count", "geometry_added_count", "visibility_added_count"))
                + sampling.get("visibility_audit_added_count", 0) + sampling.get("reference_added_count", 0), "candidate refinement counts")
        require(len(candidates) == plan["coverage_report"]["candidate_count_before_feedback"], "candidate count/report")
        for candidate in candidates.values():
            require(candidate["route_node_id"] in nodes, "candidate route node")
            require(bool(candidate["reasons"]) and all(isinstance(reason, str) for reason in candidate["reasons"]), "candidate reasons")
        require(all(v["candidate_id"] in candidates for v in viewpoints.values()), "final image must be a scored candidate")
        if "visibility_audit" in sampling:
            audit = sampling["visibility_audit"]
            added = sampling["visibility_audit_added_count"]
            require(audit["status"] in ("complete", "disabled"), "visibility audit status")
            require(type(audit["probe_count"]) is int and audit["probe_count"] >= 0
                    and type(added) is int and 0 <= added <= audit["probe_count"], "visibility audit counts")
            if audit["status"] == "disabled":
                require(audit["probe_count"] == added == 0, "disabled visibility audit has probes")
            else:
                for field in ("eligible_route_node_count", "baseline_candidate_count", "excluded_route_node_count"):
                    require(type(audit[field]) is int and audit[field] >= 0, "visibility audit node counts")
                require(audit["baseline_candidate_count"] + audit["probe_count"] == audit["eligible_route_node_count"], "visibility audit eligible count")
                require(audit["eligible_route_node_count"] + audit["excluded_route_node_count"] == len(nodes), "visibility audit graph count")
                require(audit["excluded_route_node_count"] == len(plan["excluded_viewpoints"]), "visibility audit excluded count")
                require(audit["baseline_candidate_count"] + added == plan["coverage_report"]["candidate_count_after_feedback"], "visibility audit candidate count")
                require(audit["probe_count"] <= plan["provenance"]["config"]["max_visibility_probes"], "visibility audit limit")
                require(audit["sampled_observable_union_preserved"] is True and audit["best_object_quality_preserved"] is True,
                        "visibility audit preservation flags")
                if "joint_object_threshold_patterns_preserved" in audit:
                    require(audit["joint_object_threshold_patterns_preserved"] is True, "joint object preservation flag")
                    require(type(audit["joint_object_added_count"]) is int and 0 <= audit["joint_object_added_count"] <= added,
                            "joint object added count")
                    actual = sum("joint_object_observation" in c["reasons"] for c in candidates.values())
                    require(actual == audit["joint_object_added_count"], "joint object candidate reasons/count")
                if "sampled_camera_connectivity_preserved" in audit:
                    require(audit["sampled_camera_connectivity_preserved"] is True, "camera connectivity preservation flag")
                    require(type(audit["connectivity_added_count"]) is int and 0 <= audit["connectivity_added_count"] <= added,
                            "camera connectivity added count")
                    before, after = audit["camera_components_before"], audit["camera_components_after"]
                    require(type(before) is int and type(after) is int and 0 <= after <= before,
                            "camera connectivity component counts")
    require(plan["coverage_report"]["unique_image_count"] == len(viewpoints), "final image count")
    require(plan["coverage_report"]["content_count"] == sum("content" in v["roles"] for v in viewpoints.values()), "content count")
    require(plan["coverage_report"]["transition_count"] == sum("transition" in v["roles"] for v in viewpoints.values()), "transition count")
    config = plan["provenance"]["config"]
    if "independent_coverage_validation" in config:
        require("independent_validation" in plan["coverage_report"], "missing independent coverage validation")
        require(type(config["independent_coverage_validation"]) is bool, "holdout enable flag")
        require(all(type(config[key]) is int and config[key] > 0 for key in
                    ("validation_sample_multiplier", "validation_runs")), "holdout config counts")
        require(plan["coverage_report"].get("sample_role") == "selection", "primary coverage sample role")
    if "independent_validation" in plan["coverage_report"]:
        validation = plan["coverage_report"]["independent_validation"]
        require(validation["used_for_selection"] is False, "holdout samples used for selection")
        require(validation["status"] in ("passed", "failed", "disabled"), "independent validation status")
        failures = [f for f in plan["failures"] if f["code"] == "independent_coverage_targets_unmet"]
        if not config["independent_coverage_validation"]:
            require(validation["status"] == "disabled" and validation["runs"] == [] and not failures,
                    "disabled coverage validation")
        else:
            require(validation["status"] != "disabled", "enabled coverage validation skipped")
            require(validation["method"] == "fresh_surface_samples_on_fixed_eligible_route_graph", "holdout method")
            require(len(validation["selected_viewpoint_ids"]) == len(viewpoints)
                    and set(validation["selected_viewpoint_ids"]) == set(viewpoints), "holdout final image set")
            require(validation["eligible_route_node_count"] == len(nodes) - len(excluded), "holdout eligible node count")
            count = config["surface_samples"] * config["validation_sample_multiplier"]
            require(validation["sample_count_per_run"] == count, "holdout sample count")
            require(len(validation["runs"]) == config["validation_runs"], "holdout run count")
            bad_runs = []
            for index, run in enumerate(validation["runs"]):
                require(type(run["seed"]) is int and run["seed"] == config["seed"] + index + 1, "holdout seed independence")
                require(type(run["sample_count"]) is int and run["sample_count"] == count, "holdout run sample count")
                report = run["coverage_report"]
                require(report.get("sample_role") == "holdout", "holdout coverage sample role")
                require(sum(r["sample_count"] for r in report["regions"]) == count, "holdout region samples")
                values = [r["coverage_observable"] for r in report["regions"] if r["coverage_observable"] is not None]
                require(math.isclose(report["balanced_space_coverage"], sum(values) / len(values) if values else 0,
                                     abs_tol=1e-10), "holdout balanced coverage")
                for field in ("coverage_observable", "coverage_total", "balanced_space_coverage"):
                    require(0 <= report[field] <= 1 + 1e-9, "holdout coverage range")
                expected = []
                if report["observable_area_m2"] <= 0:
                    expected.append("no_observable_surface")
                if report["balanced_space_coverage"] + 1e-10 < config["space_target"]:
                    expected.append("space_target")
                if any(value + 1e-10 < config["region_target"] for value in values):
                    expected.append("region_target")
                expected.extend("region_unassessed:" + r["region_id"] for r in report["regions"] if not r["sample_count"])
                object_coverage = report["object_importance_coverage_observable"]
                if object_coverage is not None and object_coverage + 1e-10 < config["object_target"]:
                    expected.append("object_target")
                expected.extend("must_see:" + o["instance_id"] for o in plan["object_importances"]
                                if o["must_see"] and report["object_best_quality"][o["instance_id"]] < config["object_quality_threshold"])
                require(report["unmet_targets"] == expected, "holdout target results")
                if expected:
                    bad_runs.append({"seed": run["seed"], "targets": expected})
            for output, metric in (("worst_coverage_observable", "coverage_observable"),
                                   ("worst_balanced_space_coverage", "balanced_space_coverage")):
                require(math.isclose(validation[output], min(r["coverage_report"][metric] for r in validation["runs"]),
                                     abs_tol=1e-10), "holdout worst coverage")
            require(validation["status"] == ("failed" if bad_runs else "passed"), "holdout pass/fail mismatch")
            if bad_runs:
                require(plan["planning_status"] == "incomplete" and len(failures) == 1
                        and failures[0]["runs"] == bad_runs, "holdout failure cannot be published as complete")
            else:
                require(not failures, "spurious holdout failure")
    selection = plan["coverage_report"].get("selection", {})
    if "feasibility_repair_trigger" in selection:
        trigger = selection["feasibility_repair_trigger"]
        succeeded = selection["feasibility_repair_succeeded"]
        require(isinstance(trigger, list) and len(trigger) == len(set(trigger))
                and set(trigger) <= {"targets_unmet", "connection_failure"}, "feasibility repair trigger")
        require(type(succeeded) is bool and (not succeeded or bool(trigger)), "feasibility repair status")
        if succeeded:
            require(not plan["failures"] and not plan["coverage_report"]["unmet_targets"], "successful repair has failures")
    if "surface_sampling" in plan["coverage_report"]:
        sampling = plan["coverage_report"]["surface_sampling"]
        config = plan["provenance"]["config"]
        require(sampling["method"] == "geometry_named_region_area_stratified", "surface sampling method")
        require(type(sampling["sample_count"]) is int and sampling["sample_count"] == config["surface_samples"], "surface sample count")
        require(type(sampling["stratum_count"]) is int and 0 < sampling["stratum_count"] <= sampling["sample_count"], "surface stratum count")
        require(sampling["min_region_samples"] == config["min_region_samples"], "region sample minimum")
        require(math.isclose(sampling["input_area_m2"], plan["coverage_report"]["input_area_m2"], rel_tol=1e-9, abs_tol=1e-10), "sampled input area")
        regions = {r["region_id"]: r for r in plan["coverage_report"]["regions"]}
        declared = sampling["declared_regions"]
        require(len({r["region_id"] for r in declared}) == len(declared), "duplicate declared sampled region")
        require({r["region_id"] for r in declared} == {rid for rid, r in regions.items() if r.get("declared")}, "declared region coverage partition")
        require(sum(r["sample_count"] for r in regions.values()) == sampling["sample_count"], "region sample count sum")
        for item in declared:
            region = regions[item["region_id"]]
            count = item["sample_count"]
            require(type(count) is int and count >= 0 and count == region["sample_count"], "declared region sample count")
            require(math.isclose(item["input_area_m2"], region["input_area_m2"], rel_tol=1e-9, abs_tol=1e-10), "declared region input area")
            if item["status"] == "no_surface":
                require(count == item["input_area_m2"] == 0 and region["assessment_status"] == "no_surface", "empty region status")
                require(region["coverage_observable"] is None and region["coverage_total"] is None, "empty region cannot claim coverage")
                require("region_unassessed:" + item["region_id"] in plan["coverage_report"]["unmet_targets"], "unassessed region missing failure")
            else:
                require(item["status"] == region["assessment_status"] == "sampled"
                        and count >= config["min_region_samples"] and item["input_area_m2"] > 0, "declared region sample minimum/status")
    for key, expected in (
        ("detour_link_count", sum(link.get("connection_method") == "renderable_candidate_detour" for link in plan["tour_links"])),
        ("connected_component_repair_count", sum(v.get("selection_method") == "connected_component_repair" for v in viewpoints.values() if "content" in v["roles"])),
        ("constraint_progress_count", sum(v.get("selection_method") == "constraint_progress" for v in viewpoints.values() if "content" in v["roles"]))):
        if key in selection:
            require(type(selection[key]) is int and selection[key] == expected, "selection " + key)
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
    if "reference_reuse" in plan:
        transform = plan["provenance"]["canonical_route_reuse"]
        require(requests.get("coordinate_transform") == {key: transform[key] for key in
                ("current_frame_id", "reference_frame_id", "reference_from_current")}, "request coordinate transform")
        reuse = plan["reference_reuse"]
        retained = {v["viewpoint_id"]: v for v in reuse["reused_viewpoints"]}
        missing = {v["viewpoint_id"] for v in reuse["unavailable_viewpoints"]}
        new = set(reuse["new_viewpoint_ids"])
        require(len(retained) == len(reuse["reused_viewpoints"]) == reuse["reused_count"], "reference reused count")
        require(len(missing) == len(reuse["unavailable_viewpoints"]) == reuse["unavailable_count"], "reference unavailable count")
        require(len(new) == len(reuse["new_viewpoint_ids"]) == reuse["new_count"], "reference new count")
        require(not (set(retained) & missing or new & missing or new & set(retained)), "reference identity overlap")
        require(set(viewpoints) == set(retained) | new, "reference final viewpoint partition")
        require(plan["route_id"] == reuse["reference_route_id"], "reference route lineage")
        require(type(reuse["route_changed"]) is bool and
                plan["route_version"] == reuse["reference_route_version"] + int(reuse["route_changed"]), "reference route version")
        for vid, saved in retained.items():
            require(viewpoints[vid]["roles"] == saved["roles"] and
                    np.allclose(viewpoints[vid]["reference_from_camera"], saved["reference_from_camera"], rtol=0, atol=1e-8),
                    "reference pose/roles must be preserved")
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
        if "connection_method" in link:
            require(link["connection_method"] in ("shortest_path_spacing", "renderable_candidate_detour"), "connection method")
            if link["connection_method"] == "renderable_candidate_detour":
                require(measured <= plan["provenance"]["config"]["max_link_m"] + 1e-8, "detour has oversized jump")
        if plan["planning_status"] == "complete":
            require(measured <= plan["provenance"]["config"]["max_link_m"] + 1e-8, "complete tour has oversized jump")
    if plan["planning_status"] == "complete":
        require(not plan["coverage_report"]["unmet_targets"] and not plan["failures"], "complete plan has failures")
        require(bool(viewpoints), "empty plan cannot be complete")
    return True
