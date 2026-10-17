"""Public planning API. Inputs stay read-only; output is a new directory."""

from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import numpy as np

from .importance import resolve_importance
from .coverage_validation import validate_coverage
from .coverage_repair import JointObjective, finish_repair_report, make_objectives, prepare_repair_samples
from .candidates import audit_visibility_candidates, refine_visibility_candidates
from .loader import load_scene
from .models import Config, InputError, read_json
from .routing import build_route, camera_components, connect_tour, shortest
from .selection import repair_connected_selection, select_keyframes
from .selection_guard import finish_guard_report, guard_objectives
from .validation import validate_outputs
from .visibility import evaluate_positions, sample_surface
from .reference import export_reference, finish_reuse, load_reference, reuse_route


def read_feedback(path, scene, route):
    if path is None:
        return {}, set(), []
    data = read_json(Path(path))
    if data.get("schema_name") != "spacewatch3d.virtual_tour_render_feedback" or data.get("schema_version") != "0.1.0":
        raise InputError("Invalid render feedback schema")
    for field in ("scene_id", "capture_id", "reference_frame_id"):
        if data.get(field) != getattr(scene, field):
            raise InputError(f"Feedback {field} mismatch")
    if data.get("geometry_sha256") != scene.provenance["geometry_sha256"]:
        raise InputError("Feedback geometry_sha256 mismatch")
    valid_ids = {route.viewpoint_id(i): i for i in range(len(route.cells))}
    results, banned, warnings = {}, set(), []
    for record in data.get("results", []):
        vid = record.get("viewpoint_id")
        if vid not in valid_ids or vid in results:
            raise InputError("Unknown or duplicate feedback viewpoint_id")
        node = valid_ids[vid]
        pose = np.asarray(record.get("reference_from_camera"), dtype=float)
        expected = route.camera_pose(node)
        # Pose identity is absolute in the metric reference frame. A relative
        # tolerance would accept a different camera farther from the origin.
        if pose.shape != (4, 4) or not np.allclose(pose, expected, rtol=0, atol=1e-8):
            raise InputError("Feedback pose mismatch; do not reuse feedback after moving a viewpoint")
        if record.get("render_status") not in ("failed", "succeeded"):
            raise InputError("Invalid feedback render_status")
        if record.get("quality_accepted") is not None and not isinstance(record["quality_accepted"], bool):
            raise InputError("quality_accepted must be boolean or null")
        if record["render_status"] == "failed" or record.get("quality_accepted") is False:
            banned.add(node)
        elif record.get("image_ref"):
            image_path = (Path(path).resolve().parent / record["image_ref"]).resolve()
            if not image_path.is_file():
                raise InputError(f"Feedback image missing: {image_path}")
            record = {**record, "image_path": str(image_path)}
        else:
            raise InputError("Succeeded feedback requires existing image_ref")
        results[vid] = record
    if results:
        warnings.append({"code": "external_quality_not_publication_validated",
                         "message": "Feedback is recorded, not independently verified image quality"})
    return results, banned, warnings


def create_plan(scene_path, config=None, importance_cache=None, importance_provider=None, feedback_path=None,
                reference_route_path=None, registration_path=None):
    """Return (plan, requests); only an explicitly injected provider may call an LLM."""
    config = Config.from_dict(asdict(config) if isinstance(config, Config) else config)
    scene = load_scene(Path(scene_path), config)
    if registration_path is not None and reference_route_path is None:
        raise InputError("registration requires reference_route_path")
    reference = load_reference(reference_route_path, scene, config, registration_path) if reference_route_path else None
    config = replace(config, camera_height_m=scene.camera_height_m)
    objects, warnings = resolve_importance(scene.objects, config, importance_cache, importance_provider)
    if scene.provenance["camera_height"]["source"] == "fallback_unverified":
        warnings.append({"code": "camera_height_fallback", "message": "No measured camera height supplied; using unverified 1.5m above floor"})
    route = build_route(scene, config)
    required_nodes, reuse_report = [], None
    if reference:
        route, required_nodes, reuse_report = reuse_route(scene, route, reference, config)
    feedback, banned, feedback_warnings = read_feedback(feedback_path, scene, route)
    warnings.extend(feedback_warnings)
    if reuse_report:
        rejected = [v for v in reuse_report["reused_viewpoints"] if v["viewpoint_id"] in
                    {route.viewpoint_id(n) for n in banned}]
        for view in rejected:
            reuse_report["unavailable_viewpoints"].append({"viewpoint_id": view["viewpoint_id"],
                "reference_from_camera": view["reference_from_camera"], "reason": "renderer_failure_or_quality_rejection"})
    surface = sample_surface(scene, config)
    all_visible, all_quality = evaluate_positions(route.positions[route.candidates], surface, objects, config)
    refined = refine_visibility_candidates(route, all_visible, all_quality, surface.weights, config)
    if refined:
        refined_visible, refined_quality = evaluate_positions(route.positions[refined], surface, objects, config)
        all_visible = np.concatenate([all_visible, refined_visible])
        all_quality = np.concatenate([all_quality, refined_quality])
    audited, audited_visible, audited_quality = audit_visibility_candidates(
        route, all_visible, all_quality, surface.weights, config,
        lambda nodes: evaluate_positions(route.positions[nodes], surface, objects, config), banned)
    if audited:
        all_visible = np.concatenate([all_visible, audited_visible])
        all_quality = np.concatenate([all_quality, audited_quality])
    datasets, coverage_repair = prepare_repair_samples(
        scene, route, surface, all_visible, all_quality, objects, config, banned)
    all_visible, all_quality = datasets[0]["visible"], datasets[0]["quality"]
    allowed_indices = [i for i, node in enumerate(route.candidates) if node not in banned]
    candidates = [route.candidates[i] for i in allowed_indices]
    visible, quality = all_visible[allowed_indices], all_quality[allowed_indices]
    objectives = make_objectives(datasets, allowed_indices, objects, config, [rid for rid, _ in scene.regions])
    objective = objectives[0]
    guarded_objectives = guard_objectives(objectives)
    optimization = JointObjective(guarded_objectives) if config.coverage_repair else guarded_objectives[0]
    lookup = {node: i for i, node in enumerate(candidates)}

    @lru_cache(maxsize=32)
    def paths_from(node):
        return shortest(route.adjacency, node)

    @lru_cache(maxsize=256)
    def hop_paths_from(node):
        return shortest(route.adjacency, node, max_distance=config.max_link_m)

    @lru_cache(maxsize=256)
    def evaluate_routed_set(key):
        nodes = [candidates[i] for i in key]
        _, tour_links, transition_nodes, connection_failures = connect_tour(route, nodes, config, banned, scene.start, paths_from, hop_paths_from)
        final = set(nodes) | set(transition_nodes)
        utility, selection_report = optimization.measure([lookup[node] for node in sorted(final)])
        selection_report.update(selection_image_count=len(final),
                                selection_tour_length_m=sum(link["length_m"] for link in tour_links),
                                selection_connection_failures=[f["code"] for f in connection_failures])
        return utility, selection_report

    evaluator = (lambda selected: evaluate_routed_set(tuple(sorted(selected)))) if config.route_aware_selection else None
    required_indices = [lookup[n] for n in required_nodes if n not in banned]
    chosen, history, selection_report = select_keyframes(optimization, evaluator, required=required_indices)
    repair_trigger = []
    if evaluator and (selection_report["selection_connection_failures"] or selection_report["unmet_targets"]):
        repair_trigger = (["connection_failure"] if selection_report["selection_connection_failures"] else [])
        if selection_report["unmet_targets"]:
            repair_trigger.append("targets_unmet")
        groups = [[lookup[n] for n in sorted(group)] for group in
                  camera_components(route, candidates, config, hop_paths_from)]
        chosen, history, _ = repair_connected_selection(optimization, evaluator, chosen, history, groups, required_indices)
    selected_nodes = [candidates[i] for i in chosen]
    reference_roles = {v["viewpoint_id"]: v["roles"] for v in reference["viewpoints"]} if reference else {}
    content_nodes = [n for n in selected_nodes if reference_roles.get(route.viewpoint_id(n)) != ["transition"]]
    content_report = objective.measure([lookup[n] for n in content_nodes])[1]
    visits, links, transitions, failures = connect_tour(route, selected_nodes, config, banned, scene.start, paths_from, hop_paths_from)
    final_nodes = sorted(set(selected_nodes) | set(transitions))
    transition_nodes = set(final_nodes) - set(content_nodes)
    # Every transition is an already-scored candidate. Never redefine coverage
    # denominators to just the selected images or hide unobservable geometry.
    report = objective.measure([lookup[node] for node in final_nodes])[1]
    report["sample_role"] = "selection"
    report["content_only"] = content_report
    report["selection"] = {"strategy": "routed_gain_per_added_image_with_one_swap" if evaluator else "marginal_gain",
                           "tour_length_m": sum(link["length_m"] for link in links),
                           "detour_link_count": sum(link["connection_method"] == "renderable_candidate_detour" for link in links),
                           "constraint_progress_count": sum(history[i]["selection_method"] == "constraint_progress" for i in chosen),
                           "connected_component_repair_count": sum(history[i]["selection_method"] == "connected_component_repair" for i in chosen),
                           "feasibility_repair_trigger": repair_trigger,
                           "feasibility_repair_succeeded": bool(repair_trigger) and not failures and not report["unmet_targets"],
                           "local_swap_applied": any(history[i]["selection_method"] == "route_cost_local_swap" for i in chosen)}
    report["candidate_count_before_feedback"] = len(route.candidates)
    report["candidate_count_after_feedback"] = len(candidates)
    report["observable_area_before_feedback_m2"] = float(surface.weights[np.any(all_visible, axis=0)].sum()) if len(all_visible) else 0
    report["unique_image_count"] = len(final_nodes)
    report["content_count"] = len(content_nodes)
    report["transition_count"] = len(transition_nodes)
    if getattr(surface, "sampling_report", None) is not None:
        report["surface_sampling"] = surface.sampling_report
    if not final_nodes:
        failures.append({"code": "no_selected_viewpoints"})
    if report["unmet_targets"]:
        failures.append({"code": "targets_unmet", "targets": report["unmet_targets"]})
    finish_repair_report(coverage_repair, objectives, [lookup[node] for node in final_nodes])
    report["coverage_repair"] = coverage_repair
    if coverage_repair["status"] == "targets_unmet":
        failures.append({"code": "coverage_repair_targets_unmet", "runs": [
            {"seed": run["seed"], "targets": run["coverage_report"]["unmet_targets"]}
            for run in coverage_repair["runs"] if run["coverage_report"]["unmet_targets"]]})
    guard_report = finish_guard_report(guarded_objectives, [lookup[node] for node in final_nodes],
                                       [data["seed"] for data in datasets])
    report["object_selection_guard"] = guard_report
    if guard_report["status"] == "targets_unmet":
        failures.append({"code": "object_selection_guard_targets_unmet", "runs": [
            {"seed": run["seed"], "targets": run["unmet_targets"]}
            for run in guard_report["runs"] if run["unmet_targets"]]})
    validation = validate_coverage(scene, route, final_nodes, objects, config, banned,
                                   prepared_surface=getattr(surface, "prepared", None))
    report["independent_validation"] = validation
    if validation["status"] == "failed":
        failures.append({"code": "independent_coverage_targets_unmet", "runs": [
            {"seed": run["seed"], "targets": run["coverage_report"]["unmet_targets"]}
            for run in validation["runs"] if run["coverage_report"]["unmet_targets"]]})
    elif validation["status"] == "disabled":
        warnings.append({"code": "independent_coverage_validation_disabled",
                         "message": "Coverage targets were evaluated only on selection samples"})
    report["selection"]["feasibility_repair_succeeded"] = bool(repair_trigger) and not failures and not report["unmet_targets"]
    unobservable = []
    for rid in objective.region_ids:
        mask = objective.regions == rid
        if not mask.any():
            unobservable.append({"region_id": rid, "reason": "Declared region has no input mesh surface; coverage is unassessed"})
        area = float(surface.weights[mask & ~objective.observable].sum())
        if area:
            unobservable.append({"region_id": rid, "unobservable_area_m2": area, "reason": "No eligible candidate sees sampled patches"})
    for i, obj in enumerate(objects):
        if not objective.possible_objects[i]:
            unobservable.append({"instance_id": obj["instance_id"], "must_see": obj["must_see"],
                                 "reason": "No eligible candidate reaches object quality threshold"})
    viewpoints = []
    for node in final_nodes:
        q = quality[lookup[node]]
        content = node in content_nodes
        selection = history.get(lookup[node], {})
        record = feedback.get(route.viewpoint_id(node), {})
        succeeded = record.get("render_status") == "succeeded" and node not in banned
        viewpoints.append({"viewpoint_id": route.viewpoint_id(node), "candidate_id": route.candidate_id(node),
                           "route_node_id": route.node_id(node), "component_id": route.component_id(node),
                           "reference_from_camera": route.camera_pose(node).tolist(),
                           "roles": ["content" if content else "transition"], "score_basis": "geometry",
                           "selection_step": selection.get("selection_step"),
                           "marginal_gain_at_selection": selection.get("marginal_gain_at_selection"),
                           "marginal_image_count_at_selection": selection.get("marginal_image_count_at_selection"),
                           "selection_method": selection.get("selection_method"),
                           "replaced_viewpoint_id": route.viewpoint_id(candidates[selection["replaced_index"]]) if "replaced_index" in selection else None,
                           "reason": selection.get("selection_method", "graph_path_max_link_spacing"),
                           "object_observations": [{"instance_id": obj["instance_id"], "class_name": obj["class_name"],
                                                    "importance": obj["importance"], "importance_source": obj["importance_source"],
                                                    "importance_reason": obj["importance_reason"],
                                                    "geometry_quality": float(q[i]), "image_check": None}
                                                   for i, obj in enumerate(objects)],
                           "render_status": "succeeded" if succeeded else "pending",
                           "image_ref": record.get("image_path") if succeeded else None})
    warnings.append({"code": "geometry_only_prototype",
                     "message": "No image synthesis, observed-camera support constraint, or independent image-quality validation"})
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image", "trimesh", "shapely")}
    route_digest = hashlib.sha256(json.dumps(route.export(), sort_keys=True).encode()).hexdigest()[:12]
    plan = {"schema_name": "spacewatch3d.virtual_tour_plan", "schema_version": "0.1.0",
            "scene_id": scene.scene_id, "capture_id": scene.capture_id, "synthetic": scene.synthetic,
            "reference_frame_id": scene.reference_frame_id, "units": "m", "up_axis": "+Z",
            "route_id": f"route_{route_digest}", "route_version": 1,
            "planning_status": "incomplete" if failures else "complete",
            "quality_status": "partial" if any(v["render_status"] == "succeeded" for v in viewpoints) else "unverified",
            "publication_ready": False, "route_graph": route.export(), "viewpoints": viewpoints,
            "candidate_report": {**route.candidate_sampling,
                                 "candidates": [{"candidate_id": route.candidate_id(n), "route_node_id": route.node_id(n),
                                                 "reasons": route.candidate_reasons[n]} for n in route.candidates]},
            "object_importances": objects,
            "tour_visits": [{"visit_id": f"visit_{i:05d}", "viewpoint_id": route.viewpoint_id(n),
                             "component_id": route.component_id(n)} for i, n in enumerate(visits)],
            "tour_links": links, "coverage_report": report, "unobservable_items": unobservable,
            "failures": failures, "warnings": warnings,
            "excluded_viewpoints": [{"viewpoint_id": route.viewpoint_id(n), "reason": "renderer_failure_or_quality_rejection"} for n in sorted(banned)],
            "provenance": {**scene.provenance, "config": asdict(config), "implementation": "task5b-cpu-prototype-0.13.0",
                           "dependency_versions": versions, "importance_cache": importance_cache or {},
                           "feedback_manifest": str(Path(feedback_path).resolve()) if feedback_path else None,
                           "coordinate_policy": "externally_normalized_metric_z_up",
                           "canonical_route_reuse": scene.provenance.get("canonical_route_reuse", "not_requested")}}
    if reference:
        finish_reuse(plan, reference, reuse_report)
        if reuse_report["unavailable_count"]:
            warnings.append({"code": "reference_viewpoints_unavailable",
                             "message": "Some reference views are unavailable; current coverage is evaluated independently"})
    requests = {"schema_name": "spacewatch3d.virtual_tour_render_requests", "schema_version": "0.1.0",
                "scene_id": scene.scene_id,
                "geometry_sha256": scene.provenance["geometry_sha256"], "route_id": plan["route_id"],
                "route_version": plan["route_version"],
                "requests": [{"viewpoint_id": v["viewpoint_id"], "capture_id": scene.capture_id,
                              "reference_frame_id": scene.reference_frame_id,
                              "reference_from_camera": v["reference_from_camera"], "projection": "equirectangular",
                              "width": config.width, "height": config.height, "media_origin": "novel_view",
                              "render_required": v["render_status"] == "pending",
                              "required_object_ids": [obj["instance_id"] for i, obj in enumerate(objects)
                                                      if obj["must_see"] and v["object_observations"][i]["geometry_quality"] >= config.object_quality_threshold]}
                             for v in viewpoints]}
    if reference:
        transform = scene.provenance["canonical_route_reuse"]
        requests["coordinate_transform"] = {key: transform[key] for key in
                                            ("current_frame_id", "reference_frame_id", "reference_from_current")}
    validate_outputs(plan, requests)
    return plan, requests


def generate(scene_path, output_dir, config=None, **kwargs):
    """Plan, validate and write to a fresh directory. Never overwrite a prior run."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise InputError(f"Output already exists; choose a fresh directory: {output}")
    plan, requests = create_plan(scene_path, config, **kwargs)
    # An exclusive mkdir prevents a concurrent run from being overwritten.
    output.mkdir(parents=True, exist_ok=False)
    for viewpoint in plan["viewpoints"]:
        if viewpoint["image_ref"] is not None:
            viewpoint["image_ref"] = os.path.relpath(viewpoint["image_ref"], output)
    for name, data in (("tour_plan.json", plan), ("render_requests.json", requests)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plan["planning_status"] == "complete":
        (output / "reference_route.json").write_text(
            json.dumps(export_reference(plan), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scores = {obj["instance_id"]: {"score": obj["importance"], "reason": obj["importance_reason"],
                                  "source": obj["importance_source"]} for obj in plan["object_importances"]}
    (output / "importance_scores.json").write_text(json.dumps({"scene_id": plan["scene_id"], "capture_id": plan["capture_id"],
                                                              "scores": scores}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output / "tour_plan.json"
