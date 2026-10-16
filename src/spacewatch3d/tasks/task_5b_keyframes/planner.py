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
from .candidates import refine_visibility_candidates
from .loader import load_scene
from .models import Config, InputError, read_json
from .routing import build_route, connect_tour, shortest
from .selection import Objective, select_keyframes
from .validation import validate_outputs
from .visibility import evaluate_positions, sample_surface


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
        expected = np.eye(4)
        expected[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        expected[:3, 3] = route.positions[node]
        if pose.shape != (4, 4) or not np.allclose(pose, expected, atol=1e-8):
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


def create_plan(scene_path, config=None, importance_cache=None, importance_provider=None, feedback_path=None):
    """Return (plan, requests); only an explicitly injected provider may call an LLM."""
    config = Config.from_dict(asdict(config) if isinstance(config, Config) else config)
    scene = load_scene(Path(scene_path), config)
    config = replace(config, camera_height_m=scene.camera_height_m)
    objects, warnings = resolve_importance(scene.objects, config, importance_cache, importance_provider)
    if scene.provenance["camera_height"]["source"] == "fallback_unverified":
        warnings.append({"code": "camera_height_fallback", "message": "No measured camera height supplied; using unverified 1.5m above floor"})
    route = build_route(scene, config)
    feedback, banned, feedback_warnings = read_feedback(feedback_path, scene, route)
    warnings.extend(feedback_warnings)
    surface = sample_surface(scene, config)
    all_visible, all_quality = evaluate_positions(route.positions[route.candidates], surface, objects, config)
    refined = refine_visibility_candidates(route, all_visible, all_quality, surface.weights, config)
    if refined:
        refined_visible, refined_quality = evaluate_positions(route.positions[refined], surface, objects, config)
        all_visible = np.concatenate([all_visible, refined_visible])
        all_quality = np.concatenate([all_quality, refined_quality])
    allowed_indices = [i for i, node in enumerate(route.candidates) if node not in banned]
    candidates = [route.candidates[i] for i in allowed_indices]
    visible, quality = all_visible[allowed_indices], all_quality[allowed_indices]
    objective = Objective(visible, quality, surface.weights, surface.regions, objects, config)
    lookup = {node: i for i, node in enumerate(candidates)}

    @lru_cache(maxsize=32)
    def paths_from(node):
        return shortest(route.adjacency, node)

    @lru_cache(maxsize=256)
    def evaluate_routed_set(key):
        nodes = [candidates[i] for i in key]
        _, tour_links, transition_nodes, connection_failures = connect_tour(route, nodes, config, banned, scene.start, paths_from)
        final = set(nodes) | set(transition_nodes)
        utility, selection_report = objective.measure([lookup[node] for node in sorted(final)])
        selection_report.update(selection_image_count=len(final),
                                selection_tour_length_m=sum(link["length_m"] for link in tour_links),
                                selection_connection_failures=[f["code"] for f in connection_failures])
        return utility, selection_report

    evaluator = (lambda selected: evaluate_routed_set(tuple(sorted(selected)))) if config.route_aware_selection else None
    chosen, history, _ = select_keyframes(objective, evaluator)
    content_report = objective.measure(chosen)[1]
    selected_nodes = [candidates[i] for i in chosen]
    visits, links, transitions, failures = connect_tour(route, selected_nodes, config, banned, scene.start, paths_from)
    final_nodes = sorted(set(selected_nodes) | set(transitions))
    # Every transition is an already-scored candidate. Never redefine coverage
    # denominators to just the selected images or hide unobservable geometry.
    report = objective.measure([lookup[node] for node in final_nodes])[1]
    report["content_only"] = content_report
    report["selection"] = {"strategy": "routed_gain_per_added_image_with_one_swap" if evaluator else "marginal_gain",
                           "tour_length_m": sum(link["length_m"] for link in links),
                           "local_swap_applied": any(history[i]["selection_method"] == "route_cost_local_swap" for i in chosen)}
    report["candidate_count_before_feedback"] = len(route.candidates)
    report["candidate_count_after_feedback"] = len(candidates)
    report["observable_area_before_feedback_m2"] = float(surface.weights[np.any(all_visible, axis=0)].sum()) if len(all_visible) else 0
    report["unique_image_count"] = len(final_nodes)
    report["content_count"] = len(selected_nodes)
    report["transition_count"] = len(transitions)
    if not final_nodes:
        failures.append({"code": "no_selected_viewpoints"})
    if report["unmet_targets"]:
        failures.append({"code": "targets_unmet", "targets": report["unmet_targets"]})
    unobservable = []
    for rid in objective.region_ids:
        mask = objective.regions == rid
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
        content = node in selected_nodes
        selection = history[lookup[node]] if content else {}
        record = feedback.get(route.viewpoint_id(node), {})
        succeeded = record.get("render_status") == "succeeded" and node not in banned
        viewpoints.append({"viewpoint_id": route.viewpoint_id(node), "candidate_id": route.candidate_id(node),
                           "route_node_id": route.node_id(node), "component_id": route.component_id(node),
                           "reference_from_camera": [[1, 0, 0, float(route.positions[node, 0])],
                                                     [0, 0, 1, float(route.positions[node, 1])],
                                                     [0, -1, 0, float(route.positions[node, 2])], [0, 0, 0, 1]],
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
            "provenance": {**scene.provenance, "config": asdict(config), "implementation": "task5b-cpu-prototype-0.2.0",
                           "dependency_versions": versions, "importance_cache": importance_cache or {},
                           "feedback_manifest": str(Path(feedback_path).resolve()) if feedback_path else None,
                           "coordinate_policy": "externally_normalized_metric_z_up",
                           "canonical_route_reuse": "not_implemented"}}
    requests = {"schema_name": "spacewatch3d.virtual_tour_render_requests", "schema_version": "0.1.0",
                "scene_id": scene.scene_id,
                "geometry_sha256": scene.provenance["geometry_sha256"], "route_id": plan["route_id"],
                "requests": [{"viewpoint_id": v["viewpoint_id"], "capture_id": scene.capture_id,
                              "reference_frame_id": scene.reference_frame_id,
                              "reference_from_camera": v["reference_from_camera"], "projection": "equirectangular",
                              "width": config.width, "height": config.height, "media_origin": "novel_view",
                              "render_required": v["render_status"] == "pending",
                              "required_object_ids": [obj["instance_id"] for i, obj in enumerate(objects)
                                                      if obj["must_see"] and v["object_observations"][i]["geometry_quality"] >= config.object_quality_threshold]}
                             for v in viewpoints]}
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
    scores = {obj["instance_id"]: {"score": obj["importance"], "reason": obj["importance_reason"],
                                  "source": obj["importance_source"]} for obj in plan["object_importances"]}
    (output / "importance_scores.json").write_text(json.dumps({"scene_id": plan["scene_id"], "capture_id": plan["capture_id"],
                                                              "scores": scores}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output / "tour_plan.json"
