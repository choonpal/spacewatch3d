"""Reproducible synthetic benchmark; never changes or tunes the production planner.

Two distinct panels: installed production planner end-to-end, and controlled ranking diagnostics
on one common dense graph-node pool. The latter is NOT a reproduction of production's
candidate audits, multiple optimization draws, object guard, pruning or swaps.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

import numpy as np
import shapely
from shapely.geometry import Polygon

from ..importance import resolve_importance
from ..loader import box_triangles, load_scene
from ..models import Config
from ..planner import create_plan
from ..routing import build_route, connect_tour, shortest
from ..selection import Objective
from ..visibility import evaluate_positions, sample_surface


PLANNING_SEEDS = (1000, 1100)
HOLDOUT_SEEDS = (5000, 5001, 5002)
BUDGETS = (2, 4, 8)
METHODS = ("uniform_geodesic", "space_ranking", "equal_object_ranking", "weighted_object_ranking")


def protocol():
    return {
        "benchmark_version": "task5b-synthetic-benchmark-1.0",
        "locked_before_execution": True,
        "planning_seeds": PLANNING_SEEDS,
        "diagnostic_selection_seeds": [s + 100 for s in PLANNING_SEEDS],
        "external_holdout_seeds": HOLDOUT_SEEDS,
        "holdout_samples": 1800, "holdout_min_object_samples": 48,
        "holdout_min_region_samples": 24,
        "image_budgets": BUDGETS, "target_search_image_cap": 16,
        "common_targets": {"balanced_space": .9, "each_observable_region": .6,
                           "importance_weighted_observed_objects": .9, "must_see": True},
        "quality_threshold": .35, "quality_sensitivity_thresholds": [.25, .35, .45],
        "fixed_plans_for_sensitivity": True,
        "native_panel": "Installed production create_plan, version in saved plan provenance, default config except seed; includes internal repair and holdout",
        "controlled_panel": "Common dense safe route-node pool, one 1200-sample selection draw, exact routed unique-image cost; no production guard/prune/swap",
        "uniform_definition": "Geometry-only farthest-nearest geodesic node insertion, first node nearest supplied start; spatial-uniform heuristic, NOT temporal video sampling",
        "space_ranking_definition": "Space-only utility; common hard object goals retained in target-search fallback, not a semantic-input ablation",
        "equal_definition": "Space+object utility with all object ranking importances equal; hard goals and external scoring keep original manual importances",
        "weighted_definition": "Space+object utility with scene's manual importances, NOT LLM-assessed scores",
        "fixed_budget_policy": "At most K unique content+transition images; reject disconnected or over-budget insertions, stop if no positive gain (uniform uses distance). Actual counts reported, unused capacity not padded",
        "target_search_policy": "Stop on common selection-draw goals; otherwise gain-per-added-image, zero-gain deficit fallback; cap16; no holdout-driven reselection",
        "shared_denominator": "Every safe route graph node, same patches across all methods within each scene/seed/draw; unobservable input area and objects also reported",
        "redundancy_metric": "1 - area(union visible patches)/sum(single-image visible areas), 0 for no visible area; descriptive overlap, not pure waste",
        "runtime_policy": "Native wall time separately; diagnostic wall time includes cache warming effects; peak RSS is entire benchmark process, NOT per-method memory",
        "scene_selection": "Six geometric scenarios and two known-impossible controls, fixed before seeing measurements; no exclusion for failure",
        "limits": ["Synthetic geometry only", "No RGB quality, reconstruction noise or user study",
                   "No image bytes measured", "No LLM called", "No population inference or significance tests",
                   "Thresholds/weights are prototype settings, not literature-standard ground truth",
                   "Repeated seeds are robustness repeats, not independent real buildings"],
        "literature": [
            {"title": "Learning View Selection for 3D Scenes (Sun et al., CVPR 2021)",
             "url": "https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Learning_View_Selection_for_3D_Scenes_CVPR_2021_paper.html",
             "adaptation": "Coverage at fixed camera count and count/time to coverage goals; our area-weighted 360 evaluation is NOT literal reproduction"},
            {"title": "Key views for visualizing large spaces (Cai and Zheng, 2009)",
             "url": "https://doi.org/10.1016/j.jvcir.2009.04.005",
             "adaptation": "Spatial/semantic visibility, redundancy and navigation continuity"},
            {"title": "Effective Replays and Summarization of Virtual Experiences (Ponto et al., 2012)",
             "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3354691/",
             "adaptation": "Interest-object representation; user-understanding evaluation remains unmeasured"}],
    }


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def mesh_box(identifier, label, low, high, importance=.5, must_see=False):
    return {"instance_id": identifier, "class_name": label,
            "triangles": box_triangles([low, high]).tolist(), "solid": True,
            "importance": importance, "must_see": must_see}


def structure_box(low, high):
    return {"triangles": box_triangles([low, high]).tolist(), "solid": True}


def make_scene(name, polygon, objects=(), extra=(), regions=(), start=(1, 1), description="", impossible=False):
    walls = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        a0, b0, a3, b3 = [*a, 0], [*b, 0], [*a, 2.8], [*b, 2.8]
        walls.append({"triangles": [[a0, b0, b3], [a0, b3, a3]], "solid": False})
    ceiling = []
    for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(Polygon(polygon))):
        ceiling.append([[float(x), float(y), 2.8] for x, y in list(tri.exterior.coords)[:3]])
    return {"schema_name": "spacewatch3d.planning_scene", "schema_version": "0.1.0",
            "scene_id": name, "capture_id": "synthetic_capture", "synthetic": True,
            "coordinates": {"frame_id": "benchmark:" + name, "units": "m", "up_axis": "+Z", "gravity_aligned": True},
            "geometry_complete": True, "floor": {"height_m": 0, "polygon": polygon, "confirmed": True},
            "camera": {"height_above_floor_m": 1.5}, "start_xy_m": list(start),
            "objects": list(objects), "structures": walls + [{"triangles": ceiling, "solid": False}] + list(extra),
            "regions": [{"region_id": rid, "polygon": poly} for rid, poly in regions],
            "benchmark_metadata": {"description": description, "expected_impossible": impossible}}


def synthetic_scenes():
    box = mesh_box
    return [
        make_scene("open_room", rectangle(0, 0, 6, 4),
                   [box("chair", "chair", [.5, .6, 0], [1, 1.1, .9], .3),
                    box("cabinet", "cabinet", [4.8, 2.7, 0], [5.5, 3.6, 1.7], .7)],
                   description="Open rectangular room; ceiling and floor included"),
        make_scene("cluttered_room", rectangle(0, 0, 7, 6),
                   [box("shelf1", "shelf", [1.8, 1.2, 0], [2.5, 3.8, 2.3], .5),
                    box("shelf2", "shelf", [4.2, 2.2, 0], [4.9, 4.8, 2.3], .5),
                    box("table", "table", [2.8, 4.7, 0], [3.8, 5.4, .8], .25),
                    box("panel", "control_panel", [5.9, .2, .9], [6.4, .4, 1.6], .95, True)],
                   regions=[("west", rectangle(0, 0, 3.5, 6)), ("east", rectangle(3.5, 0, 7, 6))],
                   description="Tall shelves occlude views and force detours"),
        make_scene("two_rooms_doorway", rectangle(0, 0, 7, 5),
                   [box("cabinet", "cabinet", [.3, 2, 0], [1, 3.2, 1.8], .6),
                    box("desk", "desk", [4.4, 2, 0], [5.6, 3, .8], .4),
                    box("safety", "fire_extinguisher", [5.8, .25, 0], [6.15, .6, .65], .95, True)],
                   [structure_box([3.4, 0, 0], [3.5, 3.5, 2.8])],
                   [("left", rectangle(0, 0, 3.5, 5)), ("right", rectangle(3.5, 0, 7, 5))],
                   description="Two zones joined by a 1.5m opening"),
        make_scene("l_corridor", [[0, 0], [7, 0], [7, 2], [2, 2], [2, 7], [0, 7]],
                   [box("panel", "control_panel", [5.9, .1, 1], [6.4, .25, 1.7], .9),
                    box("plant", "plant", [.1, 5.9, 0], [.4, 6.3, 1], .2)],
                   regions=[("horizontal", rectangle(0, 0, 7, 2)), ("vertical", rectangle(0, 2, 2, 7))],
                   description="Ninety-degree blind corner and long thin zones"),
        make_scene("t_junction", [[0, 0], [2, 0], [2, 4], [7, 4], [7, 6], [-5, 6], [-5, 4], [0, 4]],
                   [box("safety", "fire_extinguisher", [-4.7, 4.15, .2], [-4.4, 4.45, .8], .95, True),
                    box("panel", "control_panel", [6.2, 5.6, .9], [6.6, 5.8, 1.6], .85)],
                   regions=[("stem", rectangle(0, 0, 2, 4)), ("left", rectangle(-5, 4, 2, 6)),
                            ("right", rectangle(2, 4, 7, 6))],
                   description="Three branches; important objects at opposite ends"),
        make_scene("overhead_and_small", rectangle(0, 0, 6, 5),
                   [box("table", "table", [2.2, 1.8, 0], [3.8, 3, .8], .25),
                    box("vent", "vent", [2.4, 2, 2.55], [3.4, 2.8, 2.68], .85),
                    box("small_sensor", "sensor", [5.6, .3, 1.15], [5.68, .38, 1.27], 1., True)],
                   description="Vent above a table plus an 8cm-wide important sensor; genuine 3D visibility"),
        make_scene("disconnected_control", rectangle(0, 0, 7, 5),
                   [box("left_panel", "control_panel", [.3, .2, 1], [.8, .4, 1.7], .8),
                    box("right_panel", "control_panel", [6.2, .2, 1], [6.7, .4, 1.7], .8)],
                   [structure_box([3.4, 0, 0], [3.6, 5, 2.8])],
                   [("left", rectangle(0, 0, 3.5, 5)), ("right", rectangle(3.5, 0, 7, 5))],
                   description="Full-height wall disconnects zones: no single connected tour can meet both", impossible=True),
        make_scene("sealed_object_control", rectangle(0, 0, 6, 5),
                   [box("hidden_safety", "fire_extinguisher", [2.7, 2.2, .3], [3, 2.5, .9], 1., True)],
                   [structure_box([2.2, 1.7, 0], [3.5, 3, 2.2])],
                   description="Must-see object enclosed in an opaque solid cabinet: observation must fail", impossible=True),
    ]


def connected_result(route, selected, config, start, paths_from):
    _, links, transitions, failures = connect_tour(route, list(selected), config, set(), start, paths_from)
    final = tuple(sorted(set(selected) | set(transitions)))
    return {"nodes": final, "selected_nodes": tuple(selected),
            "transition_count": len(set(transitions) - set(selected)),
            "image_count": len(final), "tour_length_m": sum(link["length_m"] for link in links),
            "max_link_m": max((link["length_m"] for link in links), default=0.),
            "connection_failures": [f["code"] for f in failures], "links": links}


def budget_admissible(result, budget):
    return result["image_count"] <= budget and not result["connection_failures"]


def overlap_fraction(visible, weights, selected):
    if not selected:
        return 0.
    union = float(weights[np.any(visible[selected], axis=0)].sum())
    individual = float((visible[selected] * weights).sum())
    return 1. - union / individual if individual else 0.


def scored_report(objective, selected):
    report = objective.measure(list(selected))[1]
    report["overlap_fraction"] = overlap_fraction(objective.visible, objective.weights, list(selected))
    report["minimum_region_coverage"] = min((r["coverage_observable"] for r in report["regions"]
                                             if r["coverage_observable"] is not None), default=None)
    report["unobservable_object_ids"] = [o["instance_id"] for o, possible in zip(objective.objects, objective.possible_objects) if not possible]
    report["object_observed_count"] = sum(report["object_best_quality"][o["instance_id"]] >= objective.config.object_quality_threshold
                                          for o in objective.objects)
    report["object_count"] = len(objective.objects)
    report["all_common_targets_met"] = not report["unmet_targets"]
    return report


def choose_diagnostic(method, objective, ranking, route, config, start, budget, target_search=False):
    """Benchmark-only greedy policy; holdout data never enters this function."""
    @lru_cache(maxsize=None)
    def paths_from(node):
        return shortest(route.adjacency, node)

    @lru_cache(maxsize=2048)
    def routed(key):
        return connected_result(route, key, config, start, paths_from)

    selected = []
    start_node = min(range(len(route.positions)), key=lambda n: (float(np.linalg.norm(route.positions[n, :2] - start)), n))
    current = routed(())
    while len(selected) < budget:
        current_utility = ranking.measure(list(current["nodes"]))[0]
        common = objective.measure(list(current["nodes"]))[1]
        if target_search and not common["unmet_targets"]:
            break
        deficit = objective.constraint_deficit(common)
        choices = []
        for node in route.candidates:
            if node in current["nodes"]:
                continue
            trial = routed(tuple(sorted(selected + [node])))
            if not budget_admissible(trial, budget):
                continue
            added = max(1, trial["image_count"] - current["image_count"])
            gain = ranking.measure(list(trial["nodes"]))[0] - current_utility
            if method == "uniform_geodesic":
                distance = min((paths_from(n)[0].get(node, math.inf) for n in current["nodes"]), default=0.)
                rank = (distance if selected else -paths_from(start_node)[0].get(node, math.inf), -node)
            else:
                rank = (gain / added, gain, -trial["image_count"], -trial["tour_length_m"], -node)
            progress = 0.
            if target_search:
                progress = deficit - objective.constraint_deficit(objective.measure(list(trial["nodes"]))[1])
            choices.append((rank, node, gain, progress / added, trial))
        if not choices:
            break
        best = max(choices, key=lambda c: c[0])
        if method != "uniform_geodesic" and best[2] <= 1e-12:
            useful = [c for c in choices if c[3] > 1e-12] if target_search else []
            if not useful:
                break
            best = max(useful, key=lambda c: (c[3], c[2], -c[4]["image_count"], -c[1]))
        selected.append(best[1])
        current = best[4]
    common = objective.measure(list(current["nodes"]))[1]
    return {**current, "selected_nodes": tuple(selected), "selection_targets_met": not common["unmet_targets"],
            "selection_unmet_targets": common["unmet_targets"], "budget": budget}


def json_save(path, data):
    # Generated experiment artifacts only, not a source-file editing mechanism.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))}


def summarize_draws(draws):
    keys = ("coverage_observable", "coverage_total", "balanced_space_coverage",
            "minimum_region_coverage", "object_importance_coverage_observable",
            "object_importance_coverage_total", "weighted_object_quality", "overlap_fraction")
    result = {}
    for key in keys:
        values = [d[key] for d in draws if d[key] is not None]
        result[key] = {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))} if values else None
    result["external_pass_all_draws"] = all(d["all_common_targets_met"] for d in draws)
    result["unmet_targets_union"] = sorted({x for d in draws for x in d["unmet_targets"]})
    return result


def evaluate_fixed(objectives, nodes):
    draws = [scored_report(obj, nodes) for obj in objectives]
    sensitivity = {}
    for threshold in (.25, .35, .45):
        records = []
        for obj in objectives:
            changed = Objective(obj.visible, obj.quality, obj.weights, obj.regions, obj.objects,
                                replace(obj.config, object_quality_threshold=threshold), obj.declared_regions)
            records.append(scored_report(changed, nodes))
        sensitivity[str(threshold)] = summarize_draws(records)
    return {"draws": draws, "summary": summarize_draws(draws), "quality_threshold_sensitivity": sensitivity}


def run_context(scene_path, scene_data, seed, output):
    config = replace(Config(), seed=seed)
    scene = load_scene(scene_path, config)
    config = replace(config, camera_height_m=scene.camera_height_m)
    objects, _ = resolve_importance(scene.objects, config)
    route = build_route(scene, config)
    # Diagnostics intentionally expose every safe graph node equally to every
    # strategy and to the connector. This does not mutate production's route.
    route.candidates = list(range(len(route.positions)))
    context_dir = output / "cases" / scene.scene_id / str(seed)
    context_dir.mkdir(parents=True, exist_ok=False)
    print(f"START {scene.scene_id} seed={seed} nodes={len(route.positions)}", flush=True)
    started = time.perf_counter()
    plan, requests = create_plan(scene_path, replace(Config(), seed=seed))
    native_seconds = time.perf_counter() - started
    json_save(context_dir / "native_plan.json", plan)
    json_save(context_dir / "native_requests.json", requests)
    lookup = {route.node_id(i): i for i in range(len(route.positions))}
    native_nodes = [lookup[v["route_node_id"]] for v in plan["viewpoints"]]
    print(f"NATIVE {scene.scene_id}/{seed} {plan['planning_status']} images={len(native_nodes)} {native_seconds:.2f}s", flush=True)
    objectives = []
    for holdout_seed in HOLDOUT_SEEDS:
        eval_config = replace(config, seed=holdout_seed, surface_samples=1800, min_object_samples=48, min_region_samples=24)
        surface = sample_surface(scene, eval_config)
        visible, quality = evaluate_positions(route.positions, surface, objects, eval_config)
        objectives.append(Objective(visible, quality, surface.weights, surface.regions, objects, eval_config,
                                    [rid for rid, _ in scene.regions]))
    native_eval = evaluate_fixed(objectives, native_nodes)
    native = {"scene_id": scene.scene_id, "planning_seed": seed,
              "expected_impossible": scene_data["benchmark_metadata"]["expected_impossible"],
              "planning_status": plan["planning_status"], "publication_ready": plan["publication_ready"],
              "image_count": len(native_nodes), "content_count": plan["coverage_report"]["content_count"],
              "transition_count": plan["coverage_report"]["transition_count"],
              "tour_length_m": plan["coverage_report"]["selection"]["tour_length_m"],
              "max_link_m": max((link["length_m"] for link in plan["tour_links"]), default=0.),
              "connection_failures": [f["code"] for f in plan["failures"] if f["code"] in ("disconnected_tour", "transition_gap")],
              "failure_codes": [f["code"] for f in plan["failures"]],
              "runtime_seconds": native_seconds, "selected_nodes": native_nodes, **native_eval}
    # Only this draw drives the controlled ranking experiment. Evaluations above
    # cannot influence choices: choose_diagnostic receives none of those arrays.
    selection_config = replace(config, seed=seed + 100, surface_samples=1200, min_object_samples=24, min_region_samples=24)
    surface = sample_surface(scene, selection_config)
    visible, quality = evaluate_positions(route.positions, surface, objects, selection_config)
    common = Objective(visible, quality, surface.weights, surface.regions, objects, selection_config,
                       [rid for rid, _ in scene.regions])
    equal = [{**obj, "importance": 1.} for obj in objects]
    rankings = {
        "uniform_geodesic": common,
        "space_ranking": Objective(visible, quality, surface.weights, surface.regions, objects,
                                    replace(selection_config, space_weight=1., object_weight=0.), common.declared_regions),
        "equal_object_ranking": Objective(visible, quality, surface.weights, surface.regions, equal, selection_config, common.declared_regions),
        "weighted_object_ranking": common,
    }
    diagnostics = []
    for method in METHODS:
        for mode, budget in [("fixed_budget", b) for b in BUDGETS] + [("common_target_search", 16)]:
            started = time.perf_counter()
            result = choose_diagnostic(method, common, rankings[method], route, config, scene.start,
                                       budget, target_search=mode == "common_target_search")
            elapsed = time.perf_counter() - started
            diagnostic = {"scene_id": scene.scene_id, "planning_seed": seed,
                          "expected_impossible": native["expected_impossible"], "method": method,
                          "mode": mode, "runtime_seconds": elapsed, **result,
                          **evaluate_fixed(objectives, list(result["nodes"]))}
            diagnostics.append(diagnostic)
        print(f"DIAGNOSTIC {scene.scene_id}/{seed} {method} done", flush=True)
    context = {"scene_id": scene.scene_id, "planning_seed": seed,
               "route_node_count": len(route.positions), "floor_area_m2": scene.floor.area,
               "triangle_count": sum(len(g.triangles) for g in scene.geometries),
               "object_count": len(objects), "native": native, "diagnostics": diagnostics}
    json_save(context_dir / "metrics.json", context)
    return context


def percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"


def scene_svg(data, plan):
    polygon = data["floor"]["polygon"]
    xy = np.asarray(polygon)
    low, high = xy.min(axis=0), xy.max(axis=0)
    scale = 260 / max(high - low)
    def point(p):
        return f"{20 + (p[0] - low[0]) * scale:.1f},{20 + (high[1] - p[1]) * scale:.1f}"
    width, height = 40 + (high[0] - low[0]) * scale, 40 + (high[1] - low[1]) * scale
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(data["scene_id"])}"><polygon points="{" ".join(point(p) for p in polygon)}" fill="#f5f7fa" stroke="#526175"/>']
    for structure in data["structures"]:
        tri = np.asarray(structure["triangles"])
        if structure.get("solid"):
            a, b = tri[:, :, :2].min(axis=(0, 1)), tri[:, :, :2].max(axis=(0, 1))
            corners = rectangle(*a, *b)
            parts.append(f'<polygon points="{" ".join(point(p) for p in corners)}" fill="#8995a7"/>')
    for obj in data["objects"]:
        tri = np.asarray(obj["triangles"])
        a, b = tri[:, :, :2].min(axis=(0, 1)), tri[:, :, :2].max(axis=(0, 1))
        fill = "#dc8445" if tri[:, :, 2].min() > 1.9 else "#b4bfca"
        parts.append(f'<polygon points="{" ".join(point(p) for p in rectangle(*a, *b))}" fill="{fill}" stroke="#566070"><title>{html.escape(obj["instance_id"])}</title></polygon>')
    for link in plan["tour_links"]:
        parts.append(f'<polyline points="{" ".join(point(p) for p in link["polyline_m"])}" fill="none" stroke="#519c81" stroke-width="1.8"/>')
    for view in plan["viewpoints"]:
        pose = view["reference_from_camera"]
        px, py = point([pose[0][3], pose[1][3]]).split(",")
        color = "#125cc5" if "content" in view["roles"] else "#bf6517"
        parts.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="{color}"/>')
    return "".join(parts) + "</svg>"


def aggregate(contexts):
    positives = [c["native"] for c in contexts if not c["native"]["expected_impossible"]]
    controls = [c["native"] for c in contexts if c["native"]["expected_impossible"]]
    comparisons = []
    for budget in BUDGETS:
        for method in METHODS:
            rows = [d for c in contexts for d in c["diagnostics"] if not d["expected_impossible"]
                    and d["mode"] == "fixed_budget" and d["budget"] == budget and d["method"] == method]
            comparisons.append({"budget": budget, "method": method, "contexts": len(rows),
                                "mean_images": float(np.mean([r["image_count"] for r in rows])),
                                "mean_balanced_space": float(np.mean([r["summary"]["balanced_space_coverage"]["mean"] for r in rows])),
                                "mean_weighted_object_observed": float(np.mean([r["summary"]["object_importance_coverage_observable"]["mean"] for r in rows
                                    if r["summary"]["object_importance_coverage_observable"] is not None])),
                                "pass_all_draws_count": sum(r["summary"]["external_pass_all_draws"] for r in rows)})
    target_rows = []
    for method in METHODS:
        rows = [d for c in contexts for d in c["diagnostics"] if not d["expected_impossible"]
                and d["mode"] == "common_target_search" and d["method"] == method]
        passed = [r for r in rows if r["summary"]["external_pass_all_draws"]]
        target_rows.append({"method": method, "contexts": len(rows), "external_pass_count": len(passed),
                            "selection_pass_count": sum(r["selection_targets_met"] for r in rows),
                            "mean_image_count_all_attempts": float(np.mean([r["image_count"] for r in rows])),
                            "mean_image_count_external_passes_only": float(np.mean([r["image_count"] for r in passed])) if passed else None,
                            "warning": "Successful-subset counts are not comparable minima; failed cases are retained"})
    return {"native_positive_context_count": len(positives),
            "native_positive_complete_count": sum(r["planning_status"] == "complete" for r in positives),
            "native_positive_external_pass_count": sum(r["summary"]["external_pass_all_draws"] and not r["connection_failures"] for r in positives),
            "native_positive_false_complete_count": sum(r["planning_status"] == "complete" and not r["summary"]["external_pass_all_draws"] for r in positives),
            "native_control_context_count": len(controls),
            "native_control_incomplete_count": sum(r["planning_status"] == "incomplete" for r in controls),
            "native_positive_mean_images": float(np.mean([r["image_count"] for r in positives])),
            "native_positive_mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in positives])),
            "controlled_fixed_budget": comparisons, "controlled_target_search": target_rows}


def render_report(output, report, scenes):
    stats = report["aggregate"]
    panels = []
    for data in scenes:
        rows = [c["native"] for c in report["contexts"] if c["scene_id"] == data["scene_id"]]
        plan = json.loads((output / "cases" / data["scene_id"] / str(PLANNING_SEEDS[0]) / "native_plan.json").read_text())
        body = ""
        for r in rows:
            s = r["summary"]
            body += (f'<tr><td>{r["planning_seed"]}</td><td>{r["planning_status"]}</td><td>{r["image_count"]} ({r["content_count"]}+{r["transition_count"]})</td>'
                     f'<td>{percent(s["balanced_space_coverage"]["min"])}</td><td>{percent(s["coverage_total"]["min"])}</td>'
                     f'<td>{percent(s["object_importance_coverage_observable"]["min"] if s["object_importance_coverage_observable"] else None)}</td>'
                     f'<td>{"PASS" if s["external_pass_all_draws"] and not r["connection_failures"] else "FAIL"}</td><td>{r["runtime_seconds"]:.2f}s</td></tr>')
        failures = "; ".join(sorted({x for r in rows for x in r["summary"]["unmet_targets_union"]} | {x for r in rows for x in r["failure_codes"]})) or "없음"
        panels.append(f'<section><h3>{html.escape(data["scene_id"])} {"(불가능 조건 대조군)" if data["benchmark_metadata"]["expected_impossible"] else ""}</h3><p>{html.escape(data["benchmark_metadata"]["description"])}</p>{scene_svg(data, plan)}<table><tr><th>Seed</th><th>내부 판정</th><th>이미지 (핵심+이동)</th><th>구역균형 커버리지 최솟값</th><th>전체 입력 표면 최솟값</th><th>관측 가능 객체 중요도 최솟값</th><th>외부 평가</th><th>실행시간</th></tr>{body}</table><p>실패 항목: {html.escape(failures)}</p></section>')
    comparison_rows = "".join(f'<tr><td>{r["budget"]}</td><td>{r["method"]}</td><td>{r["mean_images"]:.2f}</td><td>{percent(r["mean_balanced_space"])}</td><td>{percent(r["mean_weighted_object_observed"])}</td><td>{r["pass_all_draws_count"]}/{r["contexts"]}</td></tr>' for r in stats["controlled_fixed_budget"])
    target_rows = "".join(f'<tr><td>{r["method"]}</td><td>{r["selection_pass_count"]}/{r["contexts"]}</td><td>{r["external_pass_count"]}/{r["contexts"]}</td><td>{r["mean_image_count_all_attempts"]:.2f}</td></tr>' for r in stats["controlled_target_search"])
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task5b synthetic benchmark v1</title>
<style>body{{font:15px/1.65 system-ui,sans-serif;color:#203047;max-width:1200px;margin:40px auto;padding:0 24px;background:#f4f6f9}}section{{background:white;padding:22px;margin:18px 0;border:1px solid #d8e0eb;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d9e1ec}}th{{background:#eef3fa}}svg{{max-width:330px;max-height:310px;margin:12px}}code{{overflow-wrap:anywhere}}.warn{{background:#fff4df;padding:16px}}h1,h2,h3{{line-height:1.3}}</style>
<h1>SpaceWatch3D Task5b — 합성 공간 성능 측정</h1>
<p>현재 설치된 선정기 · 버전은 저장 계획의 provenance 참조 · 8개 공간 × 2개 선정 seed · 외부 검증 3개 독립 표본 · LLM 호출/이미지 합성 0회</p>
<section><h2>실제 선정기 결과</h2><p>일반 시나리오 {stats["native_positive_context_count"]}건 중 내부 complete {stats["native_positive_complete_count"]}건, 외부 표본 전체 목표 통과 {stats["native_positive_external_pass_count"]}건. 내부 complete지만 외부 검증 실패 {stats["native_positive_false_complete_count"]}건.</p>
<p>불가능 조건 {stats["native_control_context_count"]}건 중 incomplete 보고 {stats["native_control_incomplete_count"]}건. 일반 시나리오 평균 이미지 {stats["native_positive_mean_images"]:.2f}장, 평균 계획 시간 {stats["native_positive_mean_runtime_seconds"]:.2f}초.</p>
<p class="warn">합성 기하에서의 표본 기반 평가이다. 실제 파노라마의 판독성·합성 품질·웹 이미지 용량·LLM 점수 정확도를 검증한 결과가 아니다. 관측 가능 표면 대비 커버리지와 전체 입력 표면 대비 커버리지를 구분해야 한다.</p></section>
<h2>8개 공간과 실제 선정 위치</h2><p>첫 번째 seed의 평면 투영. 파랑=핵심, 주황=이동, 초록=연결 경로, 주황 면=머리 위 객체. 평가는 높이를 포함한 3D 광선으로 진행한다. 아래 커버리지는 외부 3개 표본 중 최솟값.</p>{"".join(panels)}
<section><h2>동일 예산에서 선정 전략 비교 — 실제 전체 파이프라인과 별도</h2><p>동일 안전 그래프·선정 표본·평가 표본으로 비교했다. 모든 연결 이미지를 포함해 K장 이하이며, 공간과 객체가 이미 포화되면 예산을 다 쓰지 않는다. uniform은 촬영 시간 간격이 아니라 그래프 거리 균등화 휴리스틱이다. space_ranking은 순위 점수만 공간 기준이며 목표 탐색에서는 공통 객체 제약을 유지한다. weighted_object_ranking은 기본 점수식만 적용한 진단용 전략으로, 실제 표본 보완·guard·prune·swap·별도 객체 정밀 평가는 포함하지 않는다.</p><table><tr><th>예산 K</th><th>전략</th><th>실제 평균 장수</th><th>평균 구역균형 커버리지</th><th>평균 중요도 관측률</th><th>외부 전체 목표 통과</th></tr>{comparison_rows}</table></section>
<section><h2>공통 목표까지 선택 — 최대 16장</h2><p>선정 표본으로 중단하고 외부 표본에는 손대지 않는다. 실패를 포함한 평균 장수로, 최적 최소 장수가 아니며 실패 방법이 적게 골랐다고 우수한 것은 아니다.</p><table><tr><th>전략</th><th>선정 표본 통과</th><th>외부 전체 통과</th><th>전체 시도 평균 장수</th></tr>{target_rows}</table></section>
<section><h2>해석 범위와 재현 정보</h2><ul><li>일반 6개 공간과 불가능 대조군 2개를 별도 집계하며 원자료는 모두 보존했다.</li><li>객체의 실제 중요도 정답이 아니라 수동으로 정한 예시 점수에 대한 선정 동작을 측정했다.</li><li>객체 관측 q≥0.35는 기하학적 대리 지표다. 0.25/0.35/0.45 민감도 결과를 metrics.json에 포함했다. 민감도에서는 분모도 달라지므로 total 지표를 함께 봐야 한다.</li><li>동일 표본을 모든 방법이 공유하지만 서로 다른 건물 또는 사람이 정한 정답에 대한 일반화는 검증하지 않았다.</li><li>벽·천장·바닥을 포함한 전체 표면 중 어떤 후보에서도 볼 수 없는 면적을 숨기지 않았다.</li><li>연결 길이는 그래프 경로 기준 최대 2m이다. 사용자 시각적 이동 품질을 보장하는 수치가 아니다.</li><li>프로세스 전체 peak RSS {report["peak_rss_mib"]:.1f}MiB. 방법별 독립 메모리 측정 아님. 실행시간은 이 컴퓨터에서의 기술 통계이며 통계적 우열 주장 없음.</li><li>코드·장면·프로토콜·계획·상세 측정 JSON을 함께 제공한다. Github 변경 및 구현 버전 증가는 없다.</li></ul>
<p>출처: <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Learning_View_Selection_for_3D_Scenes_CVPR_2021_paper.html">Sun et al., CVPR 2021</a>의 고정 카메라 수 커버리지/목표 커버리지까지의 수·시간 측정을 360°·면적 평가로 변형했다. <a href="https://doi.org/10.1016/j.jvcir.2009.04.005">Cai & Zheng, 2009</a>의 공간·의미·중복·연속성 관점을 참고했다. 논문의 수치나 공식 벤치마크 재현 결과가 아니다.</p>
<p>Material Passport: origin=experiment-agent; mode=run; created={report["created_at"]}; verification=UNVERIFIED (단일 실행, 독립 재현 실행 아님); version=exp_result_v1; exit_code=0. 실행 명령: <code>{html.escape(report["command"])}</code></p></section></html>'''
    (output / "report.html").write_text(text, encoding="utf-8")


def run(output):
    started = time.perf_counter()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    locked = protocol()
    json_save(output / "protocol.json", locked)
    before = source_fingerprint()
    json_save(output / "production_source_sha256.json", before)
    scenes = synthetic_scenes()
    scene_dir = output / "scenes"
    scene_dir.mkdir()
    for data in scenes:
        json_save(scene_dir / (data["scene_id"] + ".scene.json"), data)
    contexts = []
    for data in scenes:
        for seed in PLANNING_SEEDS:
            context = run_context(scene_dir / (data["scene_id"] + ".scene.json"), data, seed, output)
            contexts.append(context)
            print(f"DONE {len(contexts)}/{len(scenes) * len(PLANNING_SEEDS)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    assert before == source_fingerprint(), "Production source changed during benchmark"
    assert set(HOLDOUT_SEEDS).isdisjoint({s + delta for s in PLANNING_SEEDS for delta in range(5)} | {s + 100 for s in PLANNING_SEEDS})
    report = {"protocol": locked, "created_at": datetime.now(timezone.utc).isoformat(),
              "command": " ".join([sys.executable, "-m", __name__.replace("__main__", "spacewatch3d.tasks.task_5b_keyframes.examples.benchmark_evaluation"), "--output", str(output)]),
              "workdir": str(Path.cwd()), "python": sys.version, "platform": platform.platform(),
              "duration_seconds": time.perf_counter() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "production_source_unchanged": True, "actual_llm_calls": 0, "actual_images_generated": 0,
              "contexts": contexts, "aggregate": aggregate(contexts)}
    json_save(output / "metrics.json", report)
    render_report(output, report, scenes)
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))
        archive.write(Path(__file__), "benchmark_evaluation.py")
    print(json.dumps(report["aggregate"], ensure_ascii=False), flush=True)
    print(f"REPORT {output / 'report.html'}\nZIP {zip_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    run(parser.parse_args().output)
