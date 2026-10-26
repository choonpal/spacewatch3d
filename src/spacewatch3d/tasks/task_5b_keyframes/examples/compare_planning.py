"""Small-scene ablation with identical patches and a shared observable denominator.

This compares candidate density and routed selection, not photographic quality
or equal-image-budget methods. All output directories must be new.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from spacewatch3d.tasks.task_5b_keyframes.importance import resolve_importance
from spacewatch3d.tasks.task_5b_keyframes.loader import load_scene
from spacewatch3d.tasks.task_5b_keyframes.models import Config, InputError, read_json
from spacewatch3d.tasks.task_5b_keyframes.planner import generate
from spacewatch3d.tasks.task_5b_keyframes.routing import build_route
from spacewatch3d.tasks.task_5b_keyframes.selection import Objective
from spacewatch3d.tasks.task_5b_keyframes.visibility import evaluate_positions, sample_surface


def compare(scene_path, output_dir):
    base = Config()
    scene = load_scene(scene_path, base)
    base = replace(base, camera_height_m=scene.camera_height_m)
    route = build_route(scene, base)
    if len(route.positions) > 1000:
        raise InputError("This diagnostic is limited to 1000 reference route nodes")
    surface = sample_surface(scene, base)
    objects, _ = resolve_importance(scene.objects, base)
    visible, quality = evaluate_positions(route.positions, surface, objects, base)
    reference = Objective(visible, quality, surface.weights, surface.regions, objects, base)
    node_index = {route.node_id(i): i for i in range(len(route.positions))}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = {
        "uniform_0_6": replace(base, adaptive_candidates=False, route_aware_selection=False),
        "uniform_0_3": replace(base, candidate_spacing_m=.3, adaptive_candidates=False, route_aware_selection=False),
        "uniform_0_15": replace(base, candidate_spacing_m=.15, adaptive_candidates=False, route_aware_selection=False),
        "adaptive_gain_only": replace(base, route_aware_selection=False),
        "adaptive_route_aware": base,
        "adaptive_route_aware_high_coverage": replace(base, space_target=.99, region_target=.99),
    }
    results = []
    for name, config in cases.items():
        started = time.perf_counter()
        plan = read_json(generate(scene_path, output_dir / name, config))
        elapsed = time.perf_counter() - started
        _, shared = reference.measure([node_index[v["route_node_id"]] for v in plan["viewpoints"]])
        results.append({"case": name, "planning_status": plan["planning_status"],
                        "candidate_count": plan["coverage_report"]["candidate_count_before_feedback"],
                        "content_count": plan["coverage_report"]["content_count"],
                        "transition_count": plan["coverage_report"]["transition_count"],
                        "unique_image_count": len(plan["viewpoints"]),
                        "tour_length_m": plan["coverage_report"]["selection"]["tour_length_m"],
                        "shared_reference_coverage_observable": shared["coverage_observable"],
                        "shared_reference_balanced_space_coverage": shared["balanced_space_coverage"],
                        "input_total_surface_coverage": shared["coverage_total"],
                        "shared_reference_object_importance_coverage": shared["object_importance_coverage_observable"],
                        "shared_reference_unmet_targets": shared["unmet_targets"],
                        "runtime_seconds": round(elapsed, 4),
                        "local_swap_applied": plan["coverage_report"]["selection"]["local_swap_applied"]})
        results[-1]["required_space_target"] = config.space_target
        results[-1]["required_region_target"] = config.region_target
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    report = {"scene_id": scene.scene_id, "synthetic": scene.synthetic,
              "comparison": "candidate_density_and_routed_selection_ablation_not_equal_image_budget",
              "reference": {"method": "all_collision_safe_route_graph_nodes", "node_count": len(route.positions),
                            "surface_sample_count": len(surface.points), "seed": base.seed,
                            "observable_area_m2": float(surface.weights[reference.observable].sum()),
                            "same_patches_and_denominator_for_all_cases": True},
              "actual_llm_calls": 0, "actual_images_generated": 0, "results": results}
    path = output_dir / "comparison_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.scene, args.output))
