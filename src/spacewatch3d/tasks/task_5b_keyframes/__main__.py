"""Independent CLI for virtual tour planning; shared CLI contract unchanged."""

import argparse
import json
from pathlib import Path
import sys

from .models import InputError, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task5b central-route / 360-degree keyframe position planner")
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--scene", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are never overwritten")
    plan_parser.add_argument("--config", type=Path)
    plan_parser.add_argument("--real-mesh", action="store_true",
        help="Real-mesh profile: exact tiles, within-budget tile allocation and safe observation spurs; raw mesh/goals remain")
    plan_parser.add_argument("--importance-cache", type=Path)
    plan_parser.add_argument("--feedback", type=Path)
    plan_parser.add_argument("--reference-route", type=Path, help="Completed reference_route.json from another capture")
    plan_parser.add_argument("--registration", type=Path, help="Successful external current-to-reference metric alignment")
    validator = commands.add_parser("validate")
    validator.add_argument("plan", type=Path)
    validator.add_argument("requests", type=Path)
    completion = commands.add_parser("repair-navigation", help="Opt-in bounded planar hole completion; observations unchanged")
    completion.add_argument("--scene", type=Path, required=True)
    completion.add_argument("--output", type=Path, required=True, help="New directory for derived scene and hypothesis report")
    completion.add_argument("--config", type=Path)
    completion.add_argument("--max-hole-span-m", type=float)
    completion.add_argument("--plane-tolerance-m", type=float)
    completion.add_argument("--max-normal-angle-deg", type=float)
    navigation = commands.add_parser("navigate", help="Minimum-image bidirectional free-exploration graph with frozen content")
    navigation.add_argument("--scene", type=Path, required=True)
    navigation.add_argument("--plan", type=Path, required=True)
    navigation.add_argument("--output", type=Path, required=True)
    navigation.add_argument("--reference-route", type=Path)
    navigation.add_argument("--registration", type=Path)
    navigation.add_argument("--primary-time-limit-s", type=float,
                            help="Minimum-count solver budget in seconds (default 120; not an end-to-end deadline)")
    navigation.add_argument("--secondary-time-limit-s", type=float,
                            help="Optional distance tie-break solver budget (default 5; 0 skips it)")
    args = parser.parse_args(argv)
    try:
        if args.command == "repair-navigation":
            from .navigation_completion import generate_completed_scene
            target = generate_completed_scene(args.scene,args.output,
                read_json(args.config) if args.config else None,
                max_hole_span_m=args.max_hole_span_m,plane_tolerance_m=args.plane_tolerance_m,
                max_normal_angle_deg=args.max_normal_angle_deg)
            print(json.dumps({"derived_scene":str(target),"report":str(target.with_name("navigation_completion.json")),
                              "physical_safety_verified":False},ensure_ascii=False))
            return 0
        if args.command == "navigate":
            from .navigation import generate_navigation
            solver_options = {key: value for key, value in (
                ('primary_time_limit_s', args.primary_time_limit_s),
                ('secondary_time_limit_s', args.secondary_time_limit_s)) if value is not None}
            path = generate_navigation(args.scene, args.plan, args.output, args.reference_route, args.registration, **solver_options)
            result = read_json(path)
            print(json.dumps({"navigation": str(path), "navigation_status": result["navigation_status"],
                              **result["report"]}, ensure_ascii=False))
            return 0 if result["navigation_status"] == "complete" else 3
        if args.command == "validate":
            from .validation import validate_outputs
            validate_outputs(read_json(args.plan), read_json(args.requests))
            print("Task5b plan and render requests are internally consistent (not image-quality validation).")
            return 0
        from .planner import generate
        cache = None
        if args.importance_cache:
            cache_data = read_json(args.importance_cache)
            scene = read_json(args.scene)
            for key in ("scene_id", "capture_id"):
                if cache_data.get(key) != scene.get(key):
                    raise InputError(f"Importance cache {key} mismatch")
            cache = cache_data.get("scores")
            if not isinstance(cache, dict):
                raise InputError("Importance cache needs a scores object")
        config=read_json(args.config) if args.config else {}
        if args.real_mesh:
            config.update(spatially_stratified_sampling=True)
            config.setdefault('balance_automatic_region_samples',True)
            config.setdefault('lateral_candidate_refinement',True)
        path = generate(args.scene, args.output, config,
                        importance_cache=cache, feedback_path=args.feedback,
                        reference_route_path=args.reference_route, registration_path=args.registration)
        plan = read_json(path)
        print(json.dumps({"planning_status": plan["planning_status"], "plan": str(path),
                          "planning_status_scope": plan["planning_status_scope"],
                          "input_coverage_status": plan["input_coverage_status"],
                          "image_count": len(plan["viewpoints"]),
                          "quality_status": plan["quality_status"],
                          "publication_ready": False}, ensure_ascii=False))
        return 0 if plan["planning_status"] == "complete" else 3
    except ModuleNotFoundError as exc:
        print(json.dumps({"error": "Task5b dependencies are missing; install the task requirements in its own environment", "detail": str(exc)}), file=sys.stderr)
        return 2
    except (InputError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"planning_status": "invalid_input", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
