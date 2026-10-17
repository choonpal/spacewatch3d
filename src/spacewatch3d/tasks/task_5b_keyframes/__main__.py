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
    plan_parser.add_argument("--importance-cache", type=Path)
    plan_parser.add_argument("--feedback", type=Path)
    plan_parser.add_argument("--reference-route", type=Path, help="Completed reference_route.json from another capture")
    plan_parser.add_argument("--registration", type=Path, help="Successful external current-to-reference metric alignment")
    validator = commands.add_parser("validate")
    validator.add_argument("plan", type=Path)
    validator.add_argument("requests", type=Path)
    args = parser.parse_args(argv)
    try:
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
        path = generate(args.scene, args.output,
                        read_json(args.config) if args.config else None,
                        importance_cache=cache, feedback_path=args.feedback,
                        reference_route_path=args.reference_route, registration_path=args.registration)
        plan = read_json(path)
        print(json.dumps({"planning_status": plan["planning_status"], "plan": str(path),
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
