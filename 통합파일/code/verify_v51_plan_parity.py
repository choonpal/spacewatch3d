#!/usr/bin/env python3
"""CPU-only full planner rerun with exact plan/request comparison."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'vendor/spacewatch3d_src'), str(ROOT/'vendor/native')]
import native_executor
import objective_executor
from spacewatch3d.tasks.task_5b_keyframes import planner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline-plan', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    native_executor.install()
    objective_executor.install(immutable=True)
    timings = {}
    for name in ('load_scene', 'build_route', 'prepare_repair_samples', 'select_keyframes',
                 'refine_lateral_plan', 'repair_connected_selection', 'validate_coverage'):
        original = getattr(planner, name)
        def wrap(function, key):
            @functools.wraps(function)
            def run(*values, **kwargs):
                print('start', key, flush=True)
                start = time.perf_counter()
                result = function(*values, **kwargs)
                timings[key] = time.perf_counter()-start
                print('finish', key, timings[key], flush=True)
                return result
            return run
        setattr(planner, name, wrap(original, name))
    start = time.perf_counter()
    target = planner.generate(args.scene.resolve(), args.output_directory,
                              json.loads(args.config.read_text()))
    elapsed = time.perf_counter()-start
    plan = json.loads(target.read_text())
    baseline = json.loads(args.baseline_plan.read_text())
    requests = target.with_name('render_requests.json')
    previous_requests = args.baseline_plan.with_name('render_requests.json')
    report = dict(scope='local CPU full planner; not server/full pipeline benchmark',
                  exact_plan_parity=plan == baseline,
                  exact_render_requests_parity=json.loads(requests.read_text()) == json.loads(previous_requests.read_text()),
                  baseline_plan=str(args.baseline_plan.resolve()), plan=str(target),
                  baseline_plan_sha256=hashlib.sha256(args.baseline_plan.read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  elapsed_seconds=elapsed, stages_seconds=timings,
                  overall_speedup_comparison_valid=False,
                  reason='Baseline includes partial profiling overhead; no overall speedup claimed',
                  source_root=str(ROOT),
                  code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(planner.__file__), Path(objective_executor.__file__),
                                ROOT/'vendor/spacewatch3d_src/spacewatch3d/tasks/task_5b_keyframes/loader.py']})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    assert report['exact_plan_parity'] and report['exact_render_requests_parity'], 'Full planner outputs changed'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
