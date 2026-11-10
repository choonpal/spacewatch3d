"""Run an immutable input profile with the release's resolved planning policy."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ['profile', 'scene', 'output', 'config']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    root = args.profile.resolve()
    sys.path[:0] = [str(root / 'code'), str(root / 'native'), str(root)]
    if args.output.exists():
        raise ValueError('Use a new plan output directory')
    import native_executor, objective_executor
    native_executor.install(); objective_executor.install()
    from cache_optimization import install
    install()
    from profile_runner import install_loader
    from spacewatch3d.tasks.task_5b_keyframes import planner
    planner.load_scene = install_loader()
    def draft(plan, requests):
        folder = args.output.with_name(args.output.name + '.unvalidated')
        folder.mkdir(exist_ok=False)
        for name, value in [('plan.json', plan), ('requests.json', requests)]:
            (folder / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    planner.save_validation_draft = draft
    target = planner.generate(args.scene, args.output, json.loads(args.config.read_text()),
        importance_cache=json.loads((root / 'importance_cache.json').read_text())['scores'])
    plan = json.loads(target.read_text())
    print(json.dumps(dict(plan=str(target), count=len(plan['viewpoints']),
        clearance_m=plan['provenance']['config']['clearance_m'],
        planning_status=plan['planning_status'], input_coverage_status=plan['input_coverage_status'])), flush=True)


if __name__ == '__main__':
    main()
