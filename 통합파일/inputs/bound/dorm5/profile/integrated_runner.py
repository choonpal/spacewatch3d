"""Isolated profile dispatch for integration. Scores are local cached data only."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'native'),str(ROOT)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--validate',type=Path);parser.add_argument('--requests',type=Path);args=parser.parse_args()
    if args.validate:
        from spacewatch3d.tasks.task_5b_keyframes.validation import validate_outputs
        validate_outputs(json.loads(args.validate.read_text()),json.loads(args.requests.read_text()));print('PROFILE_CONTRACT_OK');return
    assert args.scene and args.output and not args.output.exists()
    import native_executor,objective_executor
    native_executor.install();objective_executor.install()
    from cache_optimization import install
    install()
    from profile_runner import install_loader
    load=install_loader()
    from spacewatch3d.tasks.task_5b_keyframes import planner
    planner.load_scene=load
    def draft(plan,requests):
        folder=args.output.with_name(args.output.name+'.unvalidated');folder.mkdir(exist_ok=False)
        for name,value in [('plan.json',plan),('requests.json',requests)]:
            (folder/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    planner.save_validation_draft=draft
    target=planner.generate(args.scene,args.output,json.loads((ROOT/'config.json').read_text()),
        importance_cache=json.loads((ROOT/'importance_cache.json').read_text())['scores'])
    plan=json.loads(target.read_text());print(json.dumps(dict(plan=str(target),count=len(plan['viewpoints']),
        planning_status=plan['planning_status'],input_coverage_status=plan['input_coverage_status'])),flush=True)


if __name__=='__main__':main()
