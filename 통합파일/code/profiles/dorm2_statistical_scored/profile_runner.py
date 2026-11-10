"""Standalone scored Dorm3 profile. Never calls an LLM or overwrites a plan."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'native'),str(ROOT)]

def install_loader():
    import numpy as np
    from spacewatch3d.tasks.task_5b_keyframes import loader
    original=loader.load_scene
    def load(path,*args,**kwargs):
        scene=original(path,*args,**kwargs)
        profile=json.loads(Path(path).read_text()).get('navigation_profile')
        if profile:
            a=np.asarray(profile['route_grid_anchor_xy_m'],float)
            b=np.asarray(profile['fallback_tile_origin_xy_m'],float)
            assert a.shape==b.shape==(2,) and np.isfinite(a).all() and np.isfinite(b).all()
            scene.route_grid_anchor_xy_m=tuple(a.tolist())
            scene.fallback_tile_origin_xy_m=tuple(b.tolist())
            scene.provenance['navigation_profile']=profile
        return scene
    loader.load_scene=load
    return load

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists(),'Use a new output directory'
    import native_executor,objective_executor
    native_executor.install();objective_executor.install()
    from cache_optimization import install
    install()
    load=install_loader()
    from spacewatch3d.tasks.task_5b_keyframes import planner
    planner.load_scene=load
    # Draft output is scoped to this invocation, never the release directory.
    def draft(plan,requests):
        folder=args.output.with_name(args.output.name+'.unvalidated');folder.mkdir(exist_ok=False)
        for n,d in [('UNVALIDATED_plan.json',plan),('UNVALIDATED_requests.json',requests)]:
            (folder/n).write_text(json.dumps(d,ensure_ascii=False,allow_nan=False))
    planner.save_validation_draft=draft
    path=planner.generate(ROOT/'scene.json',args.output,json.loads((ROOT/'config.json').read_text()),
        importance_cache=json.loads((ROOT/'importance_cache.json').read_text())['scores'])
    plan=json.loads(path.read_text())
    print(json.dumps(dict(plan=str(path),count=len(plan['viewpoints']),planning_status=plan['planning_status']),ensure_ascii=False))

if __name__=='__main__':main()
