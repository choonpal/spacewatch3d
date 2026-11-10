"""Exact computed-plan checkpoint recovery after a metadata-validator failure.

Never bypass numeric or coverage checks. A changed scoring input/module fails
closed. Failed source attempts remain immutable and are cited as producers.
"""
from pathlib import Path
from .io import read,save,sha,verify

def recover(args,planner):
    receipt=read(args.recover_draft)
    for item in receipt['files']:verify(item)
    if sha(args.scene)!=receipt['scene_sha256'] or sha(args.trajectory)!=receipt['trajectory_sha256']:
        raise ValueError('Recovery input geometry/trajectory changed')
    if sha(args.importance)!=receipt['importance_sha256'] or read(args.config)!=receipt['config']:
        raise ValueError('Recovery scoring configuration changed')
    original=Path(receipt['original_profile']);current=Path(args.profile)
    for name,digest in receipt['profile_files'].items():
        if name in ['checksums.json','code/spacewatch3d/tasks/task_5b_keyframes/validation.py']:continue
        if sha(current/name)!=digest or sha(original/name)!=digest:
            raise ValueError('Computed-checkpoint scoring/profile input changed: '+name)
    plan=read(receipt['plan']['path']);requests=read(receipt['requests']['path'])
    assert all(v['image_ref'] is None for v in plan['viewpoints'])
    planner.validate_outputs(plan,requests)
    args.output.mkdir(parents=True,exist_ok=False)
    target=args.output/'tour_plan.json'
    save(target,plan);save(args.output/'render_requests.json',requests)
    if plan['planning_status']=='complete':
        save(args.output/'reference_route.json',planner.export_reference(plan))
    scores={o['instance_id']:dict(score=o['importance'],reason=o['importance_reason'],source=o['importance_source']) for o in plan['object_importances']}
    save(args.output/'importance_scores.json',dict(scene_id=plan['scene_id'],capture_id=plan['capture_id'],scores=scores))
    save(args.output/'computed-checkpoint-recovery.json',dict(original_failed_attempt=receipt['source_attempt'],
        actual_computation_seconds=receipt['actual_computation_seconds'],receipt_sha256=sha(args.recover_draft),
        numerical_validation_rerun=True,scoring_code_and_inputs_unchanged=True,
        only_changed_profile_module='validation.py; exact relative metric metadata contract'))
    return target
