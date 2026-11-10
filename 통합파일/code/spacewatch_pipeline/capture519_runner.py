"""Actual-frame route injection with optional paper selector on measured V/Q."""
import argparse
import faulthandler
import sys
from pathlib import Path
from .io import ROOT, read, save


def main():
    parser=argparse.ArgumentParser()
    for key in ('scene','trajectory','output','config'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--importance',type=Path)
    parser.add_argument('--recover-draft',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a fresh output directory')
    sys.dont_write_bytecode=True
    config=read(args.config)
    paper_options=config.pop('paper_selection',None)
    if paper_options is not None and args.recover_draft:
        raise ValueError('Paper selection must consume fresh measured observations, not recover a stored draft')
    if args.profile:
        sys.path[:0]=[str(args.profile/'code'),str(args.profile/'native'),str(args.profile)]
    else:sys.path.insert(0,str(ROOT/'vendor/native'))
    import native_executor, objective_executor
    native_executor.install();objective_executor.install()
    if args.profile:
        from cache_optimization import install
        install()
        from profile_runner import install_loader
        load=install_loader()
    from spacewatch3d.tasks.task_5b_keyframes import planner
    if args.profile:planner.load_scene=load
    if paper_options is not None:
        from .paper_selector import install as install_paper_selector
        install_paper_selector(planner,args.output.with_name(args.output.name+'.selector.json'),paper_options)
    from .captured_route import build
    trajectory=read(args.trajectory)
    planner.build_route=lambda scene,config:build(trajectory,scene,config)
    def draft(plan,requests):
        folder=args.output.with_name(args.output.name+'.unvalidated')
        folder.mkdir(exist_ok=False)
        save(folder/'plan.json',plan);save(folder/'requests.json',requests)
    planner.save_validation_draft=draft
    scores=None
    if args.importance:scores=read(args.importance)['scores']
    elif args.profile:scores=read(args.profile/'importance_cache.json')['scores']
    faulthandler.enable();faulthandler.dump_traceback_later(300,repeat=True)
    if args.recover_draft:
        from .recover_planning_draft import recover
        target=recover(args,planner)
    else:target=planner.generate(args.scene,args.output,config,importance_cache=scores)
    faulthandler.cancel_dump_traceback_later()
    plan=read(target);requests=read(target.with_name('render_requests.json'))
    frames=trajectory['frames']
    by_id={'vp_frame_'+str(f['source_frame_index']).zfill(8):f for f in frames}
    for view in plan['viewpoints']:
        frame=by_id[view['viewpoint_id']]
        if view['reference_from_camera']!=frame['reference_from_camera']:
            raise ValueError('Selected pose differs from the source frame pose')
        view['source_frame']={k:v for k,v in frame.items() if k!='reference_from_camera'}
    for request in requests['requests']:
        frame=by_id[request['viewpoint_id']]
        request.update(media_origin='original_video_frame',source_frame={k:v for k,v in frame.items() if k!='reference_from_camera'})
    # Display navigation is post-selection only. Retain every recorded bend and
    # revisit, not synthetic floor paths or extra connection-only images.
    views=sorted(plan['viewpoints'],key=lambda v:v['source_frame']['pose_index'])
    links=[]
    from .existing_path import path_between
    import math
    for first,last in zip(views,views[1:]):
        a,b=first['source_frame']['pose_index'],last['source_frame']['pose_index']
        line=path_between(trajectory,a,b)
        links.append(dict(from_viewpoint=first['viewpoint_id'],to_viewpoint=last['viewpoint_id'],
            component_id=first['component_id'],length_m=sum(math.dist(x,y) for x,y in zip(line,line[1:])),
            route_node_ids=[plan['route_graph']['nodes'][i]['node_id'] for i in range(a,b+1)],
            polyline_m=line,connection_method='captured_video_path',physical_safety_verified=False))
    plan['captured_navigation']=dict(tour_order=[v['viewpoint_id'] for v in views],links=links,
        affects_selection=False,physical_safety_verified=False,added_connection_only_images=0)
    plan['provenance']['capture_adapter']=dict(base='5.1.9',
        changed_boundaries=['route source','camera pose/height','post-selection recorded navigation','native image export']+
            (['paper multistart/exchange selector'] if paper_options is not None else []),
        original_objective_selection_visibility_validation_unchanged=paper_options is None and not config.get('object_relative_projected_quality',False),
        native_visibility_object_quality_and_statistical_validation_preserved=True,
        paper_selector_enabled=paper_options is not None,
        user_requested_object_quality_change=plan['coverage_report']['object_observation_policy'],
        pose_origin=trajectory.get('pose_origin'),pose_accuracy_verified=trajectory.get('pose_accuracy_verified',False),
        physical_safety_verified=False,no_gaussian=True,no_new_pose_inference=True)
    plan['capture_status_scope']='Recorded-frame candidates; not image-legibility or physical-safety certification'
    plan['warnings'].append(dict(code='captured_path_pose_accuracy_unverified',
        message='Non-anchor frame poses are interpolated on the recorded path; coverage is not independently measured.'))
    from spacewatch3d.tasks.task_5b_keyframes.validation import validate_outputs
    from .capture519_validate import validate_native
    validate_native(plan,requests,validate_outputs)
    save(target,plan);save(target.with_name('render_requests.json'),requests)
    print('CAPTURE519_PLAN',len(views),plan['planning_status'],flush=True)


if __name__=='__main__':main()
