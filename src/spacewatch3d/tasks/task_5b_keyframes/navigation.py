"""Frozen content + free-exploration graph; historical ordered mode retained."""
import copy
import json
from pathlib import Path
import numpy as np
from .models import Config, InputError, read_json
from .routing import footprint
from .navigation_graph import core_paths, decision_events, prepare_paths, solve_minimum

NAVIGATION_IMPLEMENTATION = 'task5b-free-navigation-0.5.0'
ORDERED_NAVIGATION_IMPLEMENTATION = 'task5b-graph-navigation-0.2.0'


def create_ordered_navigation(plan, scene):
    if plan.get('schema_name') != 'spacewatch3d.virtual_tour_plan' or plan.get('schema_version') != '0.1.0':
        raise InputError('Expected a Task5b tour plan')
    for key in ('scene_id','capture_id','reference_frame_id'):
        if plan.get(key) != getattr(scene,key):
            raise InputError('Navigation '+key+' mismatch')
    if plan['provenance']['geometry_sha256'] != scene.provenance['geometry_sha256']:
        raise InputError('Navigation geometry mismatch')
    config = Config.from_dict(plan['provenance']['config'])
    obstacles = footprint(scene,config)
    nodes = {n['node_id']:n for n in plan['route_graph']['nodes']}
    source = {v['viewpoint_id']:copy.deepcopy(v) for v in plan['viewpoints']}
    for view in source.values():
        pose = np.asarray(view['reference_from_camera'],float)
        if pose.shape != (4,4) or not np.allclose(pose[:3,3],nodes[view['route_node_id']]['position_m'],rtol=0,atol=1e-8):
            raise InputError('Navigation source image pose disagrees with route position')
    cores = {vid:v for vid,v in source.items() if 'content' in v['roles']}
    core_nodes = {v['route_node_id'] for v in cores.values()}
    by_node = {v['route_node_id']:vid for vid,v in source.items()}
    excluded = {v['viewpoint_id'] for v in plan.get('excluded_viewpoints',[])}
    excluded_nodes = set()
    for record in plan.get('excluded_viewpoints',[]):
        node = record.get('route_node_id')
        if node is None and record['viewpoint_id'].startswith('vp_'):
            node = record['viewpoint_id'][3:]
        if node not in nodes:
            raise InputError('Cannot locate an excluded image in this legacy plan; regenerate the source plan')
        excluded_nodes.add(node)
    eligible = {n for n in nodes if n not in excluded_nodes and by_node.get(n,'vp_'+n) not in excluded}
    paths,failures = core_paths(plan,source,cores)
    if not cores:
        failures.append({'code':'no_content_viewpoints'})
    events,_ = decision_events(paths,nodes,core_nodes,scene.floor,obstacles,config.clearance_m)
    prepared,gaps = prepare_paths(paths,events,nodes,eligible,scene.floor,obstacles,config.clearance_m)
    failures.extend(gaps)
    visits,optimizer = solve_minimum(prepared,core_nodes)
    if visits is None:
        failures.append({'code':'navigation_constraints_unsolved','optimizer':optimizer})
        visits = []
    views = dict(cores)
    links,event_visits = [],[]
    for path,indices in zip(prepared,visits):
        ids,points = path['route_node_ids'],path['points']
        for i in indices:
            node = ids[i]
            if node in by_node:
                vid = by_node[node]
                if vid not in views:
                    views[vid] = source[vid]
            else:
                vid = 'vp_'+node
                if vid in source or vid in views:
                    raise InputError('Navigation viewpoint ID collision')
                pose = np.eye(4)
                pose[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
                pose[:3,3] = points[i]
                views[vid] = {'viewpoint_id':vid,'route_node_id':node,'component_id':nodes[node]['component_id'],
                              'reference_from_camera':pose.tolist(),'roles':['transition'],
                              'reason':'minimum_graph_navigation_stop','image_ref':None,'render_status':'pending',
                              'score_basis':'navigation_graph_geometry_only'}
                by_node[node] = vid
        for a,b in zip(indices,indices[1:]):
            delta = points[b,:2]-points[a,:2]
            links.append({'from_viewpoint':by_node[ids[a]],'to_viewpoint':by_node[ids[b]],
                          'source_path_index':path['source_path_index'],'source_link_indices':path['source_link_indices'],
                          'route_node_ids':ids[a:b+1],'safe_route_polyline_m':points[a:b+1].tolist(),
                          'safe_chord_m':[points[a].tolist(),points[b].tolist()],
                          'length_m':float(np.linalg.norm(delta)),'geometry_safe':True,
                          'world_heading_rad':float(np.arctan2(delta[1],delta[0])),
                          'image_overlap_validated':False})
        for requirement in path['requirements']:
            event_visits.append({'source_path_index':path['source_path_index'],
                                 'event_id':requirement['event_id'],'gateway_node_id':requirement['gateway_node_id'],
                                 'witness_viewpoint_ids':[by_node[ids[i]] for i in indices if i in requirement['witness_indices']]})
    movement = [v for vid,v in views.items() if vid not in cores]
    new = [v for v in movement if v['viewpoint_id'] not in source]
    requests = [{'viewpoint_id':v['viewpoint_id'],'capture_id':scene.capture_id,'reference_frame_id':scene.reference_frame_id,
                 'reference_from_camera':v['reference_from_camera'],'projection':'equirectangular',
                 'width':config.width,'height':config.height,'media_origin':'novel_view',
                 'render_required':True,'purpose':'navigation'} for v in movement if not v.get('image_ref')]
    return {'schema_name':'spacewatch3d.tour_navigation','schema_version':'0.2.0',
            'implementation':ORDERED_NAVIGATION_IMPLEMENTATION,'scene_id':scene.scene_id,'capture_id':scene.capture_id,
            'reference_frame_id':scene.reference_frame_id,'source_route_id':plan['route_id'],
            'source_planning_status':plan['planning_status'],
            'source_input_coverage_status':plan.get('input_coverage_status','unreported'),
            'navigation_status':'incomplete' if failures or plan['planning_status'] != 'complete' else 'complete',
            'navigation_status_scope':'frozen_core_paths_safe_chords_and_routed_gateway_witnesses_only',
            'image_readability_validated':False,'publication_ready':False,'viewpoints':list(views.values()),
            'links':links,'decision_events':events,'decision_witnesses':event_visits,'render_requests':requests,'failures':failures,
            'optimizer':optimizer,
            'report':{'source_image_count':len(source),'frozen_content_count':len(cores),
                      'movement_image_count':len(movement),'added_image_count':len(new),'total_image_count':len(views),
                      'released_source_transition_count':sum(vid not in cores and vid not in views for vid in source),
                      'reused_source_transition_count':len(movement)-len(new),'forced_distance_spacing_m':None,
                      'method':'global_binary_image_activation_and_ordered_path_flows_with_gateway_witnesses',
                      'minimum_movement_images_proven':optimizer['minimum_movement_images_proven'] and not failures,
                      'minimum_scope':'given_frozen_core_paths_route_node_candidates_extracted_events_and_safe_chords',
                      'continuous_space_or_human_comfort_optimum_guaranteed':False,
                      'source_content_changed':False,'actual_media_bytes':None},
            'warnings':['Gateway witnesses use geometric safe access, not RGB recognition or a human-comfort guarantee.',
                        'Only routes connecting frozen content points are included; unused branches are not new patrol goals.',
                        'No fixed distance spacing, scene-change threshold, transition effects or intermediate images are generated.',
                        'Core coverage and source planning/whole-input failures are not repaired or rescored here.']}


def create_navigation(plan, scene, **solver_options):
    from .navigation_free import create_free_navigation
    from .navigation_rng import apply_relative_neighborhood
    return apply_relative_neighborhood(create_free_navigation(plan, scene, **solver_options))


def generate_navigation(scene_path,plan_path,output_dir,reference_route_path=None,registration_path=None,**solver_options):
    from .loader import load_scene
    from .reference import load_reference
    output = Path(output_dir).resolve()
    if output.exists():
        raise InputError('Output already exists; choose a fresh directory')
    plan = read_json(Path(plan_path))
    config = Config.from_dict(plan['provenance']['config'])
    scene = load_scene(Path(scene_path),config)
    if registration_path is not None and reference_route_path is None:
        raise InputError('registration requires reference_route_path')
    if reference_route_path is not None:
        load_reference(reference_route_path,scene,config,registration_path)
    navigation = create_navigation(plan,scene,**solver_options)
    output.mkdir(parents=True,exist_ok=False)
    for view in navigation['viewpoints']:
        if view.get('image_ref'):
            view['image_ref'] = str((Path(plan_path).resolve().parent/view['image_ref']).resolve())
    path = output/'navigation.json'
    path.write_text(json.dumps(navigation,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path
