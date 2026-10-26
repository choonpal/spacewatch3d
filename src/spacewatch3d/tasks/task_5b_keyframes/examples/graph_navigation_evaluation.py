"""Frozen-core graph navigation validation; no keyframe reselection or RGB tests."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from shapely.geometry import box, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from .benchmark_evaluation import make_scene
from ..loader import load_scene
from ..models import Config
from ..navigation import create_ordered_navigation as create_navigation
from ..routing import footprint


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def manual(scene,xy,cores,paths,transitions=()):
    points = {n:[float(p[0]),float(p[1]),1.5] for n,p in xy.items()}
    views = []
    for n in [*cores,*transitions]:
        pose = np.eye(4)
        pose[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
        pose[:3,3] = points[n]
        views.append({'viewpoint_id':'v_'+n,'route_node_id':n,'roles':['content' if n in cores else 'transition'],
                      'reference_from_camera':pose.tolist(),'image_ref':None})
    return {'schema_name':'spacewatch3d.virtual_tour_plan','schema_version':'0.1.0',
            'scene_id':scene.scene_id,'capture_id':scene.capture_id,'reference_frame_id':scene.reference_frame_id,
            'route_id':'prescribed_frozen_cores','planning_status':'complete','viewpoints':views,'excluded_viewpoints':[],
            'route_graph':{'nodes':[{'node_id':n,'position_m':p,'component_id':'component_001'} for n,p in points.items()]},
            'tour_links':[{'from_viewpoint':'v_'+p[0],'to_viewpoint':'v_'+p[-1],'route_node_ids':p,
                           'polyline_m':[points[n] for n in p]} for p in paths],
            'provenance':{'config':asdict(Config()),'geometry_sha256':scene.provenance['geometry_sha256'],
                          'implementation':'prescribed_test_core_not_a_keyframe_benchmark'}}


def independent_cases():
    xy = {'n'+str(i):(float(x),2.1) for i,x in enumerate(np.linspace(1.25,57.25,113))}
    yield 'straight_58m',box(0,0,58.7,4.2),xy,['n0','n112'],[list(xy)],[],0
    # Same geometry, legacy two-meter transit assets must not remain compulsory.
    transitions = ['n'+str(i) for i in range(4,112,4)]
    anchors = [0,*range(4,112,4),112]
    paths = [['n'+str(i) for i in range(a,b+1)] for a,b in zip(anchors,anchors[1:])]
    yield 'legacy_spacing_58m',box(0,0,58.7,4.2),xy,['n0','n112'],paths,transitions,0
    floor = unary_union([box(0,0,18.6,3.8),box(14.8,0,18.6,16.7)])
    xy = {**{'h'+str(i):(float(x),1.9) for i,x in enumerate(np.linspace(1.2,16.7,24))},
          **{'v'+str(i):(16.7,float(y)) for i,y in enumerate(np.linspace(2.5,15.5,21))}}
    yield 'long_L',floor,xy,['h0','v20'],[list(xy)],[],1
    floor = unary_union([box(0,0,14.4,3.8),box(10.6,0,14.4,16.6),box(10.6,12.8,28.9,16.6)])
    xy = {**{'h'+str(i):(float(x),1.9) for i,x in enumerate(np.linspace(1.2,12.5,18))},
          **{'v'+str(i):(12.5,float(y)) for i,y in enumerate(np.linspace(2.6,14.7,19))},
          **{'r'+str(i):(float(x),14.7) for i,x in enumerate(np.linspace(13.3,27.7,21))}}
    yield 'double_turn',floor,xy,['h0','r20'],[list(xy)],[],2
    floor = unary_union([box(0,0,32.8,3.6),box(14.6,0,18.2,21.8)])
    xy = {'a':(1.3,1.8),'j':(16.4,1.8),'b':(31.5,1.8),'c':(16.4,20.5)}
    yield 'large_T',floor,xy,['a','b','c'],[['a','j','b'],['b','j','c']],[],1
    yield 'unused_T_arm',floor,xy,['a','b'],[['a','j','b']],[],0
    # The same star-shaped route graph in an open room needs no junction image:
    # existing core panoramas already have safe access to every gateway.
    yield 'open_room_star',box(0,0,32.8,21.8),xy,['a','b','c'],[['a','j','b'],['b','j','c']],[],0
    floor = unary_union([box(0,0,11.2,3.6),box(8,0,11.2,14.6)])
    xy = {'a':(1.2,1.8),'shared':(9.4,1.8),'late':(9.8,1.8),'b':(9.8,13.3),
          'c':(1.2,2.1),'d':(9.4,13.3)}
    yield 'shared_transit',floor,xy,['a','b','c','d'],[['a','shared','late','b'],['c','shared','d']],[],1


def checks(plan,nav,scene):
    output = {v['viewpoint_id']:v for v in nav['viewpoints']}
    cores = [v for v in plan['viewpoints'] if 'content' in v['roles']]
    nodes = {n['node_id']:np.asarray(n['position_m'][:2]) for n in plan['route_graph']['nodes']}
    config = Config.from_dict(plan['provenance']['config'])
    obstacles = footprint(scene,config)
    def safe(a,b):
        line = LineString([a,b])
        return (scene.floor.covers(line) and line.distance(scene.floor.boundary) >= config.clearance_m-1e-10
                and (obstacles.is_empty or (not line.intersects(obstacles) and line.distance(obstacles) >= config.clearance_m-1e-10)))
    edges_safe = all(safe(*[np.asarray(output[link[k]]['reference_from_camera'])[:2,3]
                           for k in ('from_viewpoint','to_viewpoint')]) for link in nav['links'])
    events = {e['event_id']:e for e in nav['decision_events']}
    witnesses = True
    for obligation in nav['decision_witnesses']:
        event = events[obligation['event_id']]
        valid = False
        for vid in obligation['witness_viewpoint_ids']:
            p = np.asarray(output[vid]['reference_from_camera'])[:2,3]
            valid |= safe(p,nodes[obligation['gateway_node_id']]) and any(safe(p,nodes[n]) for n in event['junction_node_ids'])
        witnesses &= bool(valid)
    return {'frozen_core_records_identical':all(output[v['viewpoint_id']] == v for v in cores),
            'all_chords_independently_safe':bool(edges_safe),'all_reported_gateway_witnesses_valid':witnesses,
            'no_transition_effects':all('transition_mode' not in link for link in nav['links']),
            'no_media_or_comfort_claim':not nav['publication_ready'] and not nav['image_readability_validated'],
            'source_failure_status_preserved':plan['planning_status'] == 'complete' or nav['navigation_status'] == 'incomplete'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-zip',type=Path,required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    repo = Path(__file__).resolve().parents[5]
    artifacts = repo.parents[1]/'outputs'
    save(output/'protocol.json',{'frozen_core_planner':'task5b-cpu-prototype-0.28.0, no source edits or reselection',
        'independent_case_names':[c[0] for c in independent_cases()],
        'transformed_T_angles':[13,52,117],'translation_xy_m':[23.7,-12.4],
        'negative_control':'sparse long_L with sole turning image excluded',
        'comparison':'v28 navigator on the same frozen source plan, not core-selection performance',
        'native_panel':'eight stored prior plans plus actual v27 smoke and 4000m2 plan; their version strings reported verbatim',
        'failures_and_nonoptimal_results_retained':True,'no_holdout_driven_changes':True,
        'limits':['Finite given route-node pool and extracted structural obligations only','No RGB, synthesis, user comfort, image bytes or all-branch patrol coverage']})
    baseline_path = output/'baseline_v28_navigation.py'
    with zipfile.ZipFile(args.baseline_zip) as z:
        with baseline_path.open('xb') as stream:
            stream.write(z.read('src/spacewatch3d/tasks/task_5b_keyframes/navigation.py'))
    spec = importlib.util.spec_from_file_location('spacewatch3d.tasks.task_5b_keyframes._frozen_v28_navigation',baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    report = {'controlled':[],'transforms':[],'stored_plans':[],'negative_controls':[],'failures':[]}
    with tempfile.TemporaryDirectory(prefix='task5b-graph-final-',dir='/dev/shm') as temp:
        root = Path(temp)
        for name,floor,xy,cores,paths,transitions,expected in independent_cases():
            data = make_scene(name,list(map(list,list(floor.exterior.coords)[:-1])))
            source_path = root/'scene.json'
            source_path.write_text(json.dumps(data),encoding='utf-8')
            scene = load_scene(source_path,Config())
            plan = manual(scene,xy,cores,paths,transitions)
            if transitions:
                plan['provenance']['config']['max_link_m'] = 2.
            encoded = json.dumps(plan,sort_keys=True)
            old = baseline.create_navigation(plan,scene)
            start = time.perf_counter()
            nav = create_navigation(plan,scene)
            duration = time.perf_counter()-start
            checked = checks(plan,nav,scene)
            checked['source_manifest_unchanged'] = encoded == json.dumps(plan,sort_keys=True)
            row = {'scene':name,'source_core_count':len(cores),'v28_movement_count':sum('content' not in v['roles'] for v in old['viewpoints']),
                   'movement_count':nav['report']['movement_image_count'],'expected_count_predeclared':expected,
                   'events':len(nav['decision_events']),'status':nav['navigation_status'],'minimum_proven':nav['report']['minimum_movement_images_proven'],
                   'runtime_seconds':duration,**checked}
            report['controlled'].append(row)
            if not all(checked.values()) or row['movement_count'] != expected or row['status'] != 'complete' or not row['minimum_proven']:
                report['failures'].append(row)
            save(output/(name+'.scene.json'),data)
            save(output/(name+'.source_plan.json'),plan)
            save(output/(name+'.navigation.json'),nav)
            if name == 'large_T':
                for angle in (13,52,117):
                    transformed = copy.deepcopy(plan)
                    theta = np.deg2rad(angle)
                    rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    original_floor = scene.floor
                    scene.floor = translate(rotate(original_floor,angle,origin=(0,0)),23.7,-12.4)
                    for node in transformed['route_graph']['nodes']:
                        node['position_m'][:2] = (rotation @ np.asarray(node['position_m'][:2])+[23.7,-12.4]).tolist()
                    by_node = {n['node_id']:n for n in transformed['route_graph']['nodes']}
                    for view in transformed['viewpoints']:
                        for i,x in enumerate(by_node[view['route_node_id']]['position_m']):
                            view['reference_from_camera'][i][3] = x
                    # This controlled no-interior-obstacle case uses its transformed
                    # floor boundary; do not keep untransformed shell collisions.
                    geometries = scene.geometries
                    scene.geometries = []
                    result = create_navigation(transformed,scene)
                    checked = checks(transformed,result,scene)
                    row = {'angle_deg':angle,'movement_count':result['report']['movement_image_count'],
                           'events':len(result['decision_events']),'status':result['navigation_status'],**checked}
                    report['transforms'].append(row)
                    if not all(checked.values()) or row['movement_count'] != 1 or row['events'] != 1 or row['status'] != 'complete':
                        report['failures'].append(row)
                    scene.floor,scene.geometries = original_floor,geometries
        # An intentionally sparse unavailable corner must stay failed.
        data = make_scene('excluded_corner',[[0,0],[8,0],[8,8],[5,8],[5,3],[0,3]])
        path = root/'negative.json'
        path.write_text(json.dumps(data),encoding='utf-8')
        scene = load_scene(path,Config())
        plan = manual(scene,{'a':(1.2,1.5),'j':(6.5,1.5),'b':(6.5,6.8)},['a','b'],[['a','j','b']])
        plan['excluded_viewpoints'] = [{'viewpoint_id':'vp_j','route_node_id':'j'}]
        nav = create_navigation(plan,scene)
        row = {'scene':'excluded_corner','status':nav['navigation_status'],'movement_count':nav['report']['movement_image_count'],
               'minimum_proven':nav['report']['minimum_movement_images_proven'],'failures':nav['failures'],**checks(plan,nav,scene)}
        report['negative_controls'].append(row)
        if row['status'] != 'incomplete' or row['minimum_proven']:
            report['failures'].append(row)
    stored = artifacts/'task5b_v27_route_anchor_20261013'
    panels = [(name,stored/('integration_'+name)/'tour_plan.json',stored/(name+'.scene.json'))
              for name in ('open_room','cluttered_room','two_rooms_doorway','l_corridor','t_junction',
                           'overhead_and_small','disconnected_control','sealed_object_control')]
    panels += [('actual_v27_smoke',stored/'release_smoke/tour_plan.json',stored/'release_smoke.scene.json'),
               ('open_hall_4000m2',artifacts/'task5b_v27_visualization_3d_20261013/open_hall_80x50/tour_plan.json',
                artifacts/'task5b_v23_large_space_20261012/open_hall_80x50.scene.json')]
    for name,plan_path,scene_path in panels:
        source_bytes = plan_path.read_bytes()
        plan = json.loads(source_bytes)
        scene = load_scene(scene_path,Config.from_dict(plan['provenance']['config']))
        start = time.perf_counter()
        nav = create_navigation(plan,scene)
        checked = checks(plan,nav,scene)
        checked['source_file_bytes_unchanged'] = source_bytes == plan_path.read_bytes()
        row = {'scene':name,'stored_core_implementation':plan['provenance']['implementation'],
               'source_planning_status':plan['planning_status'],'source_input_coverage_status':plan.get('input_coverage_status'),
               'navigation_status':nav['navigation_status'],'events':len(nav['decision_events']),
               'runtime_seconds':time.perf_counter()-start,**nav['report'],**checked}
        report['stored_plans'].append(row)
        if not all(checked.values()):
            report['failures'].append(row)
        save(output/(name+'.stored.navigation.json'),nav)
    hashes = json.loads((output.parent/'frozen_files_before.json').read_text())
    changed = [n for n,h in hashes.items() if hashlib.sha256((repo/n).read_bytes()).hexdigest() != h]
    report['frozen_files_changed'] = changed
    if changed:
        report['failures'].append({'frozen_files_changed':changed})
    save(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(bool(report['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
