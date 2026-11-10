"""Native frozen v28 core planning -> free navigation -> independent checks.

All native cores are produced by create_plan with default policy, not manually
placed. Small manually prescribed core tests live in the separate unit suite.
This measures geometry integration, not RGB quality or human wayfinding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from shapely.geometry import LineString
from ..loader import load_scene
from ..models import Config
from ..planner import create_plan
from ..navigation import create_navigation, create_ordered_navigation
from ..routing import footprint
from .benchmark_evaluation import synthetic_scenes, make_scene, mesh_box, structure_box


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def scenes():
    base={s['scene_id']:s for s in synthetic_scenes()}
    yield base['cluttered_room']
    yield base['t_junction']
    yield make_scene('loop_office',[[0,0],[22,0],[22,18],[0,18]],
                     [mesh_box('panel_a','control_panel',[.4,.3,.9],[1.1,.5,1.7],.9),
                      mesh_box('panel_b','control_panel',[20.5,16.7,.9],[21.2,16.9,1.7],.9),
                      mesh_box('cabinet_a','cabinet',[.4,15,0],[1.3,16.5,1.9],.6),
                      mesh_box('cabinet_b','cabinet',[20.2,1.2,0],[21.1,2.7,1.9],.6)],
                     [structure_box([7,5,0],[15,13,2.8])],description='Central block with four surrounding corridors; no prescribed visit order')
    yield make_scene('large_T_native',[[32.8,3.6],[18.2,3.6],[18.2,21.8],[14.6,21.8],
                                      [14.6,3.6],[0,3.6],[0,0],[32.8,0]],
                     [mesh_box('left_panel','control_panel',[.5,.2,1],[1,.4,1.7],.9),
                      mesh_box('right_panel','control_panel',[31.7,.2,1],[32.2,.4,1.7],.9),
                      mesh_box('end_panel','control_panel',[15,21.3,1],[15.5,21.5,1.7],.9)],
                     start=(1.5,1.5),description='Large three-arm corridor with core poses selected automatically')


def independent_check(plan,nav,scene):
    output={v['viewpoint_id']:v for v in nav['viewpoints']}
    cores={v['viewpoint_id']:v for v in plan['viewpoints'] if 'content' in v['roles']}
    adjacency={n:set() for n in output}
    obstacles=footprint(scene,Config.from_dict(plan['provenance']['config']))
    clearance=plan['provenance']['config']['clearance_m']
    safe=True
    for link in nav['links']:
        a,b=link['from_viewpoint'],link['to_viewpoint']
        pa,pb=(np.asarray(output[n]['reference_from_camera'])[:2,3] for n in (a,b))
        line=LineString([pa,pb])
        safe &= bool(scene.floor.covers(line) and line.distance(scene.floor.boundary)>=clearance
                     and (obstacles.is_empty or (not line.intersects(obstacles) and line.distance(obstacles)>=clearance)))
        adjacency[a].add(b);adjacency[b].add(a)
    reached=set()
    if cores:
        start=min(cores);stack=[start];reached.add(start)
        while stack:
            n=stack.pop()
            for other in adjacency[n]-reached:
                reached.add(other);stack.append(other)
    published={row['viewpoint_id']:set(row['neighbors']) for row in nav['adjacency']}
    return {'frozen_core_records_identical':all(output.get(n)==v for n,v in cores.items()),
            'all_chords_independently_safe':bool(safe),'all_core_reachable_independent':set(cores)<=reached,
            'adjacency_matches_links':published==adjacency,
            'all_links_bidirectional':all(l['bidirectional'] for l in nav['links']),
            'source_failures_preserved':plan['planning_status']=='complete' or nav['navigation_status']=='incomplete',
            'no_media_claim':not nav['publication_ready'] and not nav['image_readability_validated']}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=[]
    for i,data in enumerate(scenes()):
        folder=output/data['scene_id'];folder.mkdir()
        path=folder/'scene.json';save(path,data)
        started=time.perf_counter()
        plan,requests=create_plan(path,Config(seed=301300+i))
        core_seconds=time.perf_counter()-started
        save(folder/'tour_plan.json',plan);save(folder/'core_render_requests.json',requests)
        before=hashlib.sha256((folder/'tour_plan.json').read_bytes()).hexdigest()
        scene=load_scene(path,Config.from_dict(plan['provenance']['config']))
        started=time.perf_counter();nav=create_navigation(plan,scene)
        nav_seconds=time.perf_counter()-started
        save(folder/'navigation.json',nav)
        old=create_ordered_navigation(plan,scene)
        save(folder/'ordered_navigation_comparison.json',old)
        checked=independent_check(plan,nav,scene)
        checked['source_file_unchanged']=before==hashlib.sha256((folder/'tour_plan.json').read_bytes()).hexdigest()
        row={'scene':data['scene_id'],'core_implementation':plan['provenance']['implementation'],
             'source_planning_status':plan['planning_status'],'source_input_coverage_status':plan['input_coverage_status'],
             'core_seconds':core_seconds,'navigation_seconds':nav_seconds,
             'ordered_movement_count':old['report']['movement_image_count'],
             'navigation_status':nav['navigation_status'],**nav['report'],**checked}
        report.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    save(output/'report.json',report)
    if any(not all(row[k] for k in ('frozen_core_records_identical','all_chords_independently_safe',
                                   'all_core_reachable_independent','adjacency_matches_links','all_links_bidirectional',
                                   'source_failures_preserved','no_media_claim','source_file_unchanged')) for row in report):
        raise SystemExit(1)


if __name__=='__main__':
    main()
