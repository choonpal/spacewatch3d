"""Export saved Task5b links in the same frozen frame as the rendered cameras."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from prepare import FrozenPartition, epoch_to_common, membership, read, save, sha, transform


def common_points(points, protocol, view, classifier):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Invalid saved route polyline')
    epoch=view['anchor_epoch']
    bridge=np.asarray(protocol['scenes'][epoch]['native_to_mesh'])@np.asarray(view['source_reference_to_gaussian'])
    labels=membership(protocol,transform(points,bridge),epoch,classifier)
    result=np.empty_like(points)
    for region in np.unique(labels):
        selected=labels==region
        result[selected]=transform(points[selected],epoch_to_common(protocol['alignment_context'],epoch,f'R{region:02d}')@bridge)
    return result


def convert_plan(plan, views, convert):
    """Keep visit order and only actual links; split at every missing link."""
    by_source={v['viewpoint_id']:v for v in views}
    nodes=[]
    for visit in plan['tour_visits']:
        view=by_source[visit['viewpoint_id']]
        nodes.append({'id':view['id'],'viewpoint_id':view['viewpoint_id'],
                      'visit_id':visit['visit_id'],'component_id':visit['component_id'],
                      'position':[row[3] for row in view['camera_to_common_cv'][:3]]})
    if len(nodes)!=len(views) or len({n['id'] for n in nodes})!=len(views):
        raise ValueError('Saved visits do not match rendered keyframes exactly')
    links=[];pairs=set()
    for link in plan['tour_links']:
        a=by_source[link['from_viewpoint']];b=by_source[link['to_viewpoint']]
        path=convert(link['polyline_m'],a)
        for index,view in [(0,a),(-1,b)]:
            expected=np.asarray(view['camera_to_common_cv'])[:3,3]
            if not np.allclose(path[index],expected,atol=1e-7,rtol=0):
                raise ValueError('Saved link endpoint differs from rendered keyframe')
        pair=(a['id'],b['id'])
        if pair in pairs or pair[::-1] in pairs:raise ValueError('Duplicate saved link')
        pairs.add(pair)
        links.append({'from':a['id'],'to':b['id'],'component_id':link['component_id'],
                      'polyline':path.tolist(),'source_polyline_points':len(path),
                      'source_length_m':link.get('length_m'),'connection_method':link.get('connection_method')})
    segments=[]
    for node in nodes:
        previous=segments[-1]['nodes'][-1] if segments else None
        connected=(previous,node['id']) in pairs or (node['id'],previous) in pairs
        if not connected:
            segments.append({'id':f'segment-{len(segments)+1:02d}','component_id':node['component_id'],'nodes':[]})
        segments[-1]['nodes'].append(node['id'])
    return {'nodes':nodes,'links':links,'segments':segments,
            'route_id':plan['route_id'],'planning_status':plan.get('planning_status'),
            'publication_ready':plan.get('publication_ready'),
            'missing_visit_connections':len(segments)-1}


def build(run, destination):
    protocol=read(run/'protocol.json')
    classifier=FrozenPartition(dict(np.load(run/'partition.npz')))
    groups=[]
    for source in protocol['keyframe_sources']:
        path=Path(source['tour_plan'])
        if sha(path)!=source['tour_plan_sha256']:raise ValueError('Source tour plan changed')
        views=[v for v in protocol['views'] if v['plan_group']==source['group']]
        group=convert_plan(read(path),views,lambda points,view:common_points(points,protocol,view,classifier))
        group.update(id=source['group'],capture=source['capture'],source_plan_sha256=source['tour_plan_sha256'])
        groups.append(group)
    result={'schema':'registered-keyframe-tour-routes-v1','groups':groups,
            'coordinate_frame':'same common Y-up frame as gaussian-data camera_to_common_cv',
            'new_links_created':False,'registration_fitted_again':False,
            'predictions':'Saved inference at exact keyframes only; hidden during transitions',
            'total_nodes':sum(len(g['nodes']) for g in groups),
            'total_links':sum(len(g['links']) for g in groups)}
    save(destination,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('outputs/gaussian-keyframes-20261026-v2'))
    p.add_argument('--destination',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-keyframe-comparison-20261026/tour-routes.json'))
    args=p.parse_args();r=build(args.run,args.destination)
    print({g['id']:{'nodes':len(g['nodes']),'links':len(g['links']),'segments':len(g['segments'])} for g in r['groups']})
