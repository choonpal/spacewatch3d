"""Saved paths and fixed Gaussian-RGB regions must survive the tour bridge."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tools/gaussian_change'
old_prepare=sys.modules.pop('prepare',None);sys.path.insert(0,str(TOOLS))
try:
    modules=[]
    for name in ['build_tour_routes','build_tour_candidates']:
        spec=importlib.util.spec_from_file_location('gaussian_'+name,TOOLS/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    routes,overlay=modules
finally:
    sys.path.pop(0);sys.modules.pop('prepare',None)
    if old_prepare is not None:sys.modules['prepare']=old_prepare


class TourBridge(unittest.TestCase):
    def test_saved_order_and_missing_links_are_preserved(self):
        views=[]
        for i in [1,2,3,4]:
            pose=np.eye(4);pose[:3,3]=[i,2,0]
            views.append({'id':str(i),'viewpoint_id':'v'+str(i),'camera_to_common_cv':pose.tolist()})
        plan={'route_id':'route','tour_visits':[{'visit_id':'visit'+str(i),'viewpoint_id':'v'+str(i),'component_id':'one' if i!=4 else 'two'} for i in [1,3,2,4]],
              'tour_links':[{'from_viewpoint':'v1','to_viewpoint':'v3','component_id':'one','polyline_m':[[1,2,0],[1,2,5],[3,2,5],[3,2,0]]},
                            {'from_viewpoint':'v3','to_viewpoint':'v2','component_id':'one','polyline_m':[[3,2,0],[2,2,0]]}]}
        result=routes.convert_plan(plan,views,lambda points,view:np.asarray(points))
        self.assertEqual([n['id'] for n in result['nodes']],['1','3','2','4'])
        self.assertEqual([s['nodes'] for s in result['segments']],[['1','3','2'],['4']])
        self.assertEqual(result['links'][0]['polyline'][1],[1,2,5])
        self.assertEqual(result['missing_visit_connections'],1)
        plan['tour_links'][0]['polyline_m'][-1][0]+=.01
        with self.assertRaisesRegex(ValueError,'endpoint'):routes.convert_plan(plan,views,lambda points,view:np.asarray(points))

    def test_depth_support_rejects_occlusion_unknown_and_empty_space(self):
        shape=(64,128);box=[[-.5,-.5,4.5],[.5,.5,5.5]];camera=np.eye(4)
        def geometry(depth,alpha=1):return {'depth':np.full(shape,depth),'alpha':np.full(shape,alpha)}
        self.assertGreaterEqual(overlay.support(box,camera,geometry(5),2),3)
        self.assertEqual(overlay.support(box,camera,geometry(2),2),0)
        self.assertEqual(overlay.support(box,camera,geometry(9),2),0)
        self.assertEqual(overlay.support(box,camera,geometry(5,0),2),0)
        self.assertEqual(overlay.support(box,camera,geometry(float('nan')),2),0)

    def test_snapshot_keeps_original_regions_and_link_camera_endpoints(self):
        base=ROOT/'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025'
        folder=base/'gaussian-keyframe-comparison-20261026'
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        fixed=read(folder/'tour-candidates.json');source=read(base/'gaussian-map-comparison-20261025/simsac-candidates.json')
        self.assertEqual(fixed['boxes'],source['boxes'])
        self.assertFalse(fixed['candidate_geometry_changed']);self.assertFalse(fixed['new_inference'])
        data=read(folder/'gaussian-data.json');by_id={v['id']:v for v in data['zones']}
        graph=read(folder/'tour-routes.json');self.assertEqual(graph['total_nodes'],46);self.assertEqual(graph['total_links'],31)
        self.assertFalse(graph['new_links_created'])
        for group in graph['groups']:
            self.assertEqual([n['id'] for n in group['nodes']],[id for s in group['segments'] for id in s['nodes']])
            for edge in group['links']:
                for i,key in [(0,'from'),(-1,'to')]:
                    np.testing.assert_allclose(edge['polyline'][i],np.asarray(by_id[edge[key]]['camera_to_common_cv'])[:3,3],atol=1e-7)
        # Two separate keyframes must keep the same region, rather than replacing
        # it with a new keyframe-local SimSaC component.
        for keyframe in ['P3-007','P3-008']:
            self.assertTrue(fixed['visibility'][keyframe]['filtered']['SimSaC-R08-001']['visible'])
        self.assertFalse(fixed['visibility']['P3-005']['filtered']['SimSaC-R08-001']['visible'])


if __name__=='__main__':unittest.main()
