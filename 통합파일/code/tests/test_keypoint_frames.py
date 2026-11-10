import unittest
import numpy as np
from tools.mesh_simsac.keypoint_frames import select_nearby,grouped_references

class KeypointFramesTests(unittest.TestCase):
    def frame(self,i,p,sharp=1):return dict(id=str(i),common_position=p,sharpness=sharp)
    def test_route_order_does_not_determine_location(self):
        fs=[self.frame(0,[20,0,0],999),self.frame(8,[.1,0,0]),self.frame(30,[.3,0,0])]
        self.assertEqual({x['id'] for x in select_nearby(fs,[0,0,0],1)}, {'8','30'})
    def test_missing_route_has_no_nearest_far_fallback(self):
        self.assertEqual(select_nearby([self.frame(0,[5,0,0])],[0,0,0],1),[])
    def test_height_counts_and_budget_is_bounded(self):
        fs=[self.frame(0,[0,10,0]),*[self.frame(i,[i/100,0,0]) for i in range(1,8)]]
        got=select_nearby(fs,[0,0,0],1,3)
        self.assertEqual(len(got),3);self.assertNotIn('0',[x['id'] for x in got])
    def test_exact_distance_does_not_depend_on_heading(self):
        fs=[dict(self.frame(1,[0,0,0]),heading=180)]
        self.assertEqual(select_nearby(fs,[0,0,0],1)[0]['distance_u'],0)
    def test_nonfinite_and_bad_settings(self):
        self.assertEqual(select_nearby([self.frame(1,[np.nan,0,0])],[0,0,0],1),[])
        with self.assertRaises(ValueError):select_nearby([], [0,0,0],0)
    def test_reference_scope_does_not_cross_keypoints(self):
        def ref(i,frame,k):return dict(id=i,frame_id=frame,keypoint_ids=k)
        choices=[(1,ref('unrelated','8',['C']),1,0),(.9,ref('a1','1',['A']),.9,0),(.8,ref('a2','1',['A']),.8,0),(.7,ref('ab','2',['A','B']),.7,0),(.6,ref('b','3',['B']),.6,0)]
        got=grouped_references(choices,['A','B'],2);lookup={x['id']:x for x in got}
        self.assertEqual(set(lookup),{'a1','ab','b'})
        self.assertEqual(lookup['ab']['keypoint_ids'],['A','B'])
        self.assertNotIn('a2',lookup)
    def test_production_pairing_rejects_unrelated_keypoint(self):
        from tools.mesh_simsac.core import choose_pairs,intrinsics,rays
        K=intrinsics(8,8);points=rays(K,(8,8))*2
        def v(i,e,k):return dict(id=i,epoch=e,frame_id=i,keypoint_ids=k,valid=np.ones((8,8),bool),person=np.zeros((8,8),bool),points=points,depth=np.full((8,8),2.),pose=np.eye(4),K=K)
        rows=choose_pairs([v('query',0,['A']),v('wrong',1,['B']),v('right',1,['A'])],.15)
        refs=next(x['refs'] for x in rows if x['query']=='query')
        self.assertEqual([x['id'] for x in refs],['right']);self.assertEqual(refs[0]['keypoint_ids'],['A'])
    def test_production_pairing_keeps_unobserved_point_without_match(self):
        from tools.mesh_simsac.core import choose_pairs,intrinsics,rays
        K=intrinsics(8,8);points=rays(K,(8,8))*2
        def v(i,e,k):return dict(id=i,epoch=e,frame_id=i,keypoint_ids=k,valid=np.ones((8,8),bool),person=np.zeros((8,8),bool),points=points,depth=np.full((8,8),2.),pose=np.eye(4),K=K)
        rows=choose_pairs([v('q',0,['A']),v('r',1,['B'])],.15)
        self.assertEqual(rows[0]['refs'],[]);self.assertEqual(rows[0]['state'],'unobserved_counterpart')

if __name__=='__main__':unittest.main()
