"""Automatic matching must preserve static objects and reject ambiguous moves."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3];TOOLS=ROOT/'tools/gaussian_change'
previous=sys.modules.pop('prepare',None);sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('automatic_correspondence_test',TOOLS/'automatic_object_matching.py')
    matching=importlib.util.module_from_spec(spec);spec.loader.exec_module(matching)
finally:
    sys.path.pop(0);sys.modules.pop('prepare',None)
    if previous is not None:sys.modules['prepare']=previous


class AutomaticMatching(unittest.TestCase):
    def test_upright_rigid_motion_is_recovered_without_object_scale_fitting(self):
        points=np.random.default_rng(42).uniform([-.3,0,-.2],[.3,.8,.2],(200,3))
        angle=np.pi/2;rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        moved=points@rotation.T+[2.,.03,-3.]
        fit=matching.rigid_shape_fit(points,moved,2.)
        self.assertGreater(fit['shape_f1'],.99)
        matrix=np.asarray(fit['reference_to_source_rigid'])
        self.assertAlmostEqual(np.linalg.det(matrix[:3,:3]),1.,places=7)
        self.assertFalse(fit['object_scale_fitted'])
        np.testing.assert_allclose(matching.transform(points,matrix),moved,atol=1e-6)

    def test_an_object_that_remains_in_place_cannot_be_stolen_for_a_move(self):
        pair={'shape_f1':.9,'directional_coverage':[.9,.9],'displacement_u':2.,
              'origin_shape_gain':.3,'destination_shape_gain':.3,'destination_vacancy_views':3,'same_position_partner':None}
        self.assertEqual(matching.assignment_gate(pair,2.,.1),[])
        for key,value,reason in [
            ('same_position_partner','static-counterpart','already_stationary_matched'),
            ('origin_shape_gain',.01,'origin_surface_still_present'),
            ('destination_shape_gain',.01,'destination_surface_already_present'),
            ('destination_vacancy_views',0,'destination_vacancy_unsupported'),
            ('displacement_u',.1,'within_registration_error')]:
            with self.subTest(reason=reason):self.assertIn(reason,matching.assignment_gate({**pair,key:value},2.,.1))

    def test_tied_origins_remain_ambiguous_on_repeated_runs(self):
        pairs=[{'reference':a,'source':'destination','shape_f1':.8,'rejection_reasons':[]} for a in ['origin-A','origin-B']]
        self.assertEqual(matching.select_assignments(pairs),[])
        self.assertTrue(all('ambiguous_assignment' in p['rejection_reasons'] for p in pairs))
        self.assertEqual(matching.select_assignments(pairs),[])

    def test_assignment_is_one_to_one_with_unmatched_objects_allowed(self):
        pairs=[{'reference':a,'source':b,'shape_f1':f,'rejection_reasons':[]} for a,b,f in
               [('a','x',.9),('a','y',.7),('b','z',.85)]]
        accepted=matching.select_assignments(pairs)
        self.assertEqual({(p['reference'],p['source']) for p in accepted},{('a','x'),('b','z')})
        self.assertEqual(matching.select_assignments([]),[])

    def test_automatic_r12_result_has_a_partitioned_origin_and_rejects_static_chairs(self):
        folder=ROOT/'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-uncapped-tour-20261026'
        result=json.loads((folder/'automatic-matching.json').read_text(encoding='utf-8'))
        evidence=json.loads((folder/'object-evidence.json').read_text(encoding='utf-8'))
        pair=next(p for p in result['accepted_matches'] if p['reference_track']=='I0-005')
        self.assertEqual(pair['source_track'],'I1-067');self.assertGreater(pair['shape_f1'],.65)
        self.assertGreater(pair['destination_vacancy_views'],1)
        self.assertEqual(pair['rejection_reasons'],[])
        protected=[p for p in result['comparisons'] if p['source_track']=='I1-067' and p['reference_track'] in ['I0-004','I0-010','I0-011','I0-013']]
        self.assertEqual(len(protected),4)
        self.assertTrue(all('already_stationary_matched' in p['rejection_reasons'] for p in protected))
        siblings=[o for o in evidence['objects'].values() if o.get('parent_instance')=='S0-010']
        seen=set()
        for obj in siblings:
            geo=json.loads((folder/obj['mesh_file']).read_text(encoding='utf-8'))
            ids=set(geo['original_face_ids']);self.assertFalse(seen&ids);seen|=ids
        self.assertEqual(len(seen),758)
        self.assertTrue(result['automatic']);self.assertFalse(result['manual_overrides_used'])


if __name__=='__main__':unittest.main()
