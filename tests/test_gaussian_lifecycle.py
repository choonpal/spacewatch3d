import unittest
import numpy as np
from tools.mesh_simsac.event_matching import decide_unmatched, movement_rejections, unique_assignments
from tools.gaussian_change.lifecycle import groups, project, independent, retain_motion_origin_hypotheses


def free_record(index):
    return dict(frame_id=str(index), usable=True, observed_pixels=50, free_fraction=.9, occluded_fraction=0., occupied_fraction=0.)


class LifecycleTests(unittest.TestCase):
    def test_missing_counterpart_does_not_imply_removal_without_observed_free_space(self):
        self.assertEqual(decide_unmatched(0,True,3,[],False)[0], 'unresolved')
        self.assertEqual(decide_unmatched(0,True,3,[free_record(1),free_record(2)],False)[0], 'removed')
        self.assertEqual(decide_unmatched(1,True,3,[free_record(1),free_record(2)],False)[0], 'added')
        self.assertEqual(decide_unmatched(1,True,3,[free_record(1),free_record(2)],True)[0], 'unresolved')

    def test_movement_requires_identity_both_vacancies_and_displacement_beyond_error(self):
        pair=dict(shape_f1=.9,directional_coverage=[.9,.9],appearance=.95,frame_support=3,margin=.15,displacement_u=1.,change_supported=True)
        free=[free_record(1),free_record(2)]
        self.assertEqual(movement_rejections(pair,free,free,.02,.02),[])
        self.assertIn('origin_vacancy_unobserved',movement_rejections(pair,[],free,.02,.02))
        self.assertIn('displacement_within_map_error',movement_rejections({**pair,'displacement_u':.02},free,free,.02,.02))

    def test_one_to_one_assignment_does_not_reuse_the_same_object(self):
        pairs=[dict(before='A',after='X',score=.9),dict(before='A',after='Y',score=.8),dict(before='B',after='X',score=.7)]
        selected=unique_assignments(pairs)
        self.assertEqual(len({p['before'] for p in selected}),len(selected))
        self.assertEqual(len({p['after'] for p in selected}),len(selected))

    def test_semantic_face_connectivity_and_cv_panorama_projection(self):
        labels=groups(np.array([[0,1,2],[2,3,4],[6,7,8]]))
        self.assertEqual(labels[0],labels[1]);self.assertNotEqual(labels[0],labels[2])
        u,v,d=project(np.array([[0,0,2],[2,0,0]]),np.eye(4),(100,200))
        np.testing.assert_array_equal(u,[100,150]);np.testing.assert_array_equal(v,[50,50]);np.testing.assert_allclose(d,2)

    def test_ambiguous_origins_are_retained_without_bypassing_physical_gates(self):
        objects={key:dict(bounds=[[0,0,0],[1,1,1]],center=[0,0,0],_points=np.ones((4,3))) for key in ['A','B','C']}
        pairs=[dict(before=key,after='X',appearance=score,shape_f1=.8,rejection_reasons=['ambiguous_assignment']) for key,score in [('A',.85),('B',.9)]]
        pairs.append(dict(before='C',after='Y',appearance=.99,shape_f1=.99,rejection_reasons=['destination_vacancy_unobserved','ambiguous_assignment']))
        result=retain_motion_origin_hypotheses(pairs,[],objects)
        self.assertEqual(len(result),1)
        self.assertTrue(result[0]['origin_identity_ambiguous'])
        self.assertEqual(result[0]['before'],'B')
        self.assertEqual({o['object_id'] for o in result[0]['origin_candidates']},{'A','B'})
        self.assertEqual(retain_motion_origin_hypotheses(pairs,[dict(before='A',after='X')],objects),[dict(before='A',after='X')])


if __name__=='__main__':unittest.main()
