import unittest
from tools.gaussian_change.comparison import compare_events


class ComparisonTests(unittest.TestCase):
    def test_lost_detection_is_not_counted_as_classification_success(self):
        old={'events':[dict(id='old',type='unresolved',reason='missing',bounds=[[0,0,0],[1,1,1]],object_ids=[])],'objects':[]}
        new={'events':[],'objects':[],'excluded_surface_candidates':[]}
        rows=compare_events(old,new,[],[],2.)
        self.assertEqual(rows[0]['outcome'],'no_corresponding_detection')

    def test_shared_observations_match_the_object_even_when_event_ids_or_bounds_change(self):
        old={'events':[dict(id='a',type='unresolved',reason='identity',bounds=[[0,0,0],[1,1,1]],object_ids=['x'])],'objects':[dict(id='x',source_observation_ids=['p','q'])]}
        new={'events':[dict(id='b',type='moved',reason='matched',bounds=[[8,8,8],[9,9,9]],object_ids=['y'])],'objects':[dict(id='y',source_observation_ids=['p','q'])]}
        rows=compare_events(old,new,[],[],2.)
        self.assertEqual(rows[0]['outcome'],'classified');self.assertEqual(rows[0]['current_matches'][0]['id'],'b')

    def test_mixed_match_does_not_hide_remaining_unresolved_evidence(self):
        old={'events':[dict(id='a',type='unresolved',reason='identity',bounds=[[0,0,0],[1,1,1]],object_ids=[])],'objects':[]}
        new={'events':[dict(id=i,type=t,reason='matching',bounds=[[0,0,0],[1,1,1]],object_ids=[]) for i,t in [('b','moved'),('c','unresolved')]],'objects':[]}
        self.assertEqual(compare_events(old,new,[],[],2.)[0]['outcome'],'partly_classified')


if __name__=='__main__':unittest.main()
