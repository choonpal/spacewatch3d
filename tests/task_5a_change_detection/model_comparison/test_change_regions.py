"""A change event is shared across views, not re-created for each camera."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3];TOOLS=ROOT/'tools/gaussian_change'
previous=sys.modules.pop('prepare',None);sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('shared_change_registry_test',TOOLS/'build_change_regions.py')
    registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)
finally:
    sys.path.pop(0);sys.modules.pop('prepare',None)
    if previous is not None:sys.modules['prepare']=previous


def box(id,x=0.):return {'id':id,'zone':'original-zone','bounds':[[x,0,0],[x+1,1,1]]}
def mode(boxes,evidence=None):
    visibility={'v':{b['id']:{'visible':True,'support_rays':{'before':10,'after':10}} for b in boxes}}
    return registry.make_mode(boxes,visibility,[{'id':'v','camera_to_common_cv':[[1,0,0,0],[0,1,0,2],[0,0,1,3],[0,0,0,1]]}],['v'],2.,evidence or {})


class SharedChangeRegions(unittest.TestCase):
    def test_duplicate_ids_are_not_new_events_and_conflicting_payloads_are_rejected(self):
        data=box('same');result=mode([data,data.copy()])
        self.assertEqual(result['counts']['unique_events'],1)
        self.assertEqual(result['observations']['v']['event_ids'],['same'])
        with self.assertRaises(ValueError):mode([data,box('same',10)])

    def test_same_automatic_object_pair_is_one_event_even_with_different_source_ids(self):
        event={'type':'moved','class_name':'chair','automatic_match':{'reference_track':'before','source_track':'after'}}
        result=mode([box('a'),box('b')],{'a':{'event':event},'b':{'event':event}})
        self.assertEqual(result['counts']['source_candidates'],2)
        self.assertEqual(result['counts']['unique_events'],1)
        self.assertEqual(result['events']['a']['candidate_ids'],['a','b'])

    def test_nearby_different_objects_share_an_area_but_remain_separate_events(self):
        near=mode([box('a'),box('b',2)])
        self.assertEqual(near['counts']['regions'],1);self.assertEqual(near['counts']['unique_events'],2)
        far=mode([box('a'),box('b',20)])
        self.assertEqual(far['counts']['regions'],2)

    def test_five_lounge_views_share_two_events_in_one_region(self):
        folder=ROOT/'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-uncapped-tour-20261026'
        registry.validate(folder)
        result=json.loads((folder/'change-regions.json').read_text(encoding='utf-8'))
        index=result['modes']['filtered'];region=index['regions'][0]
        self.assertEqual(region['observing_keyframes'],['K1','K2','K3','K5','K6'])
        self.assertEqual(set(region['event_ids']),{'SimSaC-R12-001','SimSaC-R13-001'})
        self.assertEqual(region['representative_view'],'K3')
        self.assertEqual(index['counts'],{'source_candidates':5,'unique_events':5,'regions':3,'observation_links':13})
        self.assertTrue(all(not v['new_events_created'] for v in index['observations'].values()))
        self.assertEqual(result['modes']['raw']['counts']['source_candidates'],6)
        self.assertEqual(result['modes']['raw']['counts']['unique_events'],5)


if __name__=='__main__':unittest.main()
