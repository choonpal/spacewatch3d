"""Object segmentation supports review types without turning pixels into IDs."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3];TOOLS=ROOT/'tools/gaussian_change'
previous=sys.modules.pop('prepare',None);sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('gaussian_object_hints',TOOLS/'link_segmented_objects.py')
    hints=importlib.util.module_from_spec(spec);spec.loader.exec_module(hints)
finally:
    sys.path.pop(0);sys.modules.pop('prepare',None)
    if previous is not None:sys.modules['prepare']=previous


def obj(id,epoch,name='chair',quality=1.,votes=0):
    return {'id':id,'epoch':epoch,'class_name':name,'multi_observation_fraction':quality,
            'intersection_fraction':.5,'tracks':[{'supported':True,'vacancy_views':votes}]}


class EventHints(unittest.TestCase):
    def test_both_epoch_presence_prevents_false_removal_from_before_only_pixels(self):
        result=hints.review_type([obj('before',0,'windowpane'),obj('after',1,'windowpane')],[],[],2,.1)
        self.assertEqual(result['type'],'unresolved');self.assertFalse(result['physical_event_confirmed'])
        self.assertFalse(hints.POLICY['uses_simsac_pixel_direction_for_type'])

    def test_movement_requires_geometry_correspondence_and_distance_above_noise(self):
        objects=[obj('before',0),obj('after',1)]
        pair={'reference':'before','source':'after','shape_f1':.8,'displacement_u':1.,'already_paired_with_other':None}
        self.assertEqual(hints.review_type(objects,[pair],[],2,.1)['type'],'moved')
        for change in [{'displacement_u':.1},{'shape_f1':.3},{'already_paired_with_other':'different-object'}]:
            with self.subTest(change=change):self.assertEqual(hints.review_type(objects,[{**pair,**change}],[],2,.1)['type'],'unresolved')

    def test_after_chair_is_not_hidden_by_before_cabinet_context(self):
        objects=[obj('cabinet-before',0,'cabinet'),obj('cabinet-after',1,'cabinet'),obj('chair-after',1,votes=3)]
        result=hints.review_type(objects,[],[],2,.1)
        self.assertEqual(result['type'],'added');self.assertEqual(result['class_name'],'chair')
        self.assertEqual(result['confidence'],'occupancy_supported');self.assertIn('moved',result['alternatives'])

    def test_missing_or_weak_segmentation_is_deferred_and_unconfirmed_absence_is_explicit(self):
        for objects in [[],[obj('weak',1,quality=.2)]]:
            self.assertEqual(hints.review_type(objects,[],[],2,.1)['type'],'unresolved')
        removed=hints.review_type([obj('before',0,'door')],[],[],2,.1)
        self.assertEqual(removed['type'],'removed');self.assertEqual(removed['confidence'],'segmentation_only')
        self.assertFalse(removed['physical_event_confirmed'])

    def test_origin_outside_the_change_box_prevents_a_false_addition(self):
        after={**obj('after',1,votes=3),'faces':137,'zone':'R12','class_label':'의자'}
        outside={**obj('merged-before',0),'faces':758,'zone':'R12','class_label':'의자'}
        result=hints.classify_with_context([after],[],[],2,.1,[outside])
        self.assertEqual(result['type'],'unresolved')
        self.assertEqual(result['confidence'],'ambiguous_correspondence')
        self.assertFalse(result['physical_event_confirmed'])

    def test_motion_requires_an_accepted_automatic_correspondence(self):
        after={**obj('after',1,votes=3),'faces':137,'zone':'any-zone','class_label':'의자'}
        outside={**obj('split-before',0),'faces':135,'zone':'any-zone','class_label':'의자','automatic_partition':True}
        pair={'reference':'split-before','source':'after','shape_f1':.8,'displacement_u':2.,
              'origin_shape_gain':.25,'destination_shape_gain':.2,'destination_vacancy_views':3,'rejection_reasons':[]}
        result=hints.classify_with_context([after],[],[],2,.1,[outside],[pair])
        self.assertEqual(result['type'],'moved');self.assertEqual(result['confidence'],'automatic_instance_correspondence')
        self.assertTrue(result['automatic']);self.assertFalse(result['manual_overrides_used'])
        self.assertEqual(result['movement']['origin_scope'],'automatic_mask_partition')
        rejected={**pair,'rejection_reasons':['ambiguous_assignment']}
        self.assertEqual(hints.classify_with_context([after],[],[],2,.1,[outside],[rejected])['type'],'unresolved')
        for bad in [{**outside,'multi_observation_fraction':.1},{**outside,'epoch':1},
                    {**outside,'zone':'different-zone'},{**outside,'class_name':'table'}]:
            with self.subTest(bad=bad):self.assertNotEqual(hints.classify_with_context([after],[],[],2,.1,[bad],[pair])['type'],'moved')

    def test_current_snapshot_keeps_candidates_and_exports_original_mesh_faces(self):
        folder=ROOT/'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-uncapped-tour-20261026'
        report=hints.validate(folder,metadata_only=True)
        self.assertTrue(report['original_simsac_regions_unchanged']);self.assertEqual(report['segmented_meshes'],44)
        evidence=json.loads((folder/'object-evidence.json').read_text(encoding='utf-8'))
        r=evidence['regions']['filtered']
        self.assertEqual(r['SimSaC-R01-001']['event']['type'],'unresolved')
        self.assertEqual(r['SimSaC-R08-001']['event']['type'],'moved')
        moved=r['SimSaC-R12-001']['event']
        self.assertEqual(moved['type'],'moved');self.assertEqual(moved['confidence'],'automatic_instance_correspondence')
        self.assertEqual(moved['selected_pair'],['S0-010@I0-005','S1-024'])
        self.assertEqual(moved['movement']['origin_scope'],'automatic_mask_partition')
        self.assertEqual(r['SimSaC-R13-001']['event']['confidence'],'segmentation_only')
        self.assertEqual(r['SimSaC-R08-002']['event']['type'],'unresolved')
        self.assertTrue(report['automatic']);self.assertFalse(report['manual_overrides_used'])
        self.assertFalse((folder/'event-review-overrides.json').exists())
        self.assertNotIn('review',moved)
        self.assertEqual(evidence['objects'][moved['selected_pair'][0]]['faces'],135)


if __name__=='__main__':unittest.main()
