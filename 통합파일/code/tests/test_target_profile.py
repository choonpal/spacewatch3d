import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from spacewatch_pipeline.io import read,save,sha,ROOT
from spacewatch_pipeline.target_profile import array_digest,checked_profile,target_ids,instances
from spacewatch_pipeline.engine import validate,stages


class TargetProfileTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'profile';folder.mkdir()
        data=dict(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
            faces=np.array([[0,1,2],[0,1,3]]),rgb=np.full((4,3),128,np.uint8),face_class_id=np.array([3,255],np.uint16))
        np.savez(root/'input.npz',**data);save(root/'coordinates.json',{'units':'m'});save(root/'classes.json',[dict(id=3,name='door')])
        scene=dict(scene_id='test',capture_id='one',coordinates=dict(frame_id='one_mesh'),objects=[dict(instance_id='semantic_door')])
        save(folder/'scene.json',scene);save(folder/'importance_cache.json',dict(scene_id='test',capture_id='one',scores={'semantic_door':dict(score=.5)}))
        save(folder/'input_binding.json',dict(capture_id='one',frame_id='one_mesh',coordinates_sha256=sha(root/'coordinates.json'),
            source_run=str(root/'source'),arrays={k:array_digest(v) for k,v in data.items()}))
        np.savez(folder/'face_roles.npz',source_face_id=np.arange(2),source_class_id=data['face_class_id'],target_region_index=np.array([1,0],np.uint32))
        save(folder/'target_mapping.json',[dict(region_id=1,instance_id='semantic_door',class_id=3,class_name='door',faces=1)])
        save(folder/'config.json',dict(require_all_objects=False));save(folder/'filter.json',dict(excluded_components=1))
        cap=dict(id='one',source_run=str(root/'source'),coordinates=str(root/'coordinates.json'),semantic_frame_id='one_mesh',semantic_arrays=str(root/'input.npz'))
        def seal():
            save(folder/'checksums.json',{p.name:sha(p) for p in folder.iterdir() if p.name!='checksums.json'})
            cap['planning_profile']=dict(path=str(folder),sha256=sha(folder/'checksums.json'))
        seal()
        return folder,data,cap,seal

    def test_valid_binding_and_reject_changed_array(self):
        with tempfile.TemporaryDirectory() as name:
            _,data,cap,_=self.fixture(Path(name))
            self.assertEqual(len(checked_profile({},cap,data)[2]),1)
            changed=copy.deepcopy(data);changed['vertices'][0,0]=.1
            with self.assertRaisesRegex(ValueError,'input differs'):checked_profile({},cap,changed)

    def test_reject_changed_coordinates_and_asset(self):
        with tempfile.TemporaryDirectory() as name:
            folder,data,cap,_=self.fixture(Path(name))
            (folder/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'asset changed'):checked_profile({},cap,data)
        with tempfile.TemporaryDirectory() as name:
            _,data,cap,_=self.fixture(Path(name));Path(cap['coordinates']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'coordinate/source mismatch'):checked_profile({},cap,data)

    def test_reject_target_score_and_face_correspondence_mismatch(self):
        with tempfile.TemporaryDirectory() as name:
            folder,data,cap,seal=self.fixture(Path(name));scores=read(folder/'importance_cache.json');scores['scores']={};save(folder/'importance_cache.json',scores);seal()
            with self.assertRaisesRegex(ValueError,'score/target'):checked_profile({},cap,data)
        with tempfile.TemporaryDirectory() as name:
            folder,data,cap,seal=self.fixture(Path(name));np.savez(folder/'face_roles.npz',source_face_id=np.array([1,0]),source_class_id=data['face_class_id'],target_region_index=np.array([1,0]));seal()
            with self.assertRaisesRegex(ValueError,'face correspondence'):checked_profile({},cap,data)

    def test_instance_adapter_preserves_all_geometry_and_target_ids(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);_,data,cap,_=self.fixture(root);out=root/'out';out.mkdir()
            instances(dict(capture=cap,config={},output=str(out),dependencies={'one.semantic':dict(arrays=cap['semantic_arrays'],vocabulary=str(root/'classes.json'))}))
            d=np.load(out/'mesh_arrays.npz')
            for key,value in data.items():np.testing.assert_array_equal(d[key],value)
            self.assertEqual(d['face_instance_id'].tolist(),[1,0]);self.assertTrue((d['face_region_id']>0).all())
            self.assertEqual(target_ids(read(out/'regions.json')),{'semantic_door'})
            self.assertEqual(read(out/'summary.json')['discarded_faces'],0)

    def test_bound_inputs_share_clearance_policy(self):
        cfg=read(ROOT/'configs/dorm2_dorm3.json');validate(cfg)
        self.assertIn('planning_profile',cfg['captures'][0]);self.assertEqual(cfg['captures'][0]['expected_candidates'],100)
        self.assertIn('planning_profile',cfg['captures'][1]);self.assertEqual(cfg['captures'][1]['expected_candidates'],93)
        self.assertEqual([r['action'] for r in stages(cfg) if r['gpu']],['render','render'])
        bad=copy.deepcopy(cfg);bad['captures'][1]['planning_profile']['sha256']='bad'
        with self.assertRaisesRegex(ValueError,'digest'):validate(bad)

    def test_declared_default_scores_and_reject_bad_default(self):
        with tempfile.TemporaryDirectory() as name:
            folder,data,cap,seal=self.fixture(Path(name))
            scene=read(folder/'scene.json');scene['objects'][0]['class_name']='door';save(folder/'scene.json',scene)
            cfg=read(folder/'config.json');cfg['default_importance']=.5;save(folder/'config.json',cfg)
            scores=read(folder/'importance_cache.json');scores.update(scores={},default_classes=['door'],default_importance=.5)
            save(folder/'importance_cache.json',scores);seal()
            self.assertEqual(len(checked_profile({},cap,data)[2]),1)
            for change in [dict(default_classes=['chair']),dict(default_importance=.7),dict(default_classes=['door','door']),dict(scores={'foreign':{'score':.5}})]:
                bad={**scores,**change};save(folder/'importance_cache.json',bad);seal()
                with self.assertRaisesRegex(ValueError,'score/target'):checked_profile({},cap,data)


if __name__=='__main__':unittest.main()
