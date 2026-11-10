import copy
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from spacewatch_pipeline import engine
from spacewatch_pipeline.io import ROOT,read,save
from spacewatch_pipeline.provided import checked_arrays
from spacewatch_pipeline.size_instances import assumptions,canonical_labels,instances

class SizePipelineTests(unittest.TestCase):
    def test_provided_graph_does_not_infer_or_reuse_old_scene(self):
        c=read(ROOT/'configs/dorm2_dorm3.json');engine.validate(c)
        rows=engine.stages(c);seen=set()
        for r in rows:self.assertTrue(set(r['dependencies'])<=seen);seen.add(r['name'])
        self.assertEqual([r['action'] for r in rows if r['gpu']],['render','render'])
        c['captures'][0]['planning_scene']='old.json'
        with self.assertRaises(ValueError):engine.validate(c)
    def test_dynamic_class_ids_are_matched_by_name(self):
        _,s=assumptions(read(ROOT/'configs/dorm2.json'))
        labels,inv,mapping=canonical_labels(np.array([0,1,2,255]),[dict(id=0,name='chair'),dict(id=1,name='floor'),dict(id=2,name='unmodeled')],s['priors'],{})
        self.assertEqual(labels[:2].tolist(),[6,1]);self.assertEqual(labels[-1],255)
        self.assertEqual(inv[6],0);self.assertNotIn(int(labels[2]),[p['class_id'] for p in s['priors']])
    def test_missing_and_negative_labels_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.npz'
            np.savez(p,vertices=np.eye(3),faces=np.array([[0,1,2]]),rgb=np.zeros((3,3),np.uint8),face_class_id=np.array([-1]))
            with self.assertRaises(ValueError):checked_arrays(p)
    def test_non_integer_settings_and_invalid_sizes_fail(self):
        c=read(ROOT/'configs/dorm2.json');_,s=assumptions(c)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sizes.json';c['size_segmentation']['assumptions']=str(p)
            bad=copy.deepcopy(s);bad['settings']['max_planes']=2.5;save(p,bad)
            with self.assertRaises(ValueError):assumptions(c)
            bad=copy.deepcopy(s);bad['priors'][0]['dimensions'][0]=-1;save(p,bad)
            with self.assertRaises(ValueError):assumptions(c)
    def test_all_faces_preserved_and_old_instances_never_seed_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            meshes=[trimesh.creation.box(extents=[.5,.6,.5]),trimesh.creation.box(extents=[.01,.01,.01]),trimesh.creation.box()]
            for i,m in enumerate(meshes):m.apply_translation([i*3,0,0])
            m=trimesh.util.concatenate(meshes);labels=np.repeat([6,6,255],12).astype(np.uint8)
            src=dict(vertices=m.vertices,faces=m.faces,rgb=np.full((len(m.vertices),3),128,np.uint8),face_class_id=labels,face_instance_id=np.full(36,999,np.uint32),observations=np.full(36,2))
            np.savez(root/'source.npz',**src);save(root/'classes.json',[dict(id=6,name='chair')])
            c=read(ROOT/'configs/dorm2.json');c['size_segmentation'].update(compare_geometry_only=False,sensitivity_factors=[])
            instances(dict(capture=dict(id='one'),config=c,output=str(out),dependencies={'one.semantic':dict(arrays=str(root/'source.npz'),vocabulary=str(root/'classes.json'))}))
            d=np.load(out/'mesh_arrays.npz');regions=read(out/'regions.json')
            for key in ['vertices','faces','rgb','face_class_id','observations']:np.testing.assert_array_equal(d[key],src[key])
            self.assertTrue((d['face_region_id']>0).all());self.assertEqual(sum(r['faces'] for r in regions),36)
            self.assertTrue((d['face_instance_id'][:12]>0).all());self.assertFalse(d['face_instance_id'][12:].any())
            self.assertNotIn(999,d['face_instance_id']);np.testing.assert_array_equal(d['upstream_face_instance_id'],src['face_instance_id'])
    def test_resume_checks_external_receipts_and_code_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=root/'config.json';out=root/'run';external=root/'input.txt';external.write_text('original')
            c=read(ROOT/'configs/dorm2.json');c['runtime']['allowed_hostname']=socket.gethostname();save(cfg,c)
            specs=[dict(name='one.mesh',action='mesh',dependencies=[],gpu=False),dict(name='publish',action='publish',dependencies=['one.mesh'],gpu=False)]
            def worker(command,env,log,timeout):
                job=read(command[-1]);save(Path(job['output'])/'result.json',dict(external_artifacts=[str(external)]))
            with patch.object(engine,'stages',return_value=specs),patch.object(engine,'run_worker',side_effect=worker) as run,patch.object(engine,'code_identity',return_value={'code':'v5'}):
                engine.execute(cfg,out,stop_after='one.mesh');self.assertEqual(run.call_count,1)
                engine.execute(cfg,out);self.assertEqual(run.call_count,2);self.assertEqual(read(out/'status.json')['status'],'complete')
                engine.execute(cfg,out);self.assertEqual(run.call_count,2)
                with patch.object(engine,'code_identity',return_value={'code':'changed'}):
                    with self.assertRaises(ValueError):engine.execute(cfg,out)
                external.write_text('changed')
                with self.assertRaises(ValueError):engine.execute(cfg,out)

if __name__=='__main__':unittest.main()
