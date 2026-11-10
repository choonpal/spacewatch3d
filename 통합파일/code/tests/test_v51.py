import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from spacewatch_pipeline.io import ROOT, save, read, record, sha
from spacewatch_pipeline.engine import run_dag
from spacewatch_pipeline.florence import aggregate_votes_reference
from spacewatch_pipeline.semantic_aggregation import aggregate
from spacewatch_pipeline.correspondence import ExportCorrespondence

sys.path[:0] = [str(ROOT/'vendor/spacewatch3d_src'), str(ROOT/'vendor/native')]
import objective_executor as oe
from spacewatch3d.tasks.task_5b_keyframes.selection import Objective
from spacewatch3d.tasks.task_5b_keyframes.models import Config


def worker_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'vendor/panovggt_pipeline_v2/workers/{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class PlanningCacheTests(unittest.TestCase):
    def tearDown(self):
        oe.restore()

    def make(self):
        rng = np.random.default_rng(42)
        return Objective(rng.random((24,200))>.7, rng.random((24,3)), rng.random(200)+.1,
            np.array(['a','b']*100), [dict(instance_id=str(i),importance=.7,must_see=i==0) for i in range(3)],
            Config(), ['unassessed'])

    def test_all_report_fields_and_mutation_isolation(self):
        baseline = self.make(); oe.install(immutable=True); fast = self.make()
        for selected in [[],[0],[0,1,2],list(range(24))]:
            expected = oe.ORIGINAL(baseline, selected)
            self.assertEqual(expected, fast.measure(selected))
            result = fast.measure(selected); result[1]['regions'][0]['covered_area_m2'] = -1
            result[1]['object_best_quality']['0'] = -1
            result[1]['unmet_targets'].append('changed')
            result[1]['object_observation_requirements']['unobserved_required_object_ids'].append('changed')
            self.assertEqual(expected, fast.measure(list(reversed(selected))))
        with self.assertRaises(ValueError): fast.weights[0] = 5

    def test_shallow_copies_and_replaced_matrices_invalidate(self):
        oe.install(immutable=True); fast = self.make(); fast.measure([0])
        guard = copy.copy(fast); guard.object_thresholds = np.full(3, .99)
        self.assertEqual(oe.ORIGINAL(guard,[0]), guard.measure([0]))
        lateral = copy.copy(fast); lateral.visible = np.ones_like(fast.visible)
        self.assertEqual(oe.ORIGINAL(lateral,[0]), lateral.measure([0]))
        self.assertEqual(oe.ORIGINAL(fast,[0]), fast.measure([0]))

    def test_mutable_compatibility_mode(self):
        oe.install(); obj = self.make(); obj.measure([0]); obj.weights[0] = 99
        self.assertEqual(oe.ORIGINAL(obj,[0]), obj.measure([0]))

    def test_lateral_visibility_outside_original_observable_mask(self):
        oe.install(immutable=True); obj = self.make()
        obj.observable = np.arange(len(obj.weights)) % 3 == 0
        obj.visible = np.ones_like(obj.visible)
        for selected in [[], [0], [0, 1]]:
            self.assertEqual(oe.ORIGINAL(obj, selected), obj.measure(selected))
        # A shallow copy can restore ordinary visibility without inheriting an
        # invalid denominator or a cached shortcut from the expanded objective.
        changed = copy.copy(obj); changed.observable = np.ones(len(obj.weights), dtype=bool)
        self.assertEqual(oe.ORIGINAL(changed, [0]), changed.measure([0]))


class SchedulerTests(unittest.TestCase):
    def test_overlap_dependency_order_and_one_gpu(self):
        specs = [dict(name='prepare',action='gs_prepare',dependencies=[],gpu=False),
            dict(name='plan',action='plan',dependencies=[],gpu=False),
            dict(name='train',action='gs_train',dependencies=['prepare'],gpu=True),
            dict(name='render',action='render',dependencies=['train','plan'],gpu=True)]
        barrier = threading.Barrier(2); committed=[]; active=0; peak=0; lock=threading.Lock()
        def task(spec,cancel,events):
            nonlocal active,peak
            if spec['gpu']:
                with lock: active+=1; peak=max(peak,active)
            if spec['name'] in ['train','plan']: barrier.wait(timeout=3)
            if spec['gpu']:
                with lock: active-=1
            return spec['name']
        def commit(spec,value):
            self.assertTrue(set(spec['dependencies']) <= set(committed)); committed.append(value)
        run_dag(specs,task,commit,lambda *a:None)
        self.assertEqual(committed[0],'prepare'); self.assertEqual(committed[-1],'render'); self.assertEqual(peak,1)

    def test_failure_cancels_and_drains_other_worker(self):
        started=threading.Event(); drained=threading.Event()
        specs=[dict(name='cpu',action='plan',dependencies=[],gpu=False),dict(name='gpu',action='train',dependencies=[],gpu=True)]
        def task(spec,cancel,events):
            if spec['gpu']:
                started.set(); cancel.wait(3); drained.set(); return None
            self.assertTrue(started.wait(3)); raise ValueError('test failure')
        with self.assertRaisesRegex(ValueError,'test failure'):
            run_dag(specs,task,lambda *a:None,lambda *a:None)
        self.assertTrue(drained.is_set())

    def test_serial_switch_and_cycle_detection(self):
        specs=[dict(name=str(i),action='test',dependencies=[],gpu=bool(i%2)) for i in range(4)]
        order=[]
        run_dag(specs,lambda s,*a:s['name'],lambda s,v:order.append(v),lambda *a:None,overlap=False)
        self.assertEqual(order,['0','1','2','3'])
        specs[0]['dependencies']=['missing']
        with self.assertRaisesRegex(ValueError,'cycle'):
            run_dag(specs,lambda *a:None,lambda *a:None,lambda *a:None)


class AggregationTests(unittest.TestCase):
    def test_randomized_exact_reference_parity(self):
        rng=np.random.default_rng(419)
        for n in [0,1,50,10000]:
            face=rng.integers(0,100,n); frame=rng.integers(0,13,n); label=rng.integers(0,17,n)
            for minimum in [1,2,5]:
                a=aggregate_votes_reference(face,frame,label,103,255,minimum,minimum_pixels=2)
                b=aggregate(face,frame,label,103,255,minimum,minimum_pixels=2)
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)


class CorrespondenceTests(unittest.TestCase):
    def test_export_row_order_and_changed_input_rejection(self):
        Writer=worker_module('correspondence').CorrespondenceWriter
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'unfiltered_result';out.mkdir();work=root/'work';work.mkdir()
            (root/'windows').mkdir(); (root/'windows/window_000.npz').write_bytes(b'window')
            (root/'prepared.npz').write_bytes(b'prepared');(root/'optimized.npz').write_bytes(b'optimized')
            save(root/'config.json',dict(output=dict(ply_name='map.ply')))
            cloud=out/'map.ply';cloud.write_bytes(b'verified cloud')
            writer=Writer(out/'correspondence',work,2)
            mask=np.array([[True,False],[True,True]])
            writer.collect(0,mask,np.array([1,0,1])); writer.collect(1,mask,np.array([0,0,1]))
            writer.prepare()
            writer.bucket(0,np.array([[0,0,0],[0,1,0]]),np.array([0,1,0]),0)
            writer.bucket(1,np.array([[1,0,0],[1,1,0]]),np.array([1,0,1]),2)
            writer.finish(root,cloud,.1,4)
            shared=ExportCorrespondence(root)
            np.testing.assert_array_equal(shared.frame(0)[0],[3,0,2])
            np.testing.assert_array_equal(shared.frame(1)[0],[1,0,3])
            np.testing.assert_array_equal(shared.frame(0)[1],mask)
            (root/'optimized.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'provenance'): ExportCorrespondence(root)


class JacobianTests(unittest.TestCase):
    def test_central_difference_near_zero_and_large_rotations(self):
        jacobian=worker_module('registration_jacobian').jacobian
        rng=np.random.default_rng(15);initial=np.tile(np.eye(4),(4,1,1))
        a=np.array([0,1,2,3,1]);b=np.array([1,2,3,0,3]);xa=rng.normal(size=(5,3));xb=rng.normal(size=(5,3));weights=rng.random(5)
        for scale in [0,1e-7,.4,3.]:
            x=rng.normal(size=(3,6));x[:,3:]*=scale;x=x.ravel()
            def residual(x):
                v=x.reshape(-1,6);p=initial.copy();p[1:,:3,3]=v[:,:3];p[1:,:3,:3]=Rotation.from_rotvec(v[:,3:]).as_matrix()
                return ((np.einsum('nij,nj->ni',p[a,:3,:3],xa)+p[a,:3,3]-np.einsum('nij,nj->ni',p[b,:3,:3],xb)-p[b,:3,3])*weights[:,None]).ravel()
            h=1e-6;eye=np.eye(len(x))*h
            numeric=np.column_stack([(residual(x+e)-residual(x-e))/(2*h) for e in eye])
            np.testing.assert_allclose(jacobian(x,initial,a,b,xa,xb,weights).toarray(),numeric,atol=2e-8,rtol=2e-7)


class ReuseCompatibilityTests(unittest.TestCase):
    def test_pinned_producer_only_and_input_checks_remain(self):
        from spacewatch_pipeline.reuse import gaussian_result
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';out=root/'new';coords=root/'coordinates.json';coords.write_text('{}')
            checkpoint=root/'weights.pt';checkpoint.write_bytes(b'test weights')
            producer=read(ROOT/'configs/gaussian_reuse_compatibility.json')['spacewatch_pipeline/gaussian.py']['producer_sha256'][0]
            identity=dict(config=dict(gaussian=dict(steps=10)),code={
                'spacewatch_pipeline/gaussian.py':producer,
                'spacewatch_pipeline/gaussian_data.py':sha(ROOT/'spacewatch_pipeline/gaussian_data.py')})
            save(old/'identity.json',identity)
            source=dict(source_run=str(root/'source'),coordinates=str(coords))
            save(old/'results.json',{'one.reconstruct':source,'one.gs_train':dict(checkpoint=str(checkpoint))})
            for stage,path in [('reconstruct',coords),('gs_prepare',coords),('gs_train',checkpoint)]:
                save(old/'stages'/('one.'+stage)/'complete.json',dict(artifacts=[record(path)]))
            job=dict(config=dict(reuse_gaussian_run=str(old),gaussian=dict(steps=10,decoded_cache_mib=32)),
                capture=dict(id='one'),action='gs_train',output=str(out),dependencies={'one.reconstruct':source})
            gaussian_result(job);self.assertTrue(read(out/'result.json')['reused'])
            identity['code']['spacewatch_pipeline/gaussian.py']='unreviewed';save(old/'identity.json',identity)
            with self.assertRaisesRegex(ValueError,'compatibility'):gaussian_result(job)
            identity['code']['spacewatch_pipeline/gaussian.py']=producer;save(old/'identity.json',identity)
            checkpoint.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError,'Artifact changed'):gaussian_result(job)


if __name__=='__main__':unittest.main()
