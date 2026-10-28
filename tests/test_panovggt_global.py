"""CPU geometry tests; optional reconstruction dependencies are task-local."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS=all(importlib.util.find_spec(m) is not None for m in ('numpy','scipy'))
if HAS_DEPS:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from spacewatch3d.tasks.task_1b_360_input import panovggt_global as g


@unittest.skipUnless(HAS_DEPS,'Install optional PanoVGGT geometry dependencies')
class GlobalAlignmentTests(unittest.TestCase):
    def test_similarity_recovers_scale_rotation_translation_with_outliers(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(180,3))
        truth=(1.3,Rotation.from_rotvec([.2,-.1,.3]).as_matrix(),np.array([2.,-3.,.8]))
        y=g.apply(x,truth)+rng.normal(scale=.001,size=x.shape)
        y[:50]=rng.normal(scale=8,size=(50,3))
        fitted,inliers,report=g.robust_fit(x,y,.025,g.Config(ransac_trials=160))
        self.assertLess(np.median(np.linalg.norm(g.apply(x,fitted)-g.apply(x,truth),axis=1)),.003)
        self.assertGreater(inliers.sum(),120)
        self.assertGreater(report['holdout_inlier_fraction'],.5)

    def test_false_and_degenerate_loop_rejected(self):
        line=np.column_stack((np.arange(30),np.zeros((30,2))))
        with self.assertRaisesRegex(ValueError,'collinear'):g.fit_similarity(line,line)
        rng=np.random.default_rng(1)
        with self.assertRaises(ValueError):
            g.robust_fit(rng.normal(size=(100,3)),rng.normal(size=(100,3)),.02,g.Config(ransac_trials=100))

    def test_graph_closes_drift_in_all_seven_dofs_with_fixed_anchor(self):
        n=12;truth=[]
        for i in range(n):
            angle=2*np.pi*i/(n-1)
            truth.append((1+.05*np.sin(angle),Rotation.from_rotvec([0,0,.15*np.sin(angle)]).as_matrix(),np.array([3*np.sin(angle),3*(1-np.cos(angle)),.2*np.sin(angle)])))
        bias=(1.008,Rotation.from_rotvec([.001,-.002,.012]).as_matrix(),np.array([.035,.01,-.005]))
        initial=[truth[0]];edges=[]
        for i in range(n-1):
            measurement=g.compose(g.compose(g.inverse(truth[i]),truth[i+1]),bias)
            edges.append(g.make_edge(i,i+1,measurement,3,'overlap'))
            initial.append(g.compose(initial[-1],measurement))
        edges.append(g.make_edge(0,n-1,g.compose(g.inverse(truth[0]),truth[-1]),3,'loop'))
        optimized,report=g.optimize_graph(initial,edges)
        error=lambda ts:np.linalg.norm(ts[-1][2]-truth[-1][2])+abs(np.log(ts[-1][0]/truth[-1][0]))+np.linalg.norm(Rotation.from_matrix(truth[-1][1].T@ts[-1][1]).as_rotvec())
        self.assertLess(error(optimized),.3*error(initial))
        self.assertLess(report['after_robust_cost'],report['before_robust_cost'])
        for a,b in zip(optimized[0],truth[0]):np.testing.assert_allclose(a,b)
        self.assertTrue(all(t[0]>0 and abs(np.linalg.det(t[1])-1)<1e-8 for t in optimized))

    def test_no_loop_and_disconnected_graph_fail(self):
        e=g.make_edge(0,1,g.identity(),3,'overlap')
        with self.assertRaisesRegex(ValueError,'No verified loop'):g.optimize_graph([g.identity()]*2,[e])
        e['kind']='loop'
        with self.assertRaisesRegex(ValueError,'Disconnected'):g.optimize_graph([g.identity()]*3,[e])

    def test_lift_respects_stride_valid_mask_and_bounds(self):
        d={'points':np.arange(27).reshape(1,3,3,3),'depths':np.ones((1,3,3)),'valid':np.ones((1,3,3),dtype=bool)}
        d['valid'][0,1,1]=False
        points,_,valid=g.lift(d,0,np.array([[0,0],[3,3],[6,6],[99,99]]),3,(8,8))
        np.testing.assert_array_equal(valid,[True,False,True,False])
        np.testing.assert_array_equal(points[2],d['points'][0,2,2])

    def test_missing_raw_cache_fails_before_output_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (run/'settings.json').write_text(json.dumps({'window_indices':[[0]]}))
            (run/'input_manifest.json').write_text(json.dumps({'frames':[{'file':'frames/a.jpg'}]}))
            with self.assertRaisesRegex(FileNotFoundError,'raw prediction'):
                g.optimize_run(run,root,root,root/'out')
            self.assertFalse((root/'out').exists())
            with self.assertRaisesRegex(ValueError,'read-only'):g.optimize_run(run,root,root,run)

    def test_graph_build_and_file_outputs_on_synthetic_raw_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'windows').mkdir(parents=True)
            ids=[[0,1],[1,2],[2,3],[3,4]];transforms=[]
            yy,xx=np.mgrid[:8,:8];cloud=np.stack((xx/2,yy/2,.4*np.sin(xx)+.3*np.cos(yy)),axis=-1)
            for i in range(4):
                transform=(1+.07*i,Rotation.from_rotvec([.02*i,0,.05*i]).as_matrix(),np.array([.3*i,-.2*i,.1*i]))
                transforms.append(transform)
                local=g.apply(cloud,g.inverse(transform));points=np.stack([local,local])
                poses=np.stack([np.eye(4),np.eye(4)])
                colors=np.full((2,8,8,3),[30,90,180],dtype=np.uint8)
                np.savez(run/f'windows/window_{i:03d}.npz',indices=ids[i],points=points,valid=np.ones((2,8,8),dtype=bool),depths=np.full((2,8,8),3.),colors=colors,poses=poses)
            settings=dict(window_indices=ids,pixel_step=1,voxel_size=.01)
            (run/'settings.json').write_text(json.dumps(settings))
            (run/'input_manifest.json').write_text(json.dumps({'frames':[{'file':f'frames/{i}.jpg','timestamp_s':float(i)} for i in range(5)]}))
            initial=[(s*1.01**i,r,t+np.array([.08*i,0,0])) for i,(s,r,t) in enumerate(transforms)]
            np.savez(run/'window_transforms.npz',scale=[t[0] for t in initial],rotation=[t[1] for t in initial],translation=[t[2] for t in initial])
            pixels=np.column_stack((xx.ravel(),yy.ravel()))
            candidate=dict(a=0,b=4,matches=64,pixels_a=pixels,pixels_b=pixels,shape_a=(8,8),shape_b=(8,8))
            with patch.object(g,'detect_candidates',return_value=[candidate]):
                result,report=g.optimize_run(run,root,root,root/'out',g.Config(ransac_trials=30))
            self.assertEqual(report['loop_edges'],1)
            self.assertTrue((root/'out/global_optimization.json').is_file())
            for estimated,truth in zip(result,transforms):
                np.testing.assert_allclose(g.apply(cloud,estimated),g.apply(cloud,truth),atol=.0002)
            export=g.export_dense(run,root/'out',result)
            self.assertEqual(export['raw_points'],5*64)
            self.assertFalse((run/'panovggt_merged.ply').exists())
            ply=(root/'out/panovggt_merged.ply').read_bytes()
            header,body=ply.split(b'end_header\n',1)
            dtype=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))])
            vertices=np.frombuffer(body,dtype=dtype)
            self.assertEqual(len(vertices),export['merged_points'])
            self.assertIn(b'format binary_little_endian',header)
            distances=np.linalg.norm(vertices['xyz'][:,None,:]-cloud.reshape(-1,3)[None,:,:],axis=-1)
            self.assertLess(distances.min(axis=1).max(),.0002)
            np.testing.assert_array_equal(vertices['rgb'],np.tile([30,90,180],(len(vertices),1)))
            cameras=json.loads((root/'out/camera_poses.json').read_text())
            for row in cameras:
                node=row['source_window'];p=np.array(row['camera_to_world'])
                np.testing.assert_allclose(p[:3,3],result[node][2],atol=1e-6)
                np.testing.assert_allclose(p[:3,:3],result[node][1],atol=1e-6)
            with self.assertRaises(FileExistsError):g.export_dense(run,root/'out',result)
            with np.load(run/'window_transforms.npz') as z:
                np.testing.assert_array_equal(z['translation'],[t[2] for t in initial])
            with self.assertRaises(FileExistsError):g.optimize_run(run,root,root,root/'out')


if __name__=='__main__':unittest.main()
