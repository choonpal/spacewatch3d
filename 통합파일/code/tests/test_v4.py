import unittest
from unittest.mock import patch,mock_open
import numpy as np
from spacewatch_pipeline.florence import unique_pixel_votes,aggregate_votes
from spacewatch_pipeline.engine import stages,validate,worker_environment
from spacewatch_pipeline.io import ROOT,read
from spacewatch_pipeline import gpu

class PipelineTests(unittest.TestCase):
    def test_multiple_crops_do_not_make_independent_pixels(self):
        idx=unique_pixel_votes(np.array([1,1,1,1,2]),np.array([10,10,11,11,10]),np.array([3,3,3,4,3]))
        self.assertEqual(idx.tolist(),[0,4])
    def test_same_frame_cannot_certify_id(self):
        labels,_,_=aggregate_votes(np.array([0]*50),np.array([1]*50),np.array([7]*50),1,0)
        self.assertEqual(labels.tolist(),[0])
    def test_multiframe_and_ties(self):
        labels,_,_=aggregate_votes(np.array([0,0,1,1,1,1]),np.array([1,2,1,1,2,2]),np.array([7,7,7,8,7,8]),2,0)
        self.assertEqual(labels.tolist(),[7,0])
    def test_same_frame_many_pixels_do_not_overrule_frames(self):
        labels,_,_=aggregate_votes(np.zeros(102,int),np.r_[np.ones(100,int),2,3],np.r_[np.full(100,8),7,7],1,0)
        self.assertEqual(labels.tolist(),[7])
    def test_empty_scene_outputs_unknown(self):
        labels,_,_=aggregate_votes(np.array([],int),np.array([],int),np.array([],int),3,255)
        self.assertEqual(labels.tolist(),[255]*3)
    def test_config_graph_and_environment(self):
        c=read(ROOT/'configs/dorm2_florence.json');validate(c);rows=stages(c);seen=set()
        for r in rows:
            self.assertTrue(set(r['dependencies'])<=seen);seen.add(r['name'])
        self.assertEqual([r['action'] for r in rows if r['gpu']],['infer','refine_infer','render'])
        python,env=worker_environment(c,'infer',dict(uuid='test'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'test');self.assertEqual(python,c['runtime']['florence_python'])
        c['discovery']['models']['sam2']={}
        with self.assertRaises(ValueError):validate(c)
    def test_gpu_becoming_busy_is_skipped(self):
        a=dict(index=1,uuid='a',memory=10,utilization=0);b=dict(index=2,uuid='b',memory=10,utilization=0)
        with patch.object(gpu,'inventory',side_effect=[[a,b],[dict(a,memory=999),b],[b]]),patch.object(gpu.fcntl,'flock'),patch('builtins.open',mock_open()):
            with gpu.lease(thread_id='test') as selected:self.assertEqual(selected['uuid'],'b')
    def test_busy_gpu_times_out_without_lease(self):
        with patch.object(gpu,'inventory',return_value=[dict(index=0,uuid='a',memory=999,utilization=80)]),patch.object(gpu.fcntl,'flock'),patch('builtins.open',mock_open()):
            with self.assertRaises(TimeoutError):
                with gpu.lease(timeout=0,thread_id='test'):self.fail('Busy GPU leased')

if __name__=='__main__':unittest.main()
