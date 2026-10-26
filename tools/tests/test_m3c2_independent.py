import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_m3c2_independent import exclusions, outside_boxes, ray_votes, classify_patch, rigid_patch_fit, choose_motion_pairs, integrate_patches


class IndependentChangesTests(unittest.TestCase):
    def test_exclusion_uses_only_locations_and_frame_ids_not_object_labels(self):
        event={'instance_id':19,'semantic_class':5,'type':'moved',
               'before_bounds':{'min':[0,0,0],'max':[1,1,1]},'after_bounds':None,'keyframes':[[{'frame_id':'old'}],[]]}
        a,frames=exclusions({'regions':[event]})
        event.update(instance_id=9999,semantic_class=-1,type='anything')
        b,other_frames=exclusions({'regions':[event]})
        np.testing.assert_array_equal(a,b);self.assertEqual(frames,other_frames)
        np.testing.assert_array_equal(outside_boxes(np.array([[.5,.5,.5],[3,3,3]]),a),[False,True])

    def test_raw_depth_distinguishes_free_occupied_occluded_and_invalid(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'depth.raw';np.full((480,640),2000,dtype='<u2').tofile(path)
            frame={'depth_camera_to_world':np.eye(4).tolist(),'depth_path':str(path),'depth_fitness':.9,'depth_rmse_m':.01}
            result=ray_votes(np.array([[0.,0.,1.],[0.,0.,2.],[0.,0.,3.]]),[frame])
            np.testing.assert_array_equal(result['free'][:,0],[True,False,False])
            np.testing.assert_array_equal(result['occupied'][:,0],[False,True,False])
            np.testing.assert_array_equal(result['occluded'][:,0],[False,False,True])
            np.zeros((480,640),dtype='<u2').tofile(path)
            unknown=ray_votes(np.array([[0.,0.,1.]]),[frame])
            self.assertFalse(any(unknown[k].any() for k in ['free','occupied','occluded']))

    def test_presence_requires_observed_free_space_with_separated_views(self):
        own={'occupied':.9};opposite={'free':.8,'free_views':3,'baseline_m':.3,'occluded':0.}
        self.assertEqual(classify_patch(own,opposite,significance=.8,nearest_other=.2)[0],'presence_difference')
        for altered in [dict(opposite,free=0.,occluded=.8),dict(opposite,baseline_m=.02),dict(opposite,free_views=1)]:
            self.assertEqual(classify_patch(own,altered,significance=.8,nearest_other=.2)[0],'suspected')
        self.assertEqual(classify_patch(own,opposite,significance=.3,nearest_other=.2)[0],'suspected')

    def test_dataset_pose_uses_its_calibration_and_depth_units(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'depth.raw';depth=np.zeros((20,30),dtype='<u2')
            depth[6:15,8:17]=10000;depth.tofile(path)
            frame={'status':'dataset_pose','depth_camera_to_world':np.eye(4).tolist(),
                'depth_path':str(path),'depth_size':[30,20],'depth_intrinsics':[100.,110.,12.,10.],
                'depth_units_per_meter':5000}
            result=ray_votes(np.array([[0.,0.,2.],[0.,0.,1.],[0.,0.,3.]]),[frame])
            np.testing.assert_array_equal(result['occupied'][:,0],[True,False,False])
            np.testing.assert_array_equal(result['free'][:,0],[False,True,False])
            np.testing.assert_array_equal(result['occluded'][:,0],[False,False,True])

    def test_unique_nonplanar_geometry_can_produce_a_rigid_motion(self):
        rng=np.random.default_rng(73);a=rng.normal(size=(250,3))*[.20,.12,.09]
        c,s=np.cos(np.pi/4),np.sin(np.pi/4);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        b=a@r.T+[.65,.2,.08];fit=rigid_patch_fit(a,b)
        self.assertIsNotNone(fit);self.assertGreater(fit['overlap'],.95);self.assertLess(fit['residual_m'],.01)
        self.assertGreater(fit['displacement_m'],.5)
        flat=a.copy();flat[:,2]=0
        self.assertIsNone(rigid_patch_fit(flat,flat+[.6,0,0]))

    def test_repeated_similar_patches_are_not_forced_into_a_motion_pair(self):
        rng=np.random.default_rng(73);a=rng.normal(size=(250,3))*[.20,.12,.09]
        points=np.vstack([a,a+[.6,0,0],a+[1.2,0,0]])
        pairs,ambiguous,_=choose_motion_pairs([{'ids':np.arange(250)}],
            [{'ids':np.arange(250,500)},{'ids':np.arange(500,750)}],points)
        self.assertEqual(pairs,[]);self.assertIn(0,ambiguous)

    def test_integrated_decision_has_add_remove_motion_and_uncertain_paths(self):
        rng=np.random.default_rng(73);a=rng.normal(size=(250,3))*[.20,.12,.09]
        b=a+[.65,.2,.08];points=np.vstack([a,b])
        before={'ids':np.arange(250),'decision':'presence_difference','reason':'observed free space'}
        after={'ids':np.arange(250,500),'decision':'presence_difference','reason':'observed free space'}
        drafts,_,_=integrate_patches([[before],[after]],points)
        self.assertEqual([d['type'] for d in drafts],['moved'])
        drafts,_,_=integrate_patches([[before],[]],points)
        self.assertEqual(drafts[0]['type'],'removed')
        drafts,_,_=integrate_patches([[],[after]],points)
        self.assertEqual(drafts[0]['type'],'added')
        uncertain=dict(after,decision='suspected',reason='occluded')
        drafts,_,_=integrate_patches([[],[uncertain]],points)
        self.assertEqual(drafts[0]['type'],'suspected')


if __name__=='__main__':unittest.main()
