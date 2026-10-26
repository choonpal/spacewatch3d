"""Requested virtual cameras must survive registration without nearest-frame snaps."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tools/gaussian_change'
sys.path.insert(0,str(TOOLS))
previous_prepare=sys.modules.pop('prepare',None)
try:
    spec=importlib.util.spec_from_file_location('gaussian_keyframe_preparation',TOOLS/'prepare_keyframes.py')
    keyframes=importlib.util.module_from_spec(spec);spec.loader.exec_module(keyframes)
finally:
    sys.path.pop(0)
    sys.modules.pop('prepare',None)
    if previous_prepare is not None:sys.modules['prepare']=previous_prepare


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(root):
    up=np.eye(4);up[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    correction=np.eye(4);correction[:3,:3]=Rotation.from_euler('y',.2).as_matrix()*1.2;correction[:3,3]=[4,1,2]
    ctx={'base_matrix':correction.tolist(),'methods':{'structure':{'region_matrices_after_Geo':[np.eye(4).tolist()],
         'errors':{'regions':{'R01':{'translation_bias_u':[.03,-.02,.01],'supported':True}}}}}}
    protocol={'scenes':[{'native_to_mesh':np.eye(4).tolist()},{'native_to_mesh':np.eye(4).tolist()}],
              'zones':[{'id':'R01'}],'height_u':2.,'source_hashes':{},'baseline_boxes':[{'center':[999,999,999]}],
              'membership_base_matrix':np.eye(4).tolist(),'membership_parent_transform':None,'alignment_context':ctx,'decisions':{}}
    base=root/'base/protocol.json';save(base,protocol)
    np.savez(base.parent/'partition.npz',labels=np.ones((2,2),int),origin=[-1000,-1000],step=1000.,axes=np.eye(2))
    source=root/'source'
    requested={}
    for capture in ['dorm2','dorm3']:
        camera=np.eye(4);camera[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        camera[:3,:3]=camera[:3,:3]@Rotation.from_euler('z',.31).as_matrix()
        camera[:3,3]=[3.125,7.25,1.8]
        requested[capture]=camera
        native=np.eye(4)[None];native[0,:3,3]=[0,0,2]
        glb=up[None]@native
        folder=base.parent/'source/data'/capture;folder.mkdir(parents=True)
        np.save(folder/'original_poses_glb.npy',glb);np.save(folder/'mesh_to_glb.npy',up)
        for folder in [source/'production'/capture,*([source/'task5b'] if capture=='dorm3' else [])]:
            plan_folder=folder/'plan' if folder.name=='task5b' else folder
            request={'viewpoint_id':'virtual-camera','capture_id':capture,'projection':'equirectangular',
                     'reference_from_camera':camera.tolist(),'width':2048,'height':1024,'media_origin':'novel_view'}
            save(plan_folder/'render_requests.json',{'route_id':'route','requests':[request]})
            save(plan_folder/'tour_plan.json',{'route_id':'route','viewpoints':[request],'planning_status':'incomplete',
                                             'publication_ready':False,'quality_status':'unverified','failures':[]})
            save(folder/'scene.json',{'capture_id':capture,'coordinates':{'up_axis':'+Z'},'camera':{'positions_m':[[0,0,2]]}})
    return base,source,requested


class KeyframeCameras(unittest.TestCase):
    def test_similarity_preserves_rays_and_roll_without_scaling_rotation(self):
        camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('xyz',[.3,-.7,.4]).as_matrix();camera[:3,3]=[1,2,3]
        m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',[-.2,.4,.8]).as_matrix()*1.7;m[:3,3]=[-2,4,1]
        common=keyframes.transform_camera(camera,m)
        for ray in [[0,0,1],[.3,-.2,1],[-.4,.5,1]]:
            p=camera[:3,:3]@np.asarray(ray)+camera[:3,3]
            expected=m[:3,:3]@p+m[:3,3]
            actual=common[:3,:3]@(np.asarray(ray)*1.7)+common[:3,3]
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(common[:3,:3].T@common[:3,:3],np.eye(3),atol=1e-12)

    def test_rejects_nonuniform_scale_and_reflection(self):
        for scale in [[1,2,1],[-1,1,1]]:
            with self.subTest(scale=scale),self.assertRaises(ValueError):
                keyframes.transform_camera(np.eye(4),np.diag([*scale,1]))

    def test_keeps_virtual_request_and_is_independent_of_baseline_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base,source,requested=fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):first=keyframes.build(base,source,root/'out1')
            changed=json.loads(base.read_text());changed['baseline_boxes']=[{'center':[-50,2,80]},{'center':[0,0,0]}];save(base,changed)
            with contextlib.redirect_stdout(io.StringIO()):second=keyframes.build(base,source,root/'out2')
            self.assertEqual(first['views'],second['views'])
            self.assertEqual(len(first['views']),3)
            self.assertFalse(first['baseline_boxes'])
            # The planner's Z is height; the Gaussian/common frame uses Y.
            # Camera-coordinate identity alone would not catch this bridge bug.
            before=first['views'][0]
            self.assertAlmostEqual(before['camera_to_common_cv'][1][3],1.8)
            self.assertAlmostEqual(before['camera_to_common_cv'][2][3],-7.25)
            for view in first['views']:
                np.testing.assert_array_equal(view['source_reference_from_camera'],requested[view['capture_id']])
                self.assertNotEqual(view['source_reference_from_camera'][0][3],0)
                self.assertFalse(view['existing_change_candidates_used'])
            self.assertTrue(all(s['publication_ready'] is False for s in first['keyframe_sources']))

    def test_rejects_a_plan_from_a_different_native_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base,source,_=fixture(root)
            p=source/'production/dorm2/scene.json';scene=json.loads(p.read_text());scene['camera']['positions_m'][0][0]=.01;save(p,scene)
            with self.assertRaisesRegex(ValueError,'do not match'):
                keyframes.build(base,source,root/'out')


if __name__=='__main__':unittest.main()
