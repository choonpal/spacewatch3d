"""Coordinate, covariance and candidate accounting checks without a GPU runtime."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tools/gaussian_change'
spec=importlib.util.spec_from_file_location('gaussian_bridge_preparation',TOOLS/'prepare.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
def function(path,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace[name]
product=function(TOOLS/'render_maps.py','quaternion_product',{'np':np})
clusters=function(TOOLS/'infer_simsac.py','clusters',dict(np=np,cKDTree=cKDTree,coo_matrix=coo_matrix,connected_components=connected_components))

class GaussianMapBridge(unittest.TestCase):
    def test_gaussian_covariance_follows_uniform_scale_and_rotation(self):
        source=Rotation.from_euler('xyz',[.2,-.5,.8]);global_rotation=Rotation.from_euler('xyz',[-.1,.7,-.3])
        a=global_rotation.as_quat()[[3,0,1,2]];b=source.as_quat()[[3,0,1,2]]
        out=product(a,b);R=Rotation.from_quat(out[[1,2,3,0]]).as_matrix()
        scales=np.array([.02,.1,.5]);s=.842
        expected=global_rotation.as_matrix()@source.as_matrix()@np.diag(scales**2)@source.as_matrix().T@global_rotation.as_matrix().T*s*s
        np.testing.assert_allclose(R@np.diag((scales*s)**2)@R.T,expected,atol=1e-12)

    def test_native_sh_direction_matches_inverse_world_rotation(self):
        R=Rotation.from_euler('xyz',[.2,.9,-.5]).as_matrix();direction=np.array([.5,-.4,1.2]);direction/=np.linalg.norm(direction)
        common=R@direction
        # Same row-vector operation used for evaluating native directional SH.
        np.testing.assert_allclose(common@R,direction,atol=1e-12)

    def test_frozen_epoch_transform_keeps_bias_and_reference_identity(self):
        base=np.eye(4);base[:3,:3]*=.978;base[:3,3]=[1,2,3]
        correction=np.eye(4);correction[:3,3]=[-.1,.2,.3]
        ctx={'base_matrix':base.tolist(),'methods':{'structure':{'region_matrices_after_Geo':[correction.tolist()],'errors':{'regions':{'R01':{'translation_bias_u':[.01,.02,.03]}}}}}}
        expected=correction@base;expected[:3,3]+=[.01,.02,.03]
        np.testing.assert_array_equal(bridge.epoch_to_common(ctx,0,'R01'),np.eye(4))
        np.testing.assert_allclose(bridge.epoch_to_common(ctx,1,'R01'),expected)

    def test_erp_camera_stays_level_and_right_handed(self):
        position=np.array([2.,1.7,3.]);frame=bridge.camera_frame(position,np.array([-1.,0.,8.]),np.array([0.,0.,1.]))
        np.testing.assert_allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-12)
        np.testing.assert_array_equal(frame[:3,1],[0,-1,0])
        self.assertAlmostEqual(np.linalg.det(frame[:3,:3]),1)

    def test_voxel_boxes_preserve_both_epochs_and_separate_distant_regions(self):
        before=np.array([[0.,0.,0.],[.01,0,0]])
        after=np.array([[.02,0.,0.],[5.,0.,0.],[5.01,0.,0.]])
        boxes=clusters(before,after,'R08',2.)
        self.assertEqual(len(boxes),2)
        self.assertEqual(sum(b['before_pixels'] for b in boxes),2)
        self.assertEqual(sum(b['after_pixels'] for b in boxes),3)
        for point in np.r_[before,after]:
            self.assertTrue(any(np.all(point>=b['bounds'][0]) and np.all(point<=b['bounds'][1]) for b in boxes))

if __name__=='__main__':unittest.main()
