"""360 source axes, immutable direction queries and Sim3 camera conventions."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/scenediff_mesh'))
from build_panorama_review import camera_common,erp_rays,pixel_for_direction


class PanoramaGeometry(unittest.TestCase):
    def test_erp_covers_whole_sphere_with_open_cv_y_down(self):
        rays=erp_rays(1024,512)
        np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1,atol=1e-12)
        self.assertLess(rays[0,512,1],-.999)
        self.assertGreater(rays[-1,512,1],.999)
        for direction,column in [([0,0,1],512),([1,0,0],768),([0,0,-1],0),([-1,0,0],256)]:
            x,y=pixel_for_direction(direction,1024,512)
            self.assertLessEqual(min(abs(x-column),1024-abs(x-column)),1)
            self.assertLessEqual(abs(y-256),1)

    def test_direction_sampling_keeps_depth_vector_and_input_unchanged(self):
        direction=np.array([3.,-2.,7.]);saved=direction.copy()
        pixel_for_direction(direction,1024,512)
        np.testing.assert_array_equal(direction,saved)

    def test_sim3_camera_and_shared_world_anchor_roundtrip(self):
        angle=.8;c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        matrix=np.eye(4);matrix[:3,:3]=R*1.3;matrix[:3,3]=[2,-1,4]
        ctx={'base_matrix':matrix.tolist(),'methods':{'structure':{'region_matrices_after_Geo':[np.eye(4).tolist()], 'errors':{'regions':{'R01':{'translation_bias_u':[.02,.01,-.03]}}}}}}
        pose=np.eye(4);pose[:3,3]=[1,1.6,-2]
        camera=camera_common(pose,1,'R01',ctx)
        expected=matrix[:3,:3]@pose[:3,3]+matrix[:3,3]+[.02,.01,-.03]
        np.testing.assert_allclose(camera['center'],expected)
        np.testing.assert_allclose(camera['rotation'],R)
        self.assertAlmostEqual(camera['scale'],1.3)
        anchor=np.array([4.,.3,1.]);rotation=np.array(camera['rotation'])
        local=rotation.T@(anchor-np.array(camera['center']))
        pixel_for_direction(local,1024,512)
        np.testing.assert_allclose(rotation@local+camera['center'],anchor,atol=1e-12)


if __name__=='__main__':unittest.main()
