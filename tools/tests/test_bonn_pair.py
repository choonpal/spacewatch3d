"""Camera calibration and timestamp checks for the real Bonn adapter."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_bonn_pair import interpolate_pose, M, S, DEPTH_CALIBRATION_SCALE


class BonnPoseTests(unittest.TestCase):
    def test_polar_rotation_and_depth_scale_reproduce_published_similarity(self):
        quaternion=Rotation.from_euler('xyz',[.3,.1,-.2]).as_quat()
        gt=np.array([[0.,1.,2.,3.,*quaternion],[1.,1.2,2.,3.,*quaternion]])
        corrected=interpolate_pose(gt,.5,max_gap=2.)
        native=np.eye(4);native[:3,:3]=Rotation.from_quat(quaternion).as_matrix();native[:3,3]=[1.1,2.,3.]
        s=np.eye(4);s[:3,:3]=S
        affine=s@native@s@M
        optical=np.array([[.2,-.4,2.],[.5,.1,3.]])
        exact=optical@affine[:3,:3].T+affine[:3,3]
        rigid=(optical*DEPTH_CALIBRATION_SCALE)@corrected[:3,:3].T+corrected[:3,3]
        np.testing.assert_allclose(exact,rigid,atol=.0002)
        np.testing.assert_allclose(corrected[:3,:3].T@corrected[:3,:3],np.eye(3),atol=1e-10)

    def test_pose_gaps_and_uncovered_images_are_rejected(self):
        gt=np.array([[0.,0.,0.,0.,0.,0.,0.,1.],[1.,0.,0.,0.,0.,0.,0.,1.]])
        for t in [-1.,.5,2.]:
            with self.assertRaises(ValueError):interpolate_pose(gt,t)


if __name__=='__main__':unittest.main()
