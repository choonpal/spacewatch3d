import unittest
import numpy as np
from tools.mesh_simsac.uncertainty import object_uncertainty
from tools.mesh_simsac.regional import FreshRegionalAlignment


class RegistrationUncertainty(unittest.TestCase):
    def test_whole_cloud_tail_and_held_out_region_both_raise_threshold(self):
        reg = dict(state='passed_geometric_gate', median_inlier_distances_u=[.01, .02],
                   pointcloud_residual_p90_u=[.18, .12], fresh_structural={'regions': [
                   {'supported': True, 'validation': {'tolerance_u': .25}},
                   {'supported': False, 'validation': {'tolerance_u': .4}}]})
        self.assertEqual(object_uncertainty(reg, [1], .05), (.25, True))
        self.assertEqual(object_uncertainty(reg, [1, 2], .05), (.4, False))
        self.assertEqual(object_uncertainty(reg, [], .05), (.18, True))
        self.assertEqual(object_uncertainty({k:v for k,v in reg.items() if k!='fresh_structural'}, [], .05), (.18, True))

    def test_newly_fitted_region_keeps_mesh_and_camera_in_same_frame(self):
        global_matrix = np.eye(4); global_matrix[:3, 3] = [2, 0, 0]
        local = np.eye(4); local[:3, 3] = [.1, 0, -.2]
        report = dict(global_after_to_before=global_matrix, centers=[[0,0,0]],
                      regions=[{'matrix_after_to_common': local@global_matrix}])
        regional = FreshRegionalAlignment(report)
        vertices = np.array([[0.,0,2], [1,0,2], [0,1,2]])
        out = regional.mesh(vertices, np.array([[0,1,2]]), np.array([3]), global_matrix, 1)
        np.testing.assert_allclose(out['vertices'], vertices+[2.1,0,-.2], atol=1e-6)
        np.testing.assert_allclose(regional.matrices(1)[1], local)


if __name__ == '__main__':
    unittest.main()
