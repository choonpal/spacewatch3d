"""Checks for external camera/instance conventions, not model accuracy."""
import unittest

import numpy as np
import open3d as o3d
import torch

from tools.scenediff_mesh.prepare import make_grid, rasterize, view_rotation
from tools.scenediff_mesh.run import region_match_cost, scatter_mean


class ExternalInputsTests(unittest.TestCase):
    def test_camera_pixel_roundtrip_under_rotation_and_scale(self):
        K, directions = make_grid(33)
        R = view_rotation(.7, .2)
        translation = np.array([4., -2., 7.])
        depth = np.full((33, 33), 2.5)
        world = directions * depth[..., None] @ R.T + translation
        camera = (world - translation) @ R
        pixel = camera @ K.T
        pixel = pixel[..., :2] / pixel[..., 2:]
        yy, xx = np.mgrid[:33, :33]
        np.testing.assert_allclose(pixel, np.stack([xx, yy], -1), atol=1e-5)
        np.testing.assert_allclose(camera[..., 2], depth, atol=1e-8)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)

    def test_raycast_uses_first_surface_and_preserves_missing_observation(self):
        vertices = np.array([[-1,-1,2], [1,-1,2], [0,1,2], [-1,-1,4], [1,-1,4], [0,1,4]], dtype="f4")
        faces = np.array([[0,1,2], [3,4,5]], dtype="u4")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
        _, directions = make_grid(9)
        depth, ids, valid = rasterize(scene, np.eye(4), directions, np.array([7, 99]), 2.)
        self.assertEqual(ids[4, 4], 7)
        self.assertAlmostEqual(depth[4, 4], 2., places=6)
        self.assertTrue(valid[4, 4])
        self.assertFalse(valid[0, 0])
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(depth[0, 0], 0)

    def test_non_contiguous_instance_ids_match_without_background_assumption(self):
        feature = torch.tensor([[[1.,1.],[0.,0.]], [[0.,0.],[1.,1.]]])
        a = torch.tensor([[17,17],[28,28]])
        b = torch.tensor([[3,3],[6,6]])
        valid = torch.ones(2, 2, dtype=torch.bool)
        ca, cb, *_ = region_match_cost(feature, feature, a, b, valid, valid)
        torch.testing.assert_close(ca, torch.zeros_like(ca))
        torch.testing.assert_close(cb, torch.zeros_like(cb))

    def test_empty_counterpart_region_is_not_a_tensor_max_failure(self):
        feature = torch.ones(2, 2, 2)
        source = torch.zeros(2, 2, dtype=torch.long)
        empty = torch.full_like(source, -1)
        valid = torch.ones(2, 2, dtype=torch.bool)
        ca, cb, *_ = region_match_cost(feature, feature, source, empty, valid, valid)
        torch.testing.assert_close(ca, torch.ones_like(ca))
        torch.testing.assert_close(cb, torch.zeros_like(cb))

    def test_native_voxel_mean_handles_unobserved_index(self):
        values = torch.tensor([[1., 3.], [8., 10.], [5., 7.]])
        result = scatter_mean(values, torch.tensor([0, 2, 0]))
        torch.testing.assert_close(result, torch.tensor([[3., 5.], [0., 0.], [8., 10.]]))


if __name__ == "__main__":
    unittest.main()
