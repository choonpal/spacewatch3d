import unittest
import numpy as np
from tools.mesh_simsac.build_event_viewer import silhouette_triangles


class PhotoShadows(unittest.TestCase):
    def view(self):
        return dict(pose=np.eye(4), K=np.array([[10., 0, 10], [0, 10., 10], [0, 0, 1]]), depth=np.ones((21, 21)))

    def test_offscreen_vertices_are_not_clamped_to_image_edges(self):
        points = np.array([[-4., 0, 1], [0, 1, 1], [0, -1, 1]])
        triangle = silhouette_triangles(points, np.ones(3, int), self.view(), output_size=21)
        self.assertEqual(triangle.shape, (1, 3, 2))
        self.assertEqual(triangle[0, 0, 0], -30)

    def test_behind_camera_never_draws_a_shadow(self):
        points = np.array([[-1., 0, -1], [0, 1, -1], [1, 0, -1]])
        self.assertEqual(len(silhouette_triangles(points, np.ones(3, int), self.view())), 0)

    def test_region_camera_preserves_original_photo_projection(self):
        points = np.array([[-.2, 0, 2], [0, .3, 2], [.2, 0, 2]])
        plain = silhouette_triangles(points, np.ones(3, int), self.view())
        shifted = self.view(); shifted['region_poses'] = np.stack([np.eye(4), np.eye(4)])
        shifted['region_poses'][1, :3, 3] = [3, 0, 2]
        regional = silhouette_triangles(points+[3, 0, 2], np.ones(3, int), shifted)
        np.testing.assert_allclose(plain, regional)


if __name__ == '__main__':
    unittest.main()
