"""Observed-RGB reprojection and spherical candidate-box boundaries."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools/scenediff_mesh'))
from build_panorama_interactions import sphere_rays, project, nearest, bilinear, seam_labels, circular_bounds


class PanoramaInteractions(unittest.TestCase):
    def camera(self, center=(0, 0, 0), scale=1):
        return dict(center=list(center), rotation=np.eye(3).tolist(), scale=scale)

    def test_identity_camera_samples_original_rgb_and_mask(self):
        rays = sphere_rays(32, 16)
        u, v, fallback = project(self.camera(), self.camera(), rays, np.ones((16, 32)))
        original = np.arange(16*32*3, dtype='u1').reshape(16, 32, 3)
        np.testing.assert_array_equal(nearest(original, u, v), original)
        np.testing.assert_allclose(bilinear(original, u, v).astype(float), original, atol=1)
        self.assertFalse(fallback.any())

    def test_depth_projection_uses_translation_and_sim3_scale(self):
        direction = np.array([[[0., 0., 1.]]])
        u, v, fallback = project(self.camera(scale=2), self.camera((2, 0, 0)), direction, np.array([[3.]]))
        self.assertAlmostEqual(u.item(), np.arctan2(-2, 6)/(2*np.pi)+.5)
        self.assertAlmostEqual(v.item(), .5)
        self.assertFalse(fallback.item())
        u, v, fallback = project(self.camera(), self.camera((20, 0, 0)), direction, np.zeros((1, 1)))
        self.assertTrue(fallback.item())
        self.assertAlmostEqual(u.item(), .5)  # Missing depth is explicitly a direction-only fallback.

    def test_candidate_box_wraps_seam_without_spanning_whole_panorama(self):
        mask = np.zeros((16, 32), bool)
        mask[6:8, [0, 31]] = True
        mask[12, 16] = True
        labels = seam_labels(mask, padding=0)
        self.assertEqual(labels[6, 0], labels[6, 31])
        self.assertNotEqual(labels[6, 0], labels[12, 16])
        self.assertEqual(int((labels>0).sum()), int(mask.sum()))
        center, half = circular_bounds(np.array([0, 31]), 32)
        self.assertLess(min(center, 1-center), .05)
        self.assertLess(half, .25)

    def test_seam_grouping_does_not_wrap_between_top_and_bottom(self):
        mask = np.zeros((16, 32), bool)
        mask[0, 0], mask[-1, -1] = True, True
        labels = seam_labels(mask, padding=0)
        self.assertNotEqual(labels[0, 0], labels[-1, -1])
        self.assertEqual(int((labels>0).sum()), 2)


if __name__ == '__main__':
    unittest.main()
