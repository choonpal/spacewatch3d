"""Analytic geometry checks; no object labels or benchmark masks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_mesh_methods import categories, gls_descriptors, gls_field, m3c2, regions


class MeshMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.linspace(-.5, .5, 51), np.linspace(-.5, .5, 51))
        cls.plane = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        cls.normal = np.tile([0., 0., 1.], (len(cls.plane), 1))

    def test_m3c2_measures_normal_displacement_and_error_floor(self):
        result = m3c2(np.zeros((1, 3)), np.array([[0., 0., 1.]]), self.plane, self.plane+[0, 0, .12])
        self.assertAlmostEqual(result[0, 1], .12)
        self.assertAlmostEqual(result[0, 2], 1.96*.02)
        self.assertGreater(result[0, 0], 1)

    def test_missing_counterpart_is_undefined(self):
        result = m3c2(np.zeros((1, 3)), np.array([[0., 0., 1.]]), self.plane, self.plane+[0, 0, 2.])
        self.assertTrue(np.isnan(result[0, 0]))
        m, _ = categories(result, np.array([[np.nan, 0., 0., 0.]]), np.ones(1), np.ones(1))
        self.assertEqual(m[0], 3)

    def test_gls_plane_has_analytic_offset_and_zero_curvature(self):
        d = gls_descriptors(np.array([[0., 0., .04]]), self.plane, self.normal, np.array([.2]))[0, 0]
        np.testing.assert_allclose(d[:3], [.04/.2, 0., 0.], atol=1e-10)

    def test_gls_sphere_has_known_curvature(self):
        rng = np.random.default_rng(17)
        p = rng.normal(size=(12000, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
        d = gls_descriptors(np.array([[0., 0., 1.]]), p, p, np.array([.3]))[0, 0]
        self.assertAlmostEqual(d[0], 0., places=8)
        self.assertAlmostEqual(d[1], .3, places=8)
        self.assertAlmostEqual(d[2], 0., places=8)

    def test_gls_mapping_uses_minimum_containing_scale_and_query_weights(self):
        scales = np.array([.1, .2, .3])
        centers = np.array([[.05, 0., 0.], [.15, 0., 0.], [.25, 0., 0.]])
        a = np.zeros((3, 3, 6)); b = a.copy(); a[:, :, 5] = 1.; b[:, :, 5] = 1.
        # Offset-channel deltas are [1,4,9] at the three scales.
        b[:, :, 0] = [1., 2., 3.]
        field = gls_field(np.zeros((1, 3)), centers, a, b, scales)[0]
        expected = (.25*1+.15*4+.05*9)/(.25+.15+.05)
        self.assertAlmostEqual(field[0], expected)
        self.assertEqual(field[2], 3)

    def test_equal_gls_surfaces_have_zero_score_despite_normal_convention(self):
        centers = np.array([[.05, 0., .04], [.15, 0., .04], [-.05, 0., .04], [-.15, 0., .04]])
        scales = np.array([.1, .2, .3])
        a = gls_descriptors(centers, self.plane, self.normal, scales)
        b = gls_descriptors(centers, self.plane, -self.normal, scales)
        field = gls_field(np.zeros((1, 3)), centers, a, b, scales)
        self.assertAlmostEqual(field[0, 0], 0., places=12)

    def test_large_connected_region_is_split_for_local_navigation(self):
        xyz = np.column_stack([np.linspace(0., 4., 401), np.zeros(401), np.zeros(401)])
        classes = np.full(401, 2)
        fields = np.tile([.2, .1, .05, 20.], (401, 1))
        patches = regions(xyz, (classes, classes), (fields, fields))
        self.assertEqual(len(patches), 3)
        for patch in patches:
            self.assertLessEqual(patch['bounds']['max'][0]-patch['bounds']['min'][0], 1.76)
            self.assertAlmostEqual(patch['m3c2_median_abs_m'], .1)


if __name__ == '__main__':
    unittest.main()
