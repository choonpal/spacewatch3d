"""Preserve face identity, collision inputs and the exact degeneracy boundary."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/spacewatch3d_src'))
from spacewatch3d.tasks.task_5b_keyframes.loader import deduplicate_surfaces
from spacewatch3d.tasks.task_5b_keyframes.models import Config, Geometry, InputError


class SurfaceDedupTests(unittest.TestCase):
    def test_shared_owners_first_coordinates_and_collision_union(self):
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        shifted = triangle[[2, 0, 1]] + 1e-11
        original_a = np.array([triangle, triangle[::-1]])
        original_b = np.array([shifted])
        geometries = [Geometry('a', original_a.copy(), 'a', solid=True),
                      Geometry('b', original_b.copy(), 'b', solid=False)]
        warnings = []
        deduplicate_surfaces(geometries, Config(), warnings)
        self.assertEqual(len(geometries), 3)
        self.assertEqual(len(geometries[0].triangles), 0)
        self.assertEqual(len(geometries[1].triangles), 0)
        np.testing.assert_array_equal(geometries[0].collision_triangles, original_a)
        np.testing.assert_array_equal(geometries[1].collision_triangles, original_b)
        shared = geometries[-1]
        np.testing.assert_array_equal(shared.triangles, [triangle])
        self.assertEqual(shared.object_ids, ('a', 'b'))
        self.assertFalse(shared.obstacle)
        self.assertEqual(warnings[0]['duplicate_count'], 2)
        self.assertEqual(warnings[0]['conflicting_duplicate_count'], 1)

    def test_scalar_degeneracy_boundary_and_empty_geometries(self):
        epsilon = 1e-12
        triangles = np.array([[[0., 0., 0.], [1., 0., 0.], [0., y, 0.]]
                              for y in [np.nextafter(epsilon, 0.), epsilon,
                                        np.nextafter(epsilon, np.inf)]])
        geometries = [Geometry('empty', np.empty((0, 3, 3)), None),
                      Geometry('boundary', triangles.copy(), None)]
        deduplicate_surfaces(geometries, Config(), [])
        np.testing.assert_array_equal(geometries[1].triangles, triangles[2:])
        self.assertEqual(geometries[0].triangles.shape, (0, 3, 3))

    def test_unique_limit_after_duplicate_removal(self):
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        config = replace(Config(), max_triangles=1)
        geometries = [Geometry('same', np.array([triangle, triangle[::-1]]), None)]
        deduplicate_surfaces(geometries, config, [])
        self.assertEqual(len(geometries[0].triangles), 1)
        with self.assertRaisesRegex(InputError, 'Total max_triangles'):
            deduplicate_surfaces([Geometry('distinct', np.array([triangle, triangle+2]), None)], config, [])


if __name__ == '__main__':
    unittest.main()
