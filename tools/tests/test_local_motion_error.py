import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_motion_error import local_model, assess_motion, normal_families
from build_local_motion_assessment import outside_object


class LocalMotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(412)
        self.points = []; self.normals = []
        for axis in range(3):
            p = rng.uniform(-1.5, 1.5, (350, 3)); p[:, axis] = 0.
            self.points.append(p); self.normals.append(np.tile(np.eye(3)[axis], (len(p), 1)))
        self.before = np.concatenate(self.points); self.n = np.concatenate(self.normals)
        self.drift = np.array([.07, .025, -.018])
        self.after = self.before+self.drift

    def region(self, real=(0., 0., 0.)):
        a = np.array([.4, .3, .7]); delta = self.drift+real
        transform = np.eye(4); transform[:3, 3] = delta
        return {'id': 1, 'type': 'moved', 'motion': {'before_reference_m': a.tolist(),
            'after_reference_m': (a+delta).tolist(), 'before_to_after': transform.tolist(),
            'surface_overlap': .9, 'surface_residual_m': .002}}

    def model(self, region):
        return local_model(self.after, self.before, self.n, [0., 0., 0.], region['motion']['after_reference_m'])

    def test_pure_map_drift_is_explained_instead_of_called_object_motion(self):
        region = self.region(); model = self.model(region)
        self.assertEqual(model['observable_dimensions'], 3)
        result = assess_motion(region, model)
        self.assertEqual(result['status'], 'alignment_explained')
        self.assertLess(result['corrected_displacement_m'], .001)
        self.assertAlmostEqual(result['local_map_drift_m'], np.linalg.norm(self.drift), places=3)

    def test_object_motion_remains_after_nearby_map_drift_is_removed(self):
        region = self.region(np.array([.25, .03, 0.]))
        result = assess_motion(region, self.model(region))
        self.assertEqual(result['status'], 'motion_supported')
        self.assertAlmostEqual(result['corrected_displacement_m'], np.linalg.norm([.25, .03, 0.]), places=3)

    def test_map_rotation_is_evaluated_at_the_object_position(self):
        rotation = Rotation.from_euler('xyz', [1., -.7, .5], degrees=True).as_matrix()
        source = self.before @ rotation.T+self.drift
        region = self.region(); a = np.array(region['motion']['before_reference_m'])
        matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = self.drift
        region['motion']['after_reference_m'] = (rotation @ a+self.drift).tolist()
        region['motion']['before_to_after'] = matrix.tolist()
        model = local_model(source, self.before, self.n, [0., 0., 0.], region['motion']['after_reference_m'])
        result = assess_motion(region, model)
        self.assertEqual(result['status'], 'alignment_explained')
        self.assertLess(result['corrected_rotation_deg'], .001)
        self.assertLess(result['corrected_displacement_m'], .001)

    def test_floor_cannot_resolve_horizontal_alignment(self):
        region = self.region(np.array([.25, 0., 0.]))
        model = local_model(self.after[-350:], self.before[-350:], self.n[-350:], [0., 0., 0.], region['motion']['after_reference_m'])
        self.assertEqual(model['observable_dimensions'], 1)
        result = assess_motion(region, model)
        self.assertEqual(result['status'], 'uncertain')
        self.assertIsNone(result['corrected_displacement_m'])

    def test_partial_geometry_can_support_motion_in_a_constrained_direction(self):
        region = self.region(np.array([0., 0., .25]))
        model = local_model(self.after[-350:], self.before[-350:], self.n[-350:], [0., 0., 0.], region['motion']['after_reference_m'])
        self.assertEqual(assess_motion(region, model)['status'], 'motion_supported')

    def test_noisy_parallel_walls_do_not_create_a_false_third_direction(self):
        n = np.array([[1., 0., 0.], [.999, .03, .01], [-.998, .04, 0.], [0., 0., 1.]])
        grouped, membership = normal_families(n)
        self.assertEqual(len(set(membership)), 2)
        self.assertTrue(np.allclose(grouped[0], grouped[1]))

    def test_insufficient_background_and_ambiguous_identity_are_uncertain(self):
        region = self.region(np.array([.5, 0., 0.]))
        too_small = local_model(self.after[:20], self.before[:20], self.n[:20], [0., 0., 0.], [0., 0., 0.])
        self.assertEqual(assess_motion(region, too_small)['status'], 'uncertain')
        region['identity_ambiguous'] = True
        self.assertEqual(assess_motion(region, self.model(region))['reason'], 'ambiguous_object_identity')

    def test_presence_changes_are_not_reclassified_from_registration_error(self):
        for kind in ('added', 'removed'):
            self.assertEqual(assess_motion({'id': 2, 'type': kind}, {})['status'], 'not_applicable')

    def test_object_and_both_motion_endpoints_are_excluded_from_background(self):
        region = {'before_bounds': {'min': [0., 0., 0.], 'max': [1., 1., 1.]},
                  'after_bounds': {'min': [2., 0., 0.], 'max': [3., 1., 1.]}}
        points = np.array([[.5, .5, .5], [2.5, .5, .5], [1.1, .5, .5], [4., .5, .5]])
        self.assertEqual(outside_object(points, region, np.eye(3), .15).tolist(), [False, False, False, True])

    def test_replicated_mesh_samples_do_not_shrink_the_error_envelope(self):
        rng = np.random.default_rng(74)
        source = self.after+rng.normal(0., .006, self.after.shape)
        model = local_model(source, self.before, self.n, [0., 0., 0.], [.4, .3, .7])
        dense = local_model(np.tile(source, (5, 1)), np.tile(self.before, (5, 1)), np.tile(self.n, (5, 1)), [0., 0., 0.], [.4, .3, .7])
        self.assertGreater(model['displacement_envelope_m'], .001)
        self.assertAlmostEqual(dense['displacement_envelope_m']/model['displacement_envelope_m'], 1., places=4)


if __name__ == '__main__':
    unittest.main()
