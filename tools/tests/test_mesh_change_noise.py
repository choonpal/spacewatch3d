"""Noise rejection checks with known unchanged/changed analytic surfaces."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_mesh_methods import m3c2
from mesh_change_noise import screen_surface_changes, patch_noise_decision


class NoiseScreenTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        x, y = np.meshgrid(np.linspace(-.5, .5, 71), np.linspace(-.5, .5, 71))
        self.a = np.column_stack([x.ravel(), y.ravel(), rng.normal(0, .016, x.size)])
        self.b = np.column_stack([x.ravel(), y.ravel(), .054 + rng.normal(0, .016, x.size)])
        x, y = np.meshgrid(np.linspace(-.12, .12, 5), np.linspace(-.12, .12, 5))
        self.q = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        self.n = np.tile([0., 0., 1.], (len(self.q), 1))

    def test_dense_unchanged_noisy_plane_is_rejected_even_when_nominal_ratio_exceeds_one(self):
        nominal = m3c2(self.q, self.n, self.a, self.b)
        self.assertGreater(np.median(nominal[:, 0]), 1.)
        screened = screen_surface_changes(self.q, self.n, (self.a, self.b))
        self.assertLess(screened['persistent'].mean(), .1)
        self.assertFalse(patch_noise_decision(screened, np.arange(len(self.q)))['keep'])

    def test_duplicating_mesh_samples_cannot_shrink_the_noise_envelope(self):
        once = screen_surface_changes(self.q, self.n, (self.a, self.b))
        dense = screen_surface_changes(self.q, self.n, (np.repeat(self.a, 4, axis=0), np.repeat(self.b, 4, axis=0)))
        np.testing.assert_allclose(once['envelope_m'], dense['envelope_m'], atol=.003)
        np.testing.assert_array_equal(once['persistent'], dense['persistent'])

    def test_a_real_eighteen_centimeter_surface_change_survives(self):
        changed = self.b + [0, 0, .126]
        screened = screen_surface_changes(self.q, self.n, (self.a, changed))
        self.assertGreater(screened['persistent'].mean(), .9)
        self.assertTrue(patch_noise_decision(screened, np.arange(len(self.q)))['keep'])
        reversed_normal = screen_surface_changes(self.q, -self.n, (self.a, changed))
        np.testing.assert_array_equal(screened['persistent'], reversed_normal['persistent'])

    def test_added_and_removed_surface_above_unchanged_floor_survives(self):
        x,y = np.meshgrid(np.linspace(-.14,.14,29),np.linspace(-.14,.14,29))
        top = np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.24)])
        after = np.vstack([self.a,top])
        query = self.q.copy();query[:,2]=.24
        added = screen_surface_changes(query,self.n,(self.a,after))
        removed = screen_surface_changes(query,self.n,(after,self.a))
        self.assertGreater(added['persistent'].mean(),.9)
        np.testing.assert_array_equal(added['persistent'],removed['persistent'])

    def test_same_rule_handles_walls_and_does_not_depend_on_floor_height(self):
        # Rotate the whole scene and translate it far from z=0.
        r = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        shift = np.array([4., 2., 3.])
        screened = screen_surface_changes(self.q @ r.T + shift, self.n @ r.T,
            (self.a @ r.T + shift, self.b @ r.T + shift))
        self.assertFalse(patch_noise_decision(screened, np.arange(len(self.q)))['keep'])

    def test_missing_observations_are_insufficient_support_not_zero_difference(self):
        screened = screen_surface_changes(self.q, self.n, (self.a, np.empty((0, 3))))
        self.assertFalse(screened['valid'].any())
        decision = patch_noise_decision(screened, np.arange(len(self.q)))
        self.assertFalse(decision['keep'])
        self.assertEqual(decision['reason'], 'insufficient_multiscale_support')

    def test_an_unsupported_reconstructed_surface_is_not_published_as_a_suspect(self):
        screened = screen_surface_changes(self.q, self.n, (self.a, self.b + [0, 0, .126]))
        ids = np.arange(len(self.q))
        decision = patch_noise_decision(screened, ids, {'occupied':.05})
        self.assertFalse(decision['keep'])
        self.assertEqual(decision['reason'], 'unsupported_source_surface')
        self.assertTrue(patch_noise_decision(screened, ids, {'occupied':.9})['keep'])


if __name__ == '__main__':
    unittest.main()
