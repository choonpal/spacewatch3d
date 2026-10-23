import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dorm_surface_change import (classify_patch, coherent_patches, compare_surfaces,
                                 independent_views)


def connected_surface(size=36):
    a = np.linspace(-1.2, 1.2, size)
    x, z = np.meshgrid(a, a)
    p = np.c_[x.ravel(), np.full(size*size, .6), z.ravel()]
    f = []
    for i in range(size-1):
        for j in range(size-1):
            k = i*size+j
            f.extend([[k, k+1, k+size], [k+1, k+size+1, k+size]])
    return p, np.array(f, dtype=np.uint32)


def view(frame, x, free=1., finite=1.):
    return {'frame': frame, 'camera_in_map': [x, 1., 0.], 'score': 1.,
            'visible_fraction': 1., 'free_fraction': free, 'finite_fraction': finite}


class GeometryChangeTests(unittest.TestCase):
    def test_connected_surface_changes_are_localized_without_instance_boundaries_in_any_orientation(self):
        p, f = connected_surface(); after = p.copy()
        moved = (np.abs(p[:, 0]) < .48) & (np.abs(p[:, 2]) < .48)
        after[moved, 1] += .35
        for rotation in [np.eye(3), np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])]:
            a, b = p@rotation, after@rotation
            d, _, _ = compare_surfaces(b, f, a, f)
            patches, _ = coherent_patches(b, f, d, .08, 2.)
            self.assertTrue(patches)
            ids = np.concatenate([patch['face_ids'] for patch in patches])
            self.assertLess(len(ids), len(f)//2)
            # Original, unchanged parts of the same connected mesh stay clear.
            self.assertTrue(np.all(np.abs(p[f[ids]].mean(1)[:, [0, 2]]) < .6))

    def test_sub_tolerance_registration_and_reconstruction_noise_do_not_create_regions(self):
        p, f = connected_surface(); rng = np.random.default_rng(108)
        noisy = p + rng.normal(0, .008, p.shape) + [0., .025, 0.]
        d, _, _ = compare_surfaces(noisy, f, p, f)
        patches, _ = coherent_patches(noisy, f, d, .08, 2.)
        self.assertEqual(patches, [])

    def test_isolated_large_noise_spike_is_not_a_change_region(self):
        p, f = connected_surface(); noisy = p.copy(); noisy[len(p)//2, 1] += .6
        d, _, _ = compare_surfaces(noisy, f, p, f)
        patches, _ = coherent_patches(noisy, f, d, .08, 2.)
        self.assertEqual(patches, [])

    def test_missing_target_coverage_is_deferred_despite_a_large_geometry_difference(self):
        own = [view(0, 0), view(5, .3)]
        # Rays miss the reconstructed target: no hit is not free-space proof.
        absent = [view(0, 0, finite=0.), view(5, .3, finite=0.)]
        decision, reason, _, _ = classify_patch([{'own_views': own, 'opposite_views': absent}], True, 2.)
        self.assertEqual((decision, reason), ('deferred', 'occlusion_or_incomplete_coverage'))

    def test_a_zone_without_alignment_direction_support_stays_deferred(self):
        observations = [view(0, 0), view(5, .3)]
        sides = [{'own_views': observations, 'opposite_views': observations}]
        self.assertEqual(classify_patch(sides, True, 2.)[0], 'candidate')
        self.assertEqual(classify_patch(sides, False, 2.)[:2], ('deferred', 'alignment_underconstrained'))
        repeated = [view(0, 0), view(5, .01), view(10, .02)]
        self.assertEqual(len(independent_views(repeated, 'free_fraction', .35, 2.)), 1)


if __name__ == '__main__': unittest.main()
