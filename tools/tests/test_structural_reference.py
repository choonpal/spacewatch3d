"""Structural background regression tests; no instance or change labels."""
import sys
import unittest
import copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structural_reference import (extract_planes, match_planes, background_assignment,
    reference_split, oriented_measurements, calibrate_plane, classify_targets,candidate_components)


def plane_grid(axis, value, extent0, extent1, step=.06):
    a, b = np.meshgrid(np.arange(*extent0, step), np.arange(*extent1, step))
    p = np.zeros((a.size, 3)); other = [i for i in range(3) if i != axis]
    p[:, axis] = value; p[:, other[0]] = a.ravel(); p[:, other[1]] = b.ravel()
    n = np.zeros_like(p); n[:, axis] = 1
    return p, n


class StructuralReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        floor, fn = plane_grid(2, 0, (-3, 3.01), (-2, 2.01))
        wall, wn = plane_grid(1, -2, (-3, 3.01), (0, 2.51))
        table, tn = plane_grid(2, .6, (-.6, .61), (-.5, .51))
        cls.a = np.vstack([floor, wall, table]); cls.n = np.vstack([fn, wn, tn])
        cls.b = cls.a + [0, .025, .02]
        cls.planes = [extract_planes(cls.a, cls.n), extract_planes(cls.b, cls.n)]
        cls.pairs = match_planes(cls.planes, [cls.a, cls.b])

    def test_floor_and_wall_are_separate_and_table_is_not_a_reference(self):
        self.assertEqual(sorted(p['kind'] for p in self.planes[0]), ['floor', 'wall'])
        self.assertEqual(len(self.pairs), 2)
        self.assertLess(abs(next(p['center'][2] for p in self.planes[0] if p['kind']=='floor')), .01)

    def test_common_background_is_excluded_but_table_above_it_survives(self):
        q = np.array([[1., 0, 0], [1., -2, 1.], [0, 0, .6]])
        n = np.array([[0, 0, 1.], [0, 1., 0], [0, 0, 1.]])
        assigned = background_assignment(q, n, self.pairs, [self.a, self.b])
        self.assertTrue((assigned[:2] >= 0).all())
        self.assertEqual(assigned[2], -1)

    def test_a_low_furniture_base_is_not_erased_by_a_floor_proximity_radius(self):
        assigned=background_assignment(np.array([[1.,0,.07]]),np.array([[0.,0,1.]]),
            self.pairs,[self.a,self.b],require_both=False)
        self.assertEqual(assigned[0],-1)

    def test_missing_counterpart_is_not_a_common_background_plane(self):
        self.assertEqual(match_planes([self.planes[0], []], [self.a, self.b]), [])

    def test_a_globally_common_floor_is_not_a_change_when_furniture_occludes_one_epoch(self):
        q=np.array([[1.,0,0]]);n=np.array([[0.,0,1.]])
        pairs=copy.deepcopy(self.pairs)
        floor=next(p for p in pairs if p['kind']=='floor')
        ids=floor['planes'][1]['sample_ids']
        floor['planes'][1]['sample_ids']=ids[np.linalg.norm(self.b[ids,:2]-q[:,:2],axis=1)>.3]
        common=background_assignment(q,n,pairs,[self.a,self.b],require_both=True)
        self.assertEqual(common[0],-1)
        result=background_assignment(q,n,pairs,[self.a,self.b],require_both=False)
        self.assertGreaterEqual(result[0],0)

    def test_only_connected_supported_surface_patches_are_published(self):
        q=np.c_[np.arange(12)*.1,np.zeros(12),np.zeros(12)];q[-1]=[8,0,0]
        result={'persistent':np.ones(12,bool),'status':np.ones(12,np.uint8),'reason':np.full(12,'exceeds',dtype='U64')}
        groups=candidate_components(q,result,np.full((2,12),.006))
        self.assertEqual([len(g) for g in groups],[11]);self.assertEqual(int(result['persistent'].sum()),11)
        self.assertEqual(result['status'][-1],0)

    def test_reference_split_has_a_buffer_wider_than_both_measurement_radii(self):
        u = np.arange(-4, 4.01, .02)
        uv = np.c_[u, np.zeros_like(u)]
        fit, check = reference_split(uv)
        self.assertGreater(fit.sum(), 30); self.assertGreater(check.sum(), 30)
        self.assertGreater(np.min(np.abs(u[fit, None] - u[check])), .56)

    def test_vertical_furniture_bridge_cannot_mix_floor_and_table(self):
        floor, fn = plane_grid(2, 0, (-.4, .41), (-.4, .41), .02)
        top, tn = plane_grid(2, .5, (-.28, .29), (-.28, .29), .02)
        leg, ln = plane_grid(0, .08, (-.08, .09), (0, .51), .02)
        a = floor; an = fn
        b = np.vstack([floor+[0, 0, .02], top, leg]); bn = np.vstack([fn, tn, ln])
        measured = oriented_measurements(np.array([[0., 0, 0]]), np.array([[0., 0, 1.]]), [a, b], [an, bn])
        np.testing.assert_allclose(measured['distance_m'], .02, atol=1e-12)

    def test_check_distances_do_not_train_threshold_and_failed_reference_is_disabled(self):
        uv = np.c_[np.arange(-4, 4.01, .08), np.zeros(101)]
        d = np.full((len(uv), 3), .02); valid = np.ones_like(d, bool)
        fit, check = reference_split(uv)
        model = calibrate_plane(uv, d, valid)
        self.assertTrue(model['enabled'])
        changed = d.copy(); changed[check] = .2
        rejected = calibrate_plane(uv, changed, valid)
        np.testing.assert_allclose(rejected['thresholds_m'], model['thresholds_m'])
        self.assertFalse(rejected['enabled'])
        np.testing.assert_allclose(rejected['bias_coefficients_m'],model['bias_coefficients_m'])

    def test_location_dependent_plane_bias_is_fitted_without_using_holdout(self):
        u,v=np.meshgrid(np.arange(-4,4.01,.08),np.arange(-1,1.01,.16))
        uv=np.c_[u.ravel(),v.ravel()]
        distance=np.tile((.02+.01*uv[:,0]-.003*uv[:,1])[:,None],(1,3))
        model=calibrate_plane(uv,distance,np.ones_like(distance,bool))
        self.assertTrue(model['enabled'])
        np.testing.assert_allclose(model['bias_coefficients_m'],[[.02,.01,-.003]]*3,atol=1e-12)
        self.assertLess(max(model['thresholds_m']),1e-12)

    def test_background_is_not_counted_as_change_and_large_furniture_difference_remains(self):
        d = np.array([[.02]*3, [.25]*3, [.25]*3])
        measurement = {'distance_m':d, 'spreads_m':np.full((3,3,2),.003),
                       'counts_by_epoch':np.full((3,3,2),20)}
        reference = np.full((3,3),.03)
        paired = np.array([[True]*3,[True]*3,[False]*3])
        result = classify_targets(measurement, reference, paired, np.array([0,-1,-1]))
        self.assertEqual(result['status'].tolist(), [3,1,2])
        self.assertFalse(result['persistent'][0]); self.assertTrue(result['persistent'][1])


if __name__ == '__main__':
    unittest.main()
