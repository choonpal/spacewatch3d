"""Synthetic geometry tests; not a Rescan accuracy benchmark."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_change_noise import surface_pair_measurements,screen_surface_changes
from empirical_residual import (evaluate,spatial_quantile,validate_split,
                                region_decision,fixed_control,UNASSESSABLE)


class EmpiricalResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x,y=np.meshgrid(np.arange(-2,2.01,.035),np.arange(-2,2.01,.035))
        rng=np.random.default_rng(21)
        a=np.column_stack([x.ravel(),y.ravel(),rng.normal(0,.003,x.size)])
        b=a+[0,0,.08]
        cls.surfaces=[a,b]
        x,y=np.meshgrid(np.arange(-1.8,1.81,.16),np.arange(-1.5,1.51,.16))
        cls.q=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        n=np.tile([0.,0.,1.],(len(cls.q),1))
        cls.measurement=surface_pair_measurements(cls.q,n,[a,b])
        cls.config=json.loads((Path(__file__).resolve().parents[1]/'residual-roi-rescan.json').read_text())
        cls.config['maximum_reference_distance_m']=5.
        cls.config['rois']=[
            {'id':'fit','model':'floor','role':'fit','bounds':{'min':[-1.8,-1.5,-.1],'max':[-.7,1.5,.1]}},
            {'id':'check','model':'floor','role':'check','bounds':{'min':[.4,-1.5,-.1],'max':[1.8,1.5,.1]}}]

    def run_case(self,measurement=None,paired=None):
        return evaluate(self.q,measurement or self.measurement,np.ones(len(self.q)),
            np.ones(len(self.q)),paired if paired is not None else np.ones(self.measurement['distance_m'].shape,bool),self.config)

    def test_common_eight_cm_offset_is_not_a_change_when_operator_and_scales_match(self):
        result=self.run_case()
        self.assertTrue(result['models'][0]['enabled'])
        self.assertGreater(result['eligible'].sum(),100)
        np.testing.assert_allclose(result['models'][0]['thresholds_m'],.08,atol=1e-12)
        self.assertEqual(int(result['persistent'].sum()),0)

    def test_a_local_difference_above_reference_survives_without_subtracting_global_bias(self):
        m={k:v.copy() for k,v in self.measurement.items()}
        target=np.abs(self.q[:,0])<.2
        m['distance_m'][target]=.2
        result=self.run_case(m)
        self.assertTrue(result['persistent'][target].all())
        np.testing.assert_allclose(result['threshold_m'][target],.08,atol=1e-12)

    def test_check_residuals_cannot_train_the_reference_and_failed_transfer_defers(self):
        m={k:v.copy() for k,v in self.measurement.items()}
        m['distance_m'][self.q[:,0]>=.4]=.3
        result=self.run_case(m)
        np.testing.assert_allclose(result['models'][0]['thresholds_m'],.08,atol=1e-12)
        self.assertFalse(result['models'][0]['enabled'])
        self.assertTrue((result['status']==UNASSESSABLE).all())

    def test_missing_depth_is_unassessable_even_for_a_large_difference(self):
        paired=np.ones(self.measurement['distance_m'].shape,bool)
        target=np.abs(self.q[:,0])<.2;paired[target]=False
        result=self.run_case(paired=paired)
        self.assertTrue((result['status'][target]==UNASSESSABLE).all())

    def test_floor_reference_does_not_transfer_to_vertical_or_elevated_geometry(self):
        m={k:v.copy() for k,v in self.measurement.items()}
        target=np.abs(self.q[:,0])<.2;m['normals'][target]=[1,0,0]
        self.assertTrue((self.run_case(m)['status'][target]==UNASSESSABLE).all())

    def test_overlapping_fit_and_check_neighborhoods_are_rejected(self):
        config=copy.deepcopy(self.config);config['rois'][1]['bounds']['min'][0]=-.8
        with self.assertRaisesRegex(ValueError,'overlap'):validate_split(config)

    def test_overlapping_selected_surface_samples_cannot_become_an_approved_reference(self):
        result=evaluate(self.q,self.measurement,np.ones(len(self.q)),np.ones(len(self.q)),
            np.ones(self.measurement['distance_m'].shape,bool),self.config,support_disjoint={'fit':False})
        self.assertFalse(result['models'][0]['enabled'])
        self.assertEqual(result['models'][0]['fit_reason'],'fit_check_surface_support_overlap')
        self.assertTrue((result['status']==UNASSESSABLE).all())

    def test_one_exceeding_point_cannot_decide_an_object(self):
        self.assertEqual(region_decision(1.,.3,1,.001,self.config)[0],'insufficient_change_evidence')
        self.assertEqual(region_decision(.1,1.,20,.3,self.config)[0],'unassessable')

    def test_spatial_weighting_preserves_within_block_tail_and_is_density_invariant(self):
        p=np.zeros((100,3));p[50:,0]=1.
        values=np.zeros(100);values[42:50]=.2
        threshold,_=spatial_quantile(values,p)
        repeated_p=np.vstack([np.repeat(p[:50],10,axis=0),p[50:]])
        repeated_v=np.r_[np.repeat(values[:50],10),values[50:]]
        self.assertEqual(spatial_quantile(repeated_v,repeated_p)[0],threshold)
        self.assertEqual(threshold,.2)

    def test_fixed_comparison_reproduces_existing_screen_with_identical_observations(self):
        m=self.measurement
        fixed=fixed_control(m,np.ones(m['distance_m'].shape,bool))
        legacy=screen_surface_changes(self.q,m['normals'],self.surfaces)
        np.testing.assert_allclose(fixed['threshold_m'],legacy['envelope_m'])
        np.testing.assert_array_equal(fixed['persistent'],legacy['persistent'])

    def test_unsupported_roughness_or_density_has_no_floor_reference(self):
        target=np.abs(self.q[:,0])<.2
        for key,multiplier in [('spreads_m',15),('counts_by_epoch',10)]:
            m={k:v.copy() for k,v in self.measurement.items()};m[key][target]*=multiplier
            self.assertTrue((self.run_case(m)['status'][target]==UNASSESSABLE).all())


if __name__=='__main__':unittest.main()
