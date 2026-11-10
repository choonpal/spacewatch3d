"""Contract/parity checks with small fresh in-memory observation draws."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import json
import os
import runpy
import tempfile
import time
import unittest

import numpy as np
from spacewatch_pipeline import paper_selector as m


def draws(visible, quality, importance=None):
    v, q = np.asarray(visible, bool), np.asarray(quality, float)
    importance = np.ones(q.shape[1]) if importance is None else np.asarray(importance, float)
    return [dict(visible=v.copy(), quality=q.copy(), weights=np.ones(v.shape[1]),
                 regions=np.array(['r']*v.shape[1]), importance=importance.copy()) for _ in range(3)]


class SelectorTests(unittest.TestCase):
    def test_reference_source_formula_parity(self):
        path=os.environ.get('PAPER_SELECTOR_REFERENCE_BENCHMARK')
        if not path:self.skipTest('Set PAPER_SELECTOR_REFERENCE_BENCHMARK to verify original source formulas')
        original=runpy.run_path(path,run_name='reference_benchmark')
        rng=np.random.default_rng(21)
        ds=draws(rng.random((8,15))>.5,rng.random((8,5)),[1,2,3,4,5])
        ds[1]['visible']=~ds[1]['visible']
        ds[2]['quality']=ds[2]['quality']*.8
        reference=[]
        for d in ds:
            reference.append({**d,'space_coef':original['spatial_coefficients'](d['visible'],d['weights'],d['regions']),
                              'importance_coef':d['importance']/d['importance'].sum()})
        b=m.CPUBackend(ds)
        for selected in ([],[0],[2,4],[0,3,5]):
            for sw,ow in ((1.,0.),(0.,1.),(.6,.4)):
                expected=original['utility'](reference,selected,sw,ow)
                self.assertAlmostEqual(b.utility(selected,sw,ow),expected,places=12)
                for i in range(8):
                    if i not in selected:
                        delta=original['utility'](reference,selected+[i],sw,ow)-expected
                        self.assertAlmostEqual(b.gains(selected,sw,ow)[i],delta,places=12)

    def test_gains_and_pairs_equal_exhaustive_objective(self):
        rng=np.random.default_rng(8)
        ds=draws(rng.random((9,17))>.5,rng.random((9,4)),[1,4,2,3])
        ds[1]['visible']=~ds[1]['visible']
        b=m.CPUBackend(ds)
        for base in ([],[2],[2,4]):
            gains=b.gains(base,.6,.4)
            for i in range(b.n):
                if i not in base:
                    expected=b.utility(base+[i],.6,.4)-b.utility(base,.6,.4)
                    self.assertAlmostEqual(gains[i],expected,places=12)
            pool=[i for i in range(b.n) if i not in base]
            pair,gain=b.best_pair(base,pool,.6,.4)
            exact=max(b.utility(base+list(p),.6,.4)-b.utility(base,.6,.4) for p in itertools.combinations(pool,2))
            self.assertAlmostEqual(gain,exact,places=12)
            self.assertAlmostEqual(b.utility(base+pair,.6,.4)-b.utility(base,.6,.4),exact,places=12)

    def test_raw_quality_fullpool_denominator_and_strict_stop(self):
        b=m.CPUBackend(draws([[1],[1]],[[.5,0],[0,.5]],[1,1]))
        self.assertAlmostEqual(m.object_ceiling(b),.5)
        class Fake:
            n=1
            def gains(self,selected,sw,ow):return np.array([.01*sw+.03*ow])
        self.assertFalse(m.stopping(Fake(),[],0,1,.02,1)[0])
        self.assertTrue(m.stopping(Fake(),[],0,1,.02,1.01)[0])
        self.assertFalse(m.stopping(Fake(),[],1,0,.01,1)[0])
        self.assertTrue(m.stopping(Fake(),[],1,0,.0101,1)[0])

    def test_separate_mixed_stop(self):
        class Fake:
            n=1
            def gains(self,selected,sw,ow):return np.array([.05*sw+.001*ow])
        done,info=m.stopping(Fake(),[],.6,.4,.02,1)
        self.assertFalse(done)
        self.assertTrue(info['object_done'])
        self.assertFalse(info['space_done'])

    def test_new_measurements_drive_choices_not_frame_ids_or_counts(self):
        ds=draws(np.eye(4,dtype=bool),np.eye(4))
        r=m.select_draws(ds,0,1,options={'deadline_seconds':5})
        self.assertEqual(set(r['selected']),set(range(4)))
        self.assertTrue(r['marginal_stop_verified'])
        q=np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        s=m.select_draws(draws(np.eye(4,dtype=bool),q),0,1,options={'deadline_seconds':5})
        self.assertEqual(s['selected'],[0])
        q=q[[2,3,1,0]]
        t=m.select_draws(draws(np.eye(4,dtype=bool),q),0,1,options={'deadline_seconds':5})
        self.assertEqual(t['selected'],[3])
        self.assertFalse(t['holdouts_used_for_selection'])
        self.assertFalse(t['warm_start_bank_used'])

    def test_cap_not_success_and_required_preserved(self):
        ds=draws(np.eye(5,dtype=bool),np.eye(5))
        r=m.select_draws(ds,.6,.4,required=[4],options={'max_views':2,'deadline_seconds':5})
        self.assertEqual(r['count'],2)
        self.assertIn(4,r['selected'])
        self.assertEqual(r['stop_reason'],'view_cap')
        self.assertFalse(r['marginal_stop_verified'])

    def test_deadline_not_success(self):
        ds=draws(np.eye(5,dtype=bool),np.eye(5))
        r=m.select_draws(ds,.6,.4,options={'deadline_seconds':1e-12})
        self.assertEqual(r['stop_reason'],'time_limit')
        self.assertFalse(r['marginal_stop_verified'])

    def test_reject_single_draw_and_unknown_frozen_overrides(self):
        ds=draws([[1],[0]],[[1],[0]])
        with self.assertRaises(ValueError):m.select_draws(ds[:1],.6,.4)
        with self.assertRaises(ValueError):m.select_draws(ds,.6,.4,options={'selected':[0]})

    def test_two_swap_improves_trap(self):
        # Check a real pair search, beyond syntax or duplicated implementation.
        rng=np.random.default_rng(77)
        ds=draws(rng.random((7,20))>.55,rng.random((7,3)))
        b=m.CPUBackend(ds)
        out,history=m.exchange(b,[0,1],.6,.4,time.monotonic()+5)
        self.assertEqual(len(set(out)),2)
        exact=max(b.utility(list(p),.6,.4) for p in itertools.combinations(range(7),2))
        self.assertAlmostEqual(b.utility(out,.6,.4),exact,places=12)

    def test_install_consumes_native_draws_and_blocks_posthoc_changes(self):
        ds=draws(np.eye(4,dtype=bool),np.eye(4))
        cfg=SimpleNamespace(route_aware_selection=False,lateral_candidate_refinement=False,max_link_m=None,
            coverage_repair=True,coverage_repair_runs=2,space_weight=.6,object_weight=.4)
        native=[]
        for d in ds:
            native.append(SimpleNamespace(**d,declared_regions={'r'},config=cfg))
        class Optimization:
            objectives=native
            def measure(self,selected):
                return m.CPUBackend(ds).utility(selected,.6,.4),{'unmet_targets':['original_target_stays_unmet']}
        planner=SimpleNamespace()
        planner.make_objectives=lambda *args:native
        planner.select_keyframes=lambda *args:None
        def create():
            planner.make_objectives([{'seed':i} for i in range(3)],list(range(4)),
                [{'instance_id':str(i)} for i in range(4)],cfg,['r'])
            chosen,history,report=planner.select_keyframes(Optimization())
            views=[{'route_node_id':str(i)} for i in chosen]
            if getattr(planner,'corrupt',False):views=[]
            return {'candidate_report':{'candidates':[{'route_node_id':str(i)} for i in range(4)]},
                'viewpoints':views,'coverage_report':{'selection':{},'unmet_targets':['original_target_stays_unmet']},
                'provenance':{},'planning_status':'complete','failures':[]},{}
        planner.create_plan=create
        with tempfile.TemporaryDirectory() as folder:
            receipt=Path(folder)/'selector.json'
            restore=m.install(planner,receipt,{'max_views':2,'deadline_seconds':5})
            plan,_=planner.create_plan()
            self.assertEqual(plan['planning_status'],'incomplete')
            self.assertIn('original_target_stays_unmet',plan['coverage_report']['unmet_targets'])
            r=json.loads(receipt.read_text())
            self.assertEqual(len(r['observations']),3)
            self.assertTrue(r['postselection_identity_verified'])
            planner.corrupt=True
            with self.assertRaises(RuntimeError):planner.create_plan()
            restore()
            self.assertIs(planner.create_plan,create)


if __name__=='__main__':unittest.main()
