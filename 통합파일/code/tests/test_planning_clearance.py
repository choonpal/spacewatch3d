import copy
import sys
import unittest
import numpy as np
from shapely.geometry import box
from spacewatch_pipeline.io import ROOT, read
from spacewatch_pipeline.planning_policy import effective_planning_config
sys.path.insert(0,str(ROOT/'vendor/spacewatch3d_src'))
from spacewatch3d.tasks.task_5b_keyframes.models import Config
from spacewatch3d.tasks.task_5b_keyframes.routing import safe_segments


class ClearancePolicyTests(unittest.TestCase):
    def test_default_applies_without_capture_specific_branches(self):
        self.assertEqual(Config.from_dict().clearance_m,.2)
        for capture in ['dorm2','dorm3','unseen_building']:
            cfg={'captures':[{'id':capture}]}
            self.assertEqual(effective_planning_config(cfg)['clearance_m'],.2)
            old={'clearance_m':.3,'camera_height_m':1.7}
            snapshot=copy.deepcopy(old)
            value=effective_planning_config(cfg,old)
            self.assertEqual(value,{'clearance_m':.2,'camera_height_m':1.7})
            self.assertEqual(old,snapshot)

    def test_explicit_override_is_recordable_and_invalid_values_fail(self):
        self.assertEqual(effective_planning_config({'planning':{'clearance_m':.3}},{'clearance_m':.2})['clearance_m'],.3)
        for bad in [0,-.2,float('nan'),float('inf'),True,'0.2',None]:
            with self.assertRaises(ValueError):effective_planning_config({'planning':{'clearance_m':bad}})

    def test_all_shipped_run_configs_use_common_default(self):
        for path in (ROOT/'configs').glob('*.json'):
            cfg=read(path)
            if 'captures' in cfg:self.assertEqual(cfg['planning']['clearance_m'],.2,path.name)

    def test_narrow_corridor_allows_20cm_but_preserves_collision_checks(self):
        floor=box(0,0,4,4)
        obstacles=box(1.5,0,2,4)
        good=np.array([[[1.27,.5],[1.27,3.5]]]) # 23 cm obstacle clearance
        self.assertTrue(safe_segments(floor,obstacles,good,.2).all())
        self.assertFalse(safe_segments(floor,obstacles,good,.3).any())
        crossing=np.array([[[1.27,2],[2.3,2]],[[1.31,.5],[1.31,3.5]],[[.1,.5],[.1,3.5]]])
        self.assertFalse(safe_segments(floor,obstacles,crossing,.2).any())
