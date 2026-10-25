import importlib.util,unittest
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('pair_models',Path(__file__).parents[3]/'tools/scenediff_mesh/run_pair_models.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PairModelAdapters(unittest.TestCase):
    def test_local_cost_volume_offset_direction_and_padding(self):
        a=torch.zeros(1,2,5,6);b=torch.zeros_like(a)
        a[0,0,2,3]=2;b[0,0,4,1]=3
        result=module.local_correlation(a,b)
        self.assertEqual(result.shape,(1,81,5,6))
        self.assertEqual(float(result[0,6*9+2,2,3]),3)
        self.assertEqual(int(torch.count_nonzero(result)),1)

    def test_erp_projection_cardinal_yaws_and_periodicity(self):
        w,h=1024,512
        longitude=((np.arange(w)+.5)/w-.5)*2*np.pi
        probability=np.broadcast_to(.5+.25*np.cos(longitude),(h,w)).copy()
        for yaw,expected in [(0,.75),(90,.5),(180,.25),(270,.5)]:
            projected=module.project_probability(probability,yaw,size=3)
            self.assertAlmostEqual(float(projected[1,1]),expected,places=5)
        np.testing.assert_allclose(module.project_probability(probability,0),module.project_probability(probability,360),atol=1e-6)

    def test_projection_preserves_constant_probabilities(self):
        p=np.full((40,80),.37,dtype='f4')
        np.testing.assert_allclose(module.project_probability(p,180),.37,atol=1e-6)

if __name__=='__main__':unittest.main()
