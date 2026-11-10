import unittest
from unittest.mock import patch, mock_open
from types import SimpleNamespace
import numpy as np
import importlib.util
from pathlib import Path
from florence_core import region_prompt, rasterize, mask_box_metrics
from project import associate


class Tests(unittest.TestCase):
    def test_region_quantization_edges(self):
        self.assertEqual(region_prompt([0, 0, 512, 512], (512,512)),
            '<REGION_TO_SEGMENTATION><loc_0><loc_0><loc_999><loc_999>')
        self.assertEqual(region_prompt([10,20,30,40],(100,200)),
            '<REGION_TO_SEGMENTATION><loc_100><loc_100><loc_300><loc_200>')

    def test_empty_is_not_box_fallback(self):
        m,q=rasterize({'polygons':[]},(32,32))
        self.assertFalse(m.any()); self.assertFalse(q['usable'])

    def test_multipolygon(self):
        m,q=rasterize({'polygons':[[[2,2,8,2,8,8,2,8],[20,20,26,20,26,26,20,26]]]},(32,32))
        self.assertTrue(m[4,4] and m[22,22]); self.assertFalse(m[12,12]); self.assertEqual(q['valid_polygons'],2)

    def test_invalid_polygon_rejected(self):
        for p in [[0,0,1,1], [1,1,2,2,3,3], [0,0,3,0,float('nan'),3]]:
            self.assertFalse(rasterize({'polygons':[[p]]},(32,32))[1]['usable'])

    def test_box_leakage(self):
        m=np.zeros((20,20),bool);m[2:6,2:6]=True
        self.assertEqual(mask_box_metrics(m,[2,2,5,5])['outside_box_fraction'],0)
        self.assertEqual(mask_box_metrics(m,[10,10,15,15])['outside_box_fraction'],1)

    def test_different_instances_in_same_view_never_merge(self):
        p=np.random.default_rng(1).normal(size=(50,3))*.01
        obs=[dict(label='any object',frame=0,view='front')]*2
        self.assertEqual(len(associate(obs,[p,p],.1)[0]),2)

    def test_crossframe_shape_agnostic_association(self):
        p=np.random.default_rng(1).normal(size=(80,3))
        obs=[dict(label='arbitrary object',frame=i,view='front') for i in range(3)]
        self.assertEqual(associate(obs,[p,p+.001,p+10],.05)[0],[[0,1],[2]])

    def test_class_labels_are_not_merged(self):
        p=np.random.default_rng(1).normal(size=(80,3))
        obs=[dict(label=l,frame=i,view='front') for i,l in enumerate(['black chair','blue chair'])]
        self.assertEqual(len(associate(obs,[p,p],.05)[0]),2)

    def test_complete_link_prevents_transitive_chain(self):
        p=np.array([[x,0,0] for x in np.arange(20)*.01])
        obs=[dict(label='object',frame=i,view='front') for i in range(3)]
        groups,_=associate(obs,[p,p+.04,p+.08],.06,.7)
        self.assertEqual(groups,[[0,1],[2]])

    def test_panorama_projection_roundtrip_for_recentered_views(self):
        local=Path(__file__).resolve().parents[2]/'vendor/discovery/segmentation_2d3d/scene_discovery/common.py'
        if not local.exists():local=Path('/workspace/spacewatch_pipeline_v4/vendor/discovery/segmentation_2d3d/scene_discovery/common.py')
        spec=importlib.util.spec_from_file_location('panorama_geometry_test',local)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for view in [dict(yaw=179,pitch=-35),dict(yaw=-90,pitch=-20),dict(yaw=0,pitch=0)]:
            visible,index=module.erp_to_view((518,1036),view,512,90)
            mx,my=module.perspective_map(1036,518,view,512,90)
            source=np.flatnonzero(visible)[::100]
            y,x=np.divmod(source,1036);j=index[source]
            dx=np.abs(mx.ravel()[j]-x);dx=np.minimum(dx,1036-dx)
            dy=np.abs(my.ravel()[j]-y)
            self.assertLess(np.quantile(dx,.99),2)
            self.assertLess(np.quantile(dy,.99),1)


if __name__=='__main__':unittest.main()
