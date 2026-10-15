"""All-orientation geometry and post-detection evaluation regressions."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from all_surface_change import clearance_measurements, classify_clearance, evaluate_reference_items


def grid():
    u,v=np.meshgrid(np.arange(-.5,.501,.025),np.arange(-.5,.501,.025))
    return np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]


class AllSurfaceTests(unittest.TestCase):
    def measure(self,rotation,clearance):
        surface=grid()@rotation.T
        normals=np.tile(np.array([0.,0,1.])@rotation.T,(len(surface),1))
        q=np.array([[0.,0,0]])@rotation.T
        return clearance_measurements(q,normals[:1],np.array([0]),[surface,surface],
            [normals,normals],[np.full(len(surface),clearance),np.zeros(len(surface))])

    def test_vertical_and_inclined_changes_have_the_same_decision_as_horizontal(self):
        for angle in (0,.6,np.pi/2):
            c,s=np.cos(angle),np.sin(angle);r=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
            m=self.measure(r,.18)
            result=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.001]))
            self.assertTrue(result['persistent'][0])
            self.assertEqual(result['status'][0],1)

    def test_centimeter_reconstruction_noise_is_not_a_change_in_any_orientation(self):
        r=np.array([[0.,0,1.],[0,1.,0],[-1.,0,0]])
        m=self.measure(r,.02)
        result=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.001]))
        self.assertFalse(result['persistent'][0]);self.assertEqual(result['status'][0],0)

    def test_a_few_large_outliers_do_not_change_a_stable_local_surface(self):
        p=grid();n=np.tile([0.,0,1.],(len(p),1));d=np.full(len(p),.015);d[::15]=.4
        m=clearance_measurements(np.array([[0.,0,0]]),n[:1],np.array([0]),[p,p],[n,n],[d,d])
        result=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.001]))
        self.assertFalse(result['persistent'][0])

    def test_missing_other_mesh_is_not_a_confirmed_disappearance(self):
        m=self.measure(np.eye(3),np.nan)
        r=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.001]))
        self.assertEqual(r['status'][0],2)
        self.assertNotIn('removed',r)

    def test_background_is_excluded_without_excluding_a_vertical_furniture_surface(self):
        m=self.measure(np.eye(3),.18)
        r=classify_clearance(m,np.array([.038,.036,.034]),np.array([0]),np.array([.001]))
        self.assertEqual(r['status'][0],3);self.assertFalse(r['persistent'][0])

    def test_no_reference_is_not_silently_replaced_by_fixed_two_cm(self):
        m=self.measure(np.eye(3),.18)
        r=classify_clearance(m,np.full(3,np.nan),np.array([-1]),np.array([.001]))
        self.assertEqual(r['status'][0],2)

    def test_large_local_roughness_raises_the_floor_noise_budget(self):
        m=self.measure(np.eye(3),.09);m['roughness_m'][:]=.05
        r=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.001]))
        self.assertTrue(np.all(r['threshold_m']>.13));self.assertFalse(r['persistent'][0])

    def test_reconstruction_without_nearby_raw_source_support_is_deferred(self):
        m=self.measure(np.eye(3),.18)
        r=classify_clearance(m,np.array([.038,.036,.034]),np.array([-1]),np.array([.08]))
        self.assertEqual(r['status'][0],2);self.assertFalse(r['persistent'][0])

    def test_neighboring_object_inside_the_same_box_cannot_count_as_the_reference(self):
        p=np.c_[np.arange(12)*.03,np.zeros(12),np.ones(12)]
        items=[{'id':1,'name':'target','type':'moved','before_instance_id':10,'after_instance_id':10}]
        r=evaluate_reference_items(items,p,np.ones(12,bool),np.zeros(12,int),np.full(12,20),
            np.ones(12,bool),np.full((2,12),.006))
        self.assertEqual(r[0]['outcome'],'missed')

    def test_connected_reference_surface_can_be_captured_without_using_its_change_type(self):
        p=np.c_[np.arange(12)*.03,np.zeros(12),np.ones(12)]
        items=[{'id':1,'name':'target','type':'moved','before_instance_id':10,'after_instance_id':10}]
        args=(p,np.ones(12,bool),np.zeros(12,int),np.full(12,10),np.ones(12,bool),np.full((2,12),.006))
        a=evaluate_reference_items(items,*args)[0]
        b=evaluate_reference_items([{**items[0],'type':'removed'}],*args)[0]
        self.assertEqual(a['outcome'],'captured');self.assertEqual(b['outcome'],'captured')
        self.assertGreater(a['candidate_area_m2'],.04)

    def test_scattered_or_unsupported_hits_are_not_a_captured_reference(self):
        p=np.c_[np.arange(12)*.4,np.zeros(12),np.ones(12)]
        items=[{'id':1,'name':'target','type':'moved','before_instance_id':10,'after_instance_id':10}]
        args=(p,np.ones(12,bool),np.zeros(12,int),np.full(12,10))
        partial=evaluate_reference_items(items,*args,np.ones(12,bool),np.full((2,12),.006))[0]
        missing=evaluate_reference_items(items,*args,np.zeros(12,bool),np.full((2,12),.006))[0]
        self.assertEqual(partial['outcome'],'partial');self.assertEqual(missing['outcome'],'missed')


if __name__=='__main__':unittest.main()
