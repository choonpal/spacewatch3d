import unittest
from tools.mesh_simsac.label_rich_frames import select_label_rich

class LabelRichFrameTests(unittest.TestCase):
    def f(self,i,p,labels,sharp=1):return dict(id=str(i),common_position=[p,0,0],label_counts=labels,sharpness=sharp)
    def test_variety_beats_distance_and_total_area(self):
        frames=[self.f(1,.01,{'door':10000},999),self.f(2,1,{'door':50,'box':50,'clothes':50},1)]
        self.assertEqual(select_label_rich(frames,[0,0,0],2,1)[0]['id'],'2')
    def test_spatial_gate_cannot_be_overridden_by_labels(self):
        self.assertEqual(select_label_rich([self.f(1,5,{'door':99,'box':99})],[0,0,0],2),[])
    def test_background_and_tiny_label_noise_do_not_inflate_variety(self):
        got=select_label_rich([self.f(1,0,{'wall':50000,'floor':50000,'ceiling':50000,'box':31,'door':32})],[0,0,0],1)[0]
        self.assertEqual(got['visible_labels'],['door']);self.assertEqual(got['label_type_count'],1)
    def test_ordinary_person_and_clothes_are_object_labels(self):
        got=select_label_rich([self.f(1,0,{'person':90,'clothes':40})],[0,0,0],1)[0]
        self.assertEqual(got['label_type_count'],2)
    def test_budget_is_per_keypoint_not_per_label(self):
        frames=[self.f(i,i/100,{'door':40+i,'chair':40+i,'table':40+i}) for i in range(8)]
        self.assertEqual(len(select_label_rich(frames,[0,0,0],1,3)),3)
    def test_area_then_sharpness_then_distance_break_ties(self):
        a=self.f(1,.1,{'box':50},999);b=self.f(2,.8,{'box':60},1)
        self.assertEqual(select_label_rich([a,b],[0,0,0],1,1)[0]['id'],'2')
        b=self.f(2,.8,{'box':50},1000)
        self.assertEqual(select_label_rich([a,b],[0,0,0],1,1)[0]['id'],'2')

if __name__=='__main__':unittest.main()
