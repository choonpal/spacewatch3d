import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dorm_material_surfaces import common_wall_exclusions, reflective_pane_exclusion


def wall():
    x,y=np.meshgrid(np.linspace(-2,2,22),np.linspace(0,2.3,22));p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    f=[]
    for j in range(21):
        for i in range(21):
            k=j*22+i;f.extend([[k,k+1,k+22],[k+1,k+23,k+22]])
    return p,np.array(f)


class MaterialTests(unittest.TestCase):
    def test_common_wall_matching_is_invariant_to_the_world_origin(self):
        p,f=wall();a=.02;r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        q=p@r.T+[0,0,.03]
        for offset in [np.zeros(3),np.array([200.,0,1000.])]:
            masks,pairs=common_wall_exclusions([p+offset,q+offset],[f,f],2.3,.12)
            self.assertTrue(pairs);self.assertTrue(masks[0].mean()>.8);self.assertTrue(masks[1].mean()>.8)

    def test_glass_exclusion_keeps_geometry_in_front_of_the_pane(self):
        front=np.array([[-.1,-.1,1],[.1,-.1,1],[0,.1,1.]])
        behind=front.copy();behind[:,2]=3
        p=np.r_[front,behind];f=np.array([[0,1,2],[3,4,5]])
        mask=reflective_pane_exclusion(p,f,[[400,180],[630,180],[630,350],[400,350]],np.eye(4),np.eye(4),(np.array([0,0,-1.]),2.),2.)
        self.assertEqual(mask.tolist(),[False,True])

    def test_common_wall_exclusion_does_not_erase_a_small_foreground_surface(self):
        p,f=wall();start=len(p)
        foreground=np.array([[-.3,.4,.25],[.3,.4,.25],[-.3,1.,.25],[.3,1.,.25]])
        pp=np.r_[p,foreground];ff=np.r_[f,np.array([[start,start+1,start+2],[start+1,start+3,start+2]])]
        masks,_=common_wall_exclusions([pp,pp.copy()],[ff,ff],2.3,.12)
        for mask in masks:
            self.assertTrue(mask[:-2].mean()>.8);self.assertFalse(mask[-2:].any())


if __name__=='__main__':unittest.main()
