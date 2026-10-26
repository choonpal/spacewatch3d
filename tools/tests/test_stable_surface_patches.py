import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_surface_patches import discover_stable_patches, screen_local_directions, heldout_direction_checks


def plane(axis, offset=0.):
    uv=np.linspace(-.9,.9,49); a,b=np.meshgrid(uv,uv)
    p=np.zeros((a.size,3));other=[k for k in range(3) if k!=axis]
    p[:,axis]=offset;p[:,other[0]]=a.ravel();p[:,other[1]]=b.ravel()
    n=np.tile(np.eye(3)[axis],(len(p),1));return p,n


class StablePatchTests(unittest.TestCase):
    def test_unrestricted_direction_finds_a_side_surface_with_common_map_shift(self):
        p,n=plane(0,.1);q=p+np.array([.04,.003,-.002])
        a,b,normals,ids,catalog=discover_stable_patches(p,q,n,n)
        self.assertGreater(len(catalog),3)
        self.assertGreater(abs(np.median(normals[:,0])),.99)
        self.assertAlmostEqual(float(np.median(np.abs(np.einsum('ij,ij->i',a-b,normals)))),.04,places=3)

    def test_nonoverlapping_changed_surface_and_wrong_normals_are_not_references(self):
        p,n=plane(0,.1)
        self.assertEqual(len(discover_stable_patches(p,p+np.array([.3,0.,0.]),n,n)[4]),0)
        wrong=np.tile([0.,1.,0.],(len(p),1))
        self.assertEqual(len(discover_stable_patches(p,p,n,wrong)[4]),0)

    def test_small_edge_fragments_do_not_manufacture_a_new_plane_direction(self):
        p=np.column_stack([np.zeros(80),np.linspace(-1,1,80),np.zeros(80)])
        n=np.tile([1.,0.,0.],(len(p),1))
        self.assertEqual(len(discover_stable_patches(p,p,n,n)[4]),0)

    def test_isolated_extra_patch_does_not_resolve_a_missing_direction(self):
        p,n=plane(2);a,m=plane(0,.7)
        p=p[::8];n=n[::8];a=a[::8];m=m[::8]
        source=np.r_[p,a];normal=np.r_[n,m];ids=np.r_[np.full(len(p),-1),np.ones(len(a),int)]
        keep,diagnostics=screen_local_directions(source,source,normal,ids,[0.,0.,0.])
        self.assertTrue(keep[:len(p)].all());self.assertFalse(keep[len(p):].any())
        self.assertFalse(next(d for d in diagnostics if not d['original_direction'])['admitted'])

    def test_new_direction_needs_multiple_spatial_train_and_check_patches(self):
        p,n=plane(0,1.05);p=p[::3];n=n[::3]
        cells=np.floor(p/.35).astype(int);_,ids=np.unique(cells,axis=0,return_inverse=True)
        keep,diagnostics=screen_local_directions(p,p,n,ids,[0.,0.,0.])
        self.assertTrue(keep.all());self.assertTrue(diagnostics[0]['admitted'])
        self.assertGreaterEqual(diagnostics[0]['check_blocks'],2)

    def test_new_direction_cannot_pass_when_only_training_pieces_agree(self):
        p,n=plane(0,1.05);p=p[::3];n=n[::3]
        cells=np.floor(p/.35).astype(np.int64);_,ids=np.unique(cells,axis=0,return_inverse=True)
        check=(cells@np.array([73856093,19349663,83492791]))%5==0
        q=p.copy();q[check,0]+=.1
        result=heldout_direction_checks(q,p,n,ids,[0.,0.,0.],{'usable':True,'correction_after_to_before':np.eye(4).tolist()})
        self.assertFalse(result[0]['passed']);self.assertGreater(result[0]['check_p95_m'],.09)


if __name__=='__main__':unittest.main()
