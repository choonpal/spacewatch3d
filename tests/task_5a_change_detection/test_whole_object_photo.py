import unittest
import numpy as np
from spacewatch3d.tasks.task_5a_change_detection.whole_object_photo import warp_whole_cutout


class WholeObjectPhotoTests(unittest.TestCase):
    def test_entire_original_mask_including_thin_legs_survives_without_mesh_clipping(self):
        mask=np.zeros((32,64),bool)
        mask[9:17,27:37]=True
        mask[18:21,26:38]=True
        mask[16:20,26]=True;mask[16:20,37]=True
        mask[21:27,27]=True;mask[21:27,36]=True
        rgb=np.full((32,64,3),[71,121,191],dtype=np.uint8)
        actual=warp_whole_cutout(rgb,mask,np.eye(4),np.eye(4),np.array([0.,0.,2.]))
        np.testing.assert_array_equal(actual[:,:,3]>0,mask)
        np.testing.assert_array_equal(actual[mask,:3],rgb[mask])
        self.assertEqual(actual[26,27,3],255)
        self.assertEqual(actual[26,36,3],255)
        self.assertEqual(actual[17,31,3],0)

    def test_interpolation_does_not_bring_background_colors_into_object_edges(self):
        mask=np.zeros((32,64),bool);mask[10:24,27:37]=True
        rgb=np.full((32,64,3),[255,0,0],dtype=np.uint8);rgb[mask]=[0,180,220]
        target=np.eye(4);target[0,3]=.17
        actual=warp_whole_cutout(rgb,mask,np.eye(4),target,np.array([0.,0.,2.]))
        visible=actual[:,:,3]>0
        self.assertTrue(visible.any())
        self.assertEqual(actual[visible,0].max(),0)
        self.assertTrue(np.all(actual[visible,1]>=179))

    def test_side_view_does_not_slice_or_foreshorten_the_object_outline(self):
        mask=np.zeros((64,128),bool);mask[23:41,55:73]=True
        rgb=np.full((64,128,3),120,dtype=np.uint8)
        target=np.eye(4);target[:3,3]=[1.2,0.,.4]
        actual=warp_whole_cutout(rgb,mask,np.eye(4),target,np.array([0.,0.,2.]))
        yy,xx=np.nonzero(actual[:,:,3]>0)
        self.assertLessEqual(abs((xx.max()-xx.min()+1)-18),1)
        self.assertLessEqual(abs((yy.max()-yy.min()+1)-18),1)

    def test_full_panorama_seam_keeps_both_parts_of_the_cutout(self):
        mask=np.zeros((32,64),bool);mask[14:19,[0,1,62,63]]=True
        rgb=np.full((32,64,3),120,dtype=np.uint8)
        actual=warp_whole_cutout(rgb,mask,np.eye(4),np.eye(4),np.array([0.,0.,-2.]))
        np.testing.assert_array_equal(actual[:,:,3]>0,mask)

if __name__=='__main__':unittest.main()
