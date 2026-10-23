import unittest
import numpy as np
from spacewatch3d.tasks.task_5a_change_detection.object_photo_tour import object_faces, photo_ghost
from spacewatch3d.tasks.task_5a_change_detection.surface_geometry import triangle_scene

class ObjectPhotoTourTests(unittest.TestCase):
    def test_foreground_boxes_cannot_reintroduce_known_wall_floor_or_unknown_low_floor(self):
        classes=np.array([0,3,5,8,14,255,255,19,10])
        centers=np.ones((9,3));centers[5,1]=0
        result=object_faces(classes,np.ones(9,bool),centers,2.3,{19,10})
        self.assertEqual(result.tolist(),[False,False,False,False,False,False,True,True,True])

    def test_photo_ghost_copies_only_visible_masked_original_pixels(self):
        p=np.array([[-.8,-.8,2],[.8,-.8,2],[0,.8,2]],float);f=np.array([[0,1,2]])
        scene=triangle_scene(p,f);rgb=np.full((32,64,3),[60,90,120],dtype=np.uint8)
        mask=np.ones((32,64),bool)
        rgba,_=photo_ghost(scene,scene,np.eye(4),[(np.eye(4),rgb,1,mask)],2.3,width=64,image_height=32)
        self.assertGreater(np.count_nonzero(rgba[:,:,3]),0)
        self.assertEqual(int(rgba[0,0,3]),0)
        self.assertTrue(np.all(rgba[rgba[:,:,3]>0,:3]==[60,90,120]))
        rgba,_=photo_ghost(scene,scene,np.eye(4),[(np.eye(4),rgb,1,np.zeros_like(mask))],2.3,width=64,image_height=32)
        self.assertEqual(np.count_nonzero(rgba[:,:,3]),0)
        blocked=triangle_scene(p*.5,f)
        rgba,_=photo_ghost(scene,blocked,np.eye(4),[(np.eye(4),rgb,1,mask)],2.3,width=64,image_height=32)
        self.assertEqual(np.count_nonzero(rgba[:,:,3]),0)

if __name__=='__main__':unittest.main()
