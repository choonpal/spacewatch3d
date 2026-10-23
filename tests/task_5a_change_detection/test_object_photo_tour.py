import unittest
import numpy as np
from spacewatch3d.tasks.task_5a_change_detection.object_photo_tour import object_faces

class ObjectPhotoTourTests(unittest.TestCase):
    def test_foreground_boxes_cannot_reintroduce_known_wall_floor_or_unknown_low_floor(self):
        classes=np.array([0,3,5,8,14,255,255,19,10])
        centers=np.ones((9,3));centers[5,1]=0
        result=object_faces(classes,np.ones(9,bool),centers,2.3,{19,10})
        self.assertEqual(result.tolist(),[False,False,False,False,False,False,True,True,True])


if __name__=='__main__':unittest.main()
