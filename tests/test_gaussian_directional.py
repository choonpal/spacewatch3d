import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from tools.gaussian_change.directional import tile_boxes,predict_face,stitch,render_pair


class DirectionalTests(unittest.TestCase):
    def test_native_2k_face_uses_four_overlapping_views(self):
        boxes=tile_boxes((2048,2048,3),1040,32)
        self.assertEqual(len(boxes),4)
        cover=np.zeros((2048,2048),'u1')
        for x0,y0,x1,y1 in boxes:cover[y0:y1,x0:x1]+=1
        self.assertTrue(np.all(cover>0));self.assertGreater(int(cover.max()),1)

    def test_inference_receives_local_perspective_tiles_and_preserves_query_direction(self):
        class Model:
            def __init__(self):self.calls=[]
            def predict(self,q,r):self.calls.append((q.shape,r.shape));return np.full(q.shape[:2],q[0,0,0]/255,dtype='f4')
        model=Model();a=np.full((96,96,3),51,'u1');b=np.full_like(a,204)
        prob,audit,control=predict_face(model,a,b,64,32,True)
        self.assertEqual(len(audit),4);self.assertEqual(len(model.calls),9)
        self.assertTrue(all(x[0]==(64,64,3) for x in model.calls))
        np.testing.assert_allclose(prob[0],.2);np.testing.assert_allclose(prob[1],.8)
        self.assertEqual(control['input_kind'],'native perspective tile')

    @unittest.skipUnless(find_spec('cv2'),'OpenCV rendering dependency is not installed')
    def test_cube_depth_is_converted_to_radial_depth_when_probabilities_are_projected(self):
        faces=[]
        for i in range(6):faces.append(dict(rgb=np.full((64,64,3),i,'u1'),depth=np.ones((64,64),'f4'),alpha=np.ones((64,64),'f4'),person=np.zeros((64,64),bool),probability=np.full((64,64),i/10,'f4')))
        p=stitch(faces,128,64,64)
        self.assertEqual(p['rgb'].shape,(64,128,3));self.assertTrue(np.all(p['depth']>=1))
        self.assertAlmostEqual(float(p['probability'][32,64]),0.)
        self.assertAlmostEqual(float(p['probability'][32,96]),.1,places=5)
        self.assertTrue(np.isfinite(p['depth']).all())

    @unittest.skipUnless(find_spec('cv2'),'OpenCV rendering dependency is not installed')
    def test_render_pair_never_sends_the_stitched_panorama_to_the_model(self):
        class Renderer:
            def face(self,pose,size):return np.full((size,size,3),80,'u1'),np.ones((size,size),'f4'),np.ones((size,size),'f4')
        class Detector:
            def predict(self,*args,**kwargs):return [SimpleNamespace(masks=None,boxes=None)]
        class Model:
            def __init__(self):self.shapes=[]
            def predict(self,a,b):self.shapes.append(a.shape);return np.full(a.shape[:2],.3,'f4')
        model=Model()
        with TemporaryDirectory() as temp:
            pair,prob,audit,control=render_pair([Renderer(),Renderer()],np.eye(4),Detector(),model,128,64,64,Path(temp),32,0,True)
        self.assertEqual(audit['tiles_per_phase'],24);self.assertEqual(audit['full_panorama_model_calls'],0)
        self.assertEqual(len(model.shapes),49);self.assertTrue(all(s==(32,32,3) for s in model.shapes))
        self.assertEqual(pair[0]['rgb'].shape,(64,128,3));np.testing.assert_allclose(prob[0],.3,atol=1e-7)


if __name__=='__main__':unittest.main()
