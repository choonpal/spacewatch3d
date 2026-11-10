import tempfile
import unittest
from pathlib import Path
import numpy as np
from spacewatch_pipeline.viewer import mesh_bytes

class ViewerGeometryTests(unittest.TestCase):
    def test_shared_vertices_keep_separate_instance_colors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mesh.bin';v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
            f=np.array([[0,1,2],[1,3,2]]);rgb=np.full((4,3),123,np.uint8)
            mesh_bytes(v,f,rgb,np.array([7,7]),[dict(id=7,rgb=[5,6,7])],p,np.array([1,2],np.uint32))
            b=p.read_bytes();self.assertEqual(len(b),16+6*30)
            np.testing.assert_equal(np.frombuffer(b,np.float32,count=18,offset=16),v[f].ravel())
            colors=np.frombuffer(b,np.uint8,count=18,offset=16+6*27).reshape(2,3,3)
            np.testing.assert_equal(colors[:,0],colors[:,1]);self.assertFalse(np.array_equal(colors[0,0],colors[1,0]))

    def test_face_semantics_preserve_per_vertex_rgb_and_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mesh.bin'
            v=np.array([[1,2,3],[2,2,3],[1,3,3],[9,9,9]],dtype=np.float32)
            f=np.array([[0,1,2]],dtype=np.int64);rgb=np.array([[11,12,13],[21,22,23],[31,32,33],[0,0,0]],np.uint8)
            mesh_bytes(v,f,rgb,np.array([2]),[dict(id=2,rgb=[80,90,100])],p)
            b=p.read_bytes();self.assertEqual(len(b),106)
            np.testing.assert_equal(np.frombuffer(b,dtype='<u4',count=4),[0x53575031,3,1,0])
            np.testing.assert_equal(np.frombuffer(b,dtype='<f4',count=9,offset=16),v[f].ravel())
            np.testing.assert_equal(np.frombuffer(b,dtype=np.uint8,count=9,offset=88),rgb[f].ravel())
            np.testing.assert_equal(np.frombuffer(b,dtype=np.uint8,count=9,offset=97),[80,90,100]*3)
