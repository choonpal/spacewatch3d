import tempfile
from pathlib import Path
import unittest

import numpy as np
from tools.mesh_simsac.import_semantic_scene import associate, region_faces, sample_ply


class SemanticImportTests(unittest.TestCase):
    def test_original_triangle_identity_and_wrong_mesh_rejection(self):
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], float)
        faces = np.array([[0,1,2],[0,2,3]])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'object.obj'
            p.write_text('v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n')
            np.testing.assert_array_equal(region_faces(vertices, faces, p), [1])
            # OBJ exports may merge coincident vertices. Exact triangle geometry
            # must still recover the original face rather than a nearest vertex ID.
            duplicated = np.vstack([vertices,vertices[0]])
            with_duplicate_index = np.array([[0,1,2],[4,2,3]])
            np.testing.assert_array_equal(region_faces(duplicated,with_duplicate_index,p),[1])
            repeated_faces=np.array([[0,2,3],[0,2,3]])
            np.testing.assert_array_equal(region_faces(vertices,repeated_faces,p,claimed=[True,False]),[1])
            with self.assertRaisesRegex(ValueError,'overlap'):
                region_faces(vertices,repeated_faces,p,claimed=[True,True])
            with self.assertRaisesRegex(ValueError, 'source mesh'):
                region_faces(vertices+10, faces, p)
            with self.assertRaisesRegex(ValueError, 'outside'):
                region_faces(vertices, faces[:1], p)

    def test_class_only_masks_cannot_invent_instances_or_override_person(self):
        classes = np.array([[4,4,7,4,4]])
        hit = np.array([[0,1,0,0,2**32-1]], 'u4')
        person = np.array([[False,False,False,True,False]])
        np.testing.assert_array_equal(associate(classes,hit,np.array([12,0]),np.array([3,3]),person),
                                      [[12,0,0,0,0]])

    def test_pointcloud_export_is_sampled_from_real_records(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'cloud.ply'
            header=('ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\n'
                    'property float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            records=np.zeros(5,dtype=[('xyz','<f4',3),('rgb','u1',3)])
            records['xyz'][:,0]=np.arange(5)
            p.write_bytes(header.encode()+records.tobytes())
            points, count=sample_ply(p,3)
            self.assertEqual(count,5)
            np.testing.assert_array_equal(points[:,0],[0,2,4])


if __name__ == '__main__':
    unittest.main()
