"""Piecewise alignment must preserve original camera rays in both directions."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/mesh_simsac'))
from tools.mesh_simsac.core import counterpart_support, project_in_view, transform
from tools.mesh_simsac.regional import RegionalAlignment


class RegionalProjection(unittest.TestCase):
    def test_two_regions_in_one_photo_use_their_own_camera_transforms(self):
        # Both pixels observe z=2. Region 2 was shifted +10 in map coordinates.
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[2, 0, 3] = 10.
        reference = dict(pose=np.eye(4), K=np.eye(3), depth=np.full((1, 2), 2.),
                         valid=np.ones((1, 2), bool), person=np.zeros((1, 2), bool),
                         region_poses=poses, point_region=np.array([[1, 2]]))
        points = np.array([[0., 0., 2.], [12., 0., 2.]])
        supported, hidden = counterpart_support(points, reference, .1, regions=np.array([1, 2]))
        self.assertTrue(supported.all())
        self.assertFalse(hidden.any())

    def test_different_region_at_projected_pixel_is_not_vacancy_evidence(self):
        reference = dict(pose=np.eye(4), K=np.eye(3), depth=np.full((1, 1), 4.),
                         valid=np.ones((1, 1), bool), region_poses=np.repeat(np.eye(4)[None], 3, axis=0),
                         point_region=np.array([[2]]))
        supported, hidden = counterpart_support(np.array([[0., 0., 2.]]), reference, .1, regions=np.array([1]))
        self.assertFalse(supported.any())
        self.assertFalse(hidden.any())

    def test_boundary_triangles_preserve_shape_and_roundtrip_to_original_rays(self):
        correction = np.eye(4)
        correction[:3, 3] = [10, 2, -1]
        context = dict(base_matrix=np.eye(4).tolist(), membership_base_matrix=np.eye(4).tolist(),
                       membership_parent_transform=dict(axis=[1,0,0], transition_start_u=0, transition_width_u=1,
                                                        correction_matrix=np.eye(4).tolist()),
                       methods=dict(structure=dict(region_matrices_after_Geo=[np.eye(4).tolist(), correction.tolist()])))
        zones = dict(labels=np.array([[1], [2]]), origin=np.array([-1., 1.]), step=np.array(1.), axes=np.eye(2))
        alignment = RegionalAlignment(context, zones)
        vertices = np.array([[-1.,0.,2.], [0.,0.,2.], [0.,1.,2.], [1.,0.,2.]])
        faces = np.array([[0,1,2], [1,3,2]])
        mesh = alignment.mesh(vertices, faces, np.array([4,5]), np.eye(4), 1)
        np.testing.assert_array_equal(mesh['face_region'], [1,2])
        self.assertEqual(len(mesh['vertices']), 6)
        before, after = vertices[faces], mesh['vertices'][mesh['faces']]
        np.testing.assert_allclose(np.linalg.norm(before[:,1]-before[:,0], axis=1),
                                   np.linalg.norm(after[:,1]-after[:,0], axis=1))
        points = np.array([[[0.,0.,2.], [2.,0.,2.]]])
        view = dict(points=points.copy(), face=np.array([[0,1]]), mesh_hit=np.ones((1,2),bool),
                    pose=np.eye(4), K=np.eye(3), depth=np.full((1,2),2.))
        aligned = alignment.view(view, mesh['face_region'], 1)
        ij, depth, inside = project_in_view(aligned['points'], aligned, aligned['point_region'])
        np.testing.assert_array_equal(ij, [[[0,0], [1,0]]])
        np.testing.assert_allclose(depth, 2.)
        self.assertTrue(inside.all())


if __name__ == '__main__':
    unittest.main()
