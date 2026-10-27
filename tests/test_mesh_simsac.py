"""Regression checks for camera support, unknown observations and multi-frame votes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/mesh_simsac"))
import core as mesh_core
from tools.mesh_simsac.run import Evidence


def view(vid="q", epoch=0, frame="1", z=2., valid=True):
    K = np.array([[1., 0, 1], [0, 1, 1], [0, 0, 1]])
    pose = np.eye(4)
    depth = np.full((3, 3), z)
    points = mesh_core.rays(K, (3, 3)) * z
    return dict(id=vid, epoch=epoch, frame_id=frame, pose=pose, K=K, points=points,
                depth=depth, valid=np.full((3, 3), valid), person=np.zeros((3, 3), bool),
                instance=np.full((3, 3), 7), rgb=np.zeros((3, 3, 3), "u1"))


class MeshSimSaC(unittest.TestCase):
    def test_projected_mesh_without_its_source_segmentation_is_not_object_evidence(self):
        query, reference = view(), view('r', 1)
        query['segmentation_supported'] = np.zeros((3,3), bool)
        masks = mesh_core.evidence_masks(query, reference, np.ones((3,3)), query['person'], reference['person'], .1)
        self.assertEqual(int(masks['candidate'].sum()), 0)
        self.assertEqual(int(masks['observable'].sum()), 0)

    def test_occluded_object_is_unknown_not_a_removal(self):
        query, reference = view(), view("r", 1, z=1.)
        masks = mesh_core.evidence_masks(query, reference, np.ones((3, 3)), query["person"], reference["person"], .1)
        self.assertEqual(int(masks["candidate"].sum()), 0)
        self.assertEqual(int(masks["occluded"].sum()), 9)
        self.assertEqual(int(masks["observable"].sum()), 0)

    def test_visible_background_supports_a_missing_foreground(self):
        query, reference = view(), view("r", 1, z=3.)
        masks = mesh_core.evidence_masks(query, reference, np.ones((3, 3)), query["person"], reference["person"], .1)
        self.assertEqual(int(masks["candidate"].sum()), 9)

    def test_missing_reference_geometry_never_proves_absence(self):
        query, reference = view(), view("r", 1, valid=False)
        masks = mesh_core.evidence_masks(query, reference, np.ones((3, 3)), query["person"], reference["person"], .1)
        self.assertEqual(int(masks["candidate"].sum()), 0)
        self.assertEqual(int(masks["unobserved_counterpart"].sum()), 9)

    def test_reference_person_blocks_observation_at_its_projected_position(self):
        query, reference = view(), view("r", 1)
        reference["person"][1, 1] = True
        masks = mesh_core.evidence_masks(query, reference, np.ones((3, 3)), query["person"], reference["person"], .1)
        self.assertFalse(masks["candidate"][1, 1])
        self.assertFalse(masks["observable"][1, 1])
        self.assertEqual(int(masks["candidate"].sum()), 8)

    def test_translated_camera_samples_correct_reference_pixel(self):
        reference = view("r", 1, z=3.)
        reference["pose"][0, 3] = 2.
        ij, z, inside = mesh_core.project(np.array([[2., 0., 2.], [0., 0., 2.]]), reference["pose"], reference["K"], (3, 3))
        np.testing.assert_array_equal(ij, [[1, 1], [0, 1]])
        self.assertTrue(inside.all())
        np.testing.assert_allclose(z, [2, 2])

    def test_pair_selection_deduplicates_faces_from_one_original_frame(self):
        q = view()
        refs = [view("a", 1, frame="10"), view("b", 1, frame="10"), view("c", 1, frame="11")]
        pairs = mesh_core.choose_pairs([q, *refs], .1, max_refs=2)
        self.assertEqual([r["id"] for r in pairs[0]["refs"]], ["a", "c"])
        self.assertEqual(mesh_core.choose_pairs([q], .1)[0]["state"], "unobserved_counterpart")

    def test_multiple_cube_faces_are_one_vote(self):
        self.assertEqual(mesh_core.vote_state(["f1", "f1"], ["f1", "f1"]), "insufficient_observations")
        self.assertEqual(mesh_core.vote_state(["f1", "f2"], ["f1", "f2"]), "change_candidate")
        self.assertEqual(mesh_core.vote_state([], []), "insufficient_observations")

    def test_unlabelled_surfaces_survive_object_aggregation(self):
        settings = dict(voxel_size_u=10., min_frames=2, changed_frame_ratio=.5,
                        min_object_pixels=1, object_pixel_ratio=.1)
        evidence = Evidence(settings)
        for frame in ["1", "2"]:
            v = view(frame=frame)
            v["instance"][:] = 0
            evidence.add(v, np.ones((3, 3), bool), np.ones((3, 3), bool))
        objects, regions, _ = evidence.finish([dict(epoch=0, instance_id=7)])
        self.assertEqual(objects[0]["state"], "insufficient_observations")
        self.assertTrue(regions)
        self.assertEqual(regions[0]["instance_ids"], [])
        self.assertEqual(regions[0]["source_frames"], ["1", "2"])

    def test_unchanged_support_counts_in_vote_denominator(self):
        settings = dict(voxel_size_u=10., min_frames=2, changed_frame_ratio=.6,
                        min_object_pixels=1, object_pixel_ratio=.1)
        evidence = Evidence(settings)
        for number in range(4):
            v = view(frame=str(number))
            evidence.add(v, np.ones((3, 3), bool), np.full((3, 3), number < 2))
        objects, regions, _ = evidence.finish([dict(epoch=0, instance_id=7)])
        self.assertEqual(objects[0]["state"], "no_supported_change")
        self.assertEqual(regions, [])

    def test_scaled_alignment_keeps_camera_rotation_rigid(self):
        matrix = np.diag([2., 2., 2., 1.])
        matrix[:3, 3] = [1, 2, 3]
        pose = np.eye(4)
        pose[:3, 3] = [1, 0, 1]
        result = mesh_core.camera_transform(pose, matrix)
        np.testing.assert_allclose(result[:3, :3], np.eye(3))
        np.testing.assert_allclose(result[:3, 3], [3, 2, 5])
        with self.assertRaises(ValueError):
            mesh_core.similarity(np.diag([1., 2., 3., 1.]))

    def test_erp_seam_interpolates_across_wrap(self):
        image = np.zeros((2, 4, 3), "u1")
        image[:, 0] = 100
        image[:, -1] = 200
        projected = mesh_core.erp_project(image, np.array([[[0., 0., -1.]]]), np.eye(3))
        np.testing.assert_array_equal(projected, [[[150, 150, 150]]])


if __name__ == "__main__":
    unittest.main()
