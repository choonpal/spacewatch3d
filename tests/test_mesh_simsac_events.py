"""Event typing must separate changes, displacement, occlusion and repeated furniture."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/mesh_simsac"))
from tools.mesh_simsac.classify import classify, decide_unmatched, movement_rejections, opposite_observations, rigid_fit, unique_assignments
from tools.mesh_simsac.core import write, sha


def vacancy(frame, free=.9, occluded=.05, occupied=.05, usable=True):
    return dict(frame_id=str(frame), free_fraction=free, occluded_fraction=occluded,
                occupied_fraction=occupied, usable=usable)


class MeshSimSaCEvents(unittest.TestCase):
    def test_unrelated_bad_region_cannot_turn_displaced_doors_into_modified(self):
        # Minimized replay of EV-003: local residual .178, unrelated region
        # residual 1.124, doors separated by 1.6 map units.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'results').mkdir()
            v = np.array([[0.,0,0], [0,2,0], [0,0,1], [0,2,1]])
            faces = np.array([[0,1,2], [1,3,2]])
            meshes = [(v, dict(faces=faces, face_instance_id=np.ones(2, int), face_region=np.full(2, 3))),
                      (v+[1.6,0,0], dict(faces=faces, face_instance_id=np.ones(2, int), face_region=np.full(2, 3)))]
            registration = dict(state='passed_geometric_gate', new_alignment_fitted=True,
                overlap_fractions=[.99,.99], pointcloud_residual_p90_u=[.091,.087],
                fresh_structural={'regions': [
                    {'supported': False, 'validation': {'tolerance_u': 1.124}},
                    {'supported': True, 'validation': {'tolerance_u': .185}},
                    {'supported': True, 'validation': {'tolerance_u': .178}}]})
            write(root/'manifest.json', dict(views=[], settings={'voxel_size_u': .04, 'visibility_tolerance_u': .12},
                  registration=registration, scope='regression'))
            source = root/'input.json'
            write(source, dict(before={'mesh':'before.npz'}, after={'mesh':'after.npz'}))
            records = [dict(epoch=e, instance_id=1, label='door', regions=[3], state='change_candidate',
                           bounds=[points.min(0).tolist(), points.max(0).tolist()],
                           changed_frames=['1','2'], observed_frames=['1','2'])
                       for e,(points,_) in enumerate(meshes)]
            write(root/'results/changes.json', {'objects':records})
            np.savez(root/'results/object-appearance.npz', **{'E0-I1':np.eye(2), 'E1-I1':np.eye(2)})
            write(root/'results/object-appearance.json', dict(source_manifest_sha256=sha(root/'manifest.json'),
                  descriptors_sha256=sha(root/'results/object-appearance.npz'), objects={}))
            with patch('tools.mesh_simsac.classify.aligned_mesh', side_effect=meshes):
                result = classify(root, source)
            self.assertEqual(result['counts']['modified'], 0,
                             'Unrelated regional error falsely makes distinct doors stationary')
            self.assertEqual(result['stationary_pairs'], [])

    def test_added_and_removed_require_two_free_space_observations(self):
        records = [vacancy(1), vacancy(2)]
        self.assertEqual(decide_unmatched(1, True, 2, records, False)[0], "added")
        self.assertEqual(decide_unmatched(0, True, 2, records, False)[0], "removed")
        self.assertEqual(decide_unmatched(0, True, 2, records[:1], False)[0], "unresolved")
        self.assertEqual(decide_unmatched(0, True, 2, [records[0], records[0]], False)[0], "unresolved")

    def test_occlusion_or_occupied_surface_never_means_disappearance(self):
        for records in [[vacancy(1, .0, .9), vacancy(2, .0, .9)],
                        [vacancy(1, .1, 0, .9), vacancy(2, .1, 0, .9)],
                        [vacancy(1, usable=False), vacancy(2, usable=False)]]:
            self.assertEqual(decide_unmatched(0, True, 2, records, False)[0], "unresolved")

    def test_ambiguous_counterpart_blocks_added_removed(self):
        self.assertEqual(decide_unmatched(1, True, 4, [vacancy(1), vacancy(2)], True)[0], "unresolved")

    def test_rgb_segmentation_presence_vetoes_mesh_only_addition(self):
        records=[{**vacancy(i), 'same_class_fraction': .8} for i in [1,2]]
        kind, reason=decide_unmatched(1, True, 4, records, False)
        self.assertEqual(kind, 'unresolved')
        self.assertEqual(reason, 'counterpart_in_rgb_segmentation_but_mesh_incomplete')

    def test_background_mesh_is_not_evidence_that_a_segmented_object_existed(self):
        records=[{**vacancy(i, free=0, occupied=.95, occluded=0),
                  'segmentation_available':True, 'same_class_fraction':0,
                  'unknown_fraction':.05} for i in [1,2]]
        kind, reason=decide_unmatched(1, True, 3, records, False, own_segmented_frames=3)
        self.assertEqual(kind, 'added')
        self.assertEqual(reason, 'segmented_object_appears_over_observed_background')
        self.assertEqual(decide_unmatched(1, True, 3, records, False)[0], 'unresolved')
        self.assertEqual(decide_unmatched(1, True, 3, [{**r,'unknown_fraction':.6} for r in records], False,
                                         own_segmented_frames=3)[0], 'unresolved')

    def test_no_repeatable_simsac_change_does_not_create_an_event(self):
        self.assertEqual(decide_unmatched(1, False, 3, [vacancy(1), vacancy(2)], False)[0], "unchanged")

    def test_unlabelled_category_cannot_supply_semantic_absence(self):
        pixels = np.indices((4, 4)).reshape(2, -1).T
        view = dict(epoch=0, id='before', frame_id='1', depth=np.ones((4, 4)),
                    valid=np.ones((4, 4), bool), person=np.zeros((4, 4), bool),
                    segmentation_class=np.zeros((4, 4), int), segmentation_labels=['chair'])
        projection = (pixels, np.ones(16), np.ones(16, bool))
        with patch('tools.mesh_simsac.classify.project_in_view', return_value=projection):
            records = opposite_observations(np.zeros((16, 3)), 1, [view], .1, label='fire extinguisher')
        self.assertFalse(records[0]['segmentation_available'])

    def test_movement_requires_identity_error_margin_and_both_vacancies(self):
        pair = dict(shape_f1=.95, directional_coverage=[.9, .9], appearance=.95, frame_support=3,
                    margin=.15, displacement_u=1., change_supported=True)
        records = [vacancy(1), vacancy(2)]
        self.assertEqual(movement_rejections(pair, records, records, .1, .1), [])
        self.assertIn("repeated_object_identity_ambiguous", movement_rejections({**pair, "margin": .01}, records, records, .1, .1))
        self.assertIn("displacement_within_map_error", movement_rejections({**pair, "displacement_u": .2}, records, records, .1, .1))
        self.assertIn("origin_vacancy_unobserved", movement_rejections(pair, records[:1], records, .1, .1))

    def test_assignment_does_not_reuse_one_chair_for_two_moves(self):
        pairs = [dict(before="a", after="x", score=.9), dict(before="b", after="x", score=.8),
                 dict(before="b", after="y", score=.7)]
        accepted = unique_assignments(pairs)
        self.assertEqual({(p["before"], p["after"]) for p in accepted}, {("a", "x"), ("b", "y")})

    def test_rigid_fit_keeps_real_displacement_and_never_fits_scale(self):
        rng = np.random.default_rng(12)
        a = rng.normal(size=(150, 3)) * [1, .4, .3]
        b = a + [1.3, .05, -.4]
        fit = rigid_fit(a, b, .06)
        m = np.array(fit["before_to_after_rigid"])
        self.assertGreater(fit["shape_f1"], .98)
        np.testing.assert_allclose(m[:3, 3], [1.3, .05, -.4], atol=.02)
        np.testing.assert_allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
