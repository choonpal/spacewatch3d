import copy
import math
from pathlib import Path
import re
import unittest

import numpy as np

from spacewatch_pipeline.paper_viewer import bbox_angles, build_changes, coordinate_metadata, change_summary


def fixture():
    manifest = {"captures": [
        {"id": "before", "views": [{"id": "before-view", "source_frame": {"source_frame_index": 20}}]},
        {"id": "after", "views": [{"id": "fresh-a", "source_frame": {"source_frame_index": 20}}, {"id": "fresh-b", "source_frame": {"source_frame_index": 21}}]},
    ]}
    photo = {"image": "events/photos/actual.jpg", "capture_id": "after", "frame_id": "frame-a", "source_frame_index": 20,
             "view": "source-cube", "width": 768, "height": 768, "bbox": [330, 330, 438, 438], "role": "observed_object"}
    detector = {"events": [{"id": "event-a", "type": "unresolved", "label": "box", "verified_change": "change-a", "photo_sets": [{"1": photo}]}],
                "verified_changes": [{"id": "change-a", "label": "box", "event_ids": ["event-a"]}]}
    inputs = {"after": {"id": "after", "frames": [{"id": "frame-a", "source_frame_index": 20, "sha256": "input-hash", "keypoint_ids": ["anchor-a", "anchor-b"]}]}}
    plan = {"keypoints": [{"id": "anchor-a", "anchor_capture": "after", "viewpoint_id": "fresh-a"}, {"id": "anchor-b", "anchor_capture": "after", "viewpoint_id": "fresh-b"}]}
    metadata = {"views": [{"id": "source-cube", "epoch": 1, "frame_id": "frame-a", "K": [[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]], "face_rotation": np.eye(3).tolist(), "inference_size": [224, 224]}]}
    return manifest, detector, inputs, plan, metadata


class CurrentNativeViewerContract(unittest.TestCase):
    def test_grouping_does_not_clear_unresolved_status(self):
        summary = change_summary(fixture()[1])
        self.assertEqual(summary['candidates'], 1)
        self.assertEqual(summary['verified_count'], 1)
        self.assertEqual(summary['deferred'], 1)
        self.assertEqual(summary['ungrouped'], 0)
        self.assertIn('not_independent_ground_truth', summary['verification_kind'])

    def build(self, *values):
        return build_changes(*values, lambda photo: "/pipeline-assets/actual.jpg")

    def test_dynamic_current_views_and_after_only(self):
        values = fixture()
        result = self.build(*values)
        self.assertEqual([row["viewpoint_id"] for row in result["captures"]["after"]["views"]], ["fresh-a", "fresh-b"])
        self.assertTrue(all(not row["items"] and not row["evidence_ids"] for row in result["captures"]["before"]["views"]))
        self.assertEqual(result["counts"]["verified_changes"], 1)

    def test_only_same_physical_frame_gets_panorama_box(self):
        result = self.build(*fixture())
        first, nearby = result["captures"]["after"]["views"]
        self.assertEqual(len(first["items"]), 1)
        self.assertEqual(nearby["items"], [])
        self.assertTrue(nearby["evidence_ids"])
        self.assertEqual(nearby["verified_change_ids"], ["change-a"])
        self.assertEqual(first["items"][0]["frame_binding"]["source_frame_index"], 20)

    def test_source_decision_is_unchanged_and_group_is_separate(self):
        values = fixture()
        before = copy.deepcopy(values[1])
        result = self.build(*values)
        self.assertEqual(values[1], before)
        item = result["captures"]["after"]["views"][0]["items"][0]
        self.assertEqual(item["type"], "unresolved")
        self.assertEqual(item["verified_change"], "change-a")

    def test_missing_recorded_projection_keeps_actual_photo_evidence(self):
        values = fixture()
        values[-1]["views"][0].pop("face_rotation")
        result = self.build(*values)
        self.assertEqual(result["counts"]["same_frame_erp_boxes"], 0)
        self.assertTrue(result["evidence"])
        self.assertEqual(result["evidence"][0]["bbox"], [330, 330, 438, 438])

    def test_stale_plan_is_rejected(self):
        values = fixture()
        values[3]["keypoints"][0]["viewpoint_id"] = "old-fifteen-view"
        with self.assertRaisesRegex(ValueError, "freshly selected"):
            self.build(*values)

    def test_stale_anchor_and_mismatched_source_frame_rejected(self):
        values = fixture()
        values[2]["after"]["frames"][0]["keypoint_ids"] = ["old-anchor"]
        with self.assertRaisesRegex(ValueError, "stale or unknown"):
            self.build(*values)
        values = fixture()
        values[1]["events"][0]["photo_sets"][0]["1"]["source_frame_index"] = 999
        with self.assertRaisesRegex(ValueError, "source-frame mismatch"):
            self.build(*values)

    def test_projection_uses_recorded_rays_not_raw_cube_pixels(self):
        values = fixture()
        angles = bbox_angles(values[1]["events"][0]["photo_sets"][0]["1"], values[-1]["views"][0])
        self.assertEqual(len(angles), 80)
        self.assertTrue(np.max(np.abs(angles)) < 15)
        self.assertLess(abs(np.mean(np.asarray(angles)[:, 0])), .1)

    def test_coordinate_scale_metadata_comes_from_current_dependencies(self):
        job = {"config": {"paper_integration": {"coordinate_units": "u"}, "captures": [{"id": "before"}, {"id": "after"}]},
               "dependencies": {"before.reconstruct": {"metric_scale_verified": True}, "after.reconstruct": {"metric_scale_verified": False}}}
        metadata = coordinate_metadata(job)
        self.assertEqual(metadata["coordinate_units"], "u")
        self.assertFalse(metadata["metric_scale_verified"])
        self.assertEqual(metadata["captures"], {"before": True, "after": False})
        job["dependencies"]["after.reconstruct"]["metric_scale_verified"] = True
        self.assertTrue(coordinate_metadata(job)["metric_scale_verified"])

    def test_ui_ids_and_above_floor_orientation(self):
        root = Path(__file__).resolve().parents[1]
        app = root / "apps/autotour360"
        javascript = (app / "survey.js").read_text()
        markup = (app / "pipeline.html").read_text()
        ids = set(re.findall(r'id="([^"]+)"', markup))
        required = set(re.findall(r"\$\('([^']+)'\)", javascript))
        self.assertEqual(required - ids, set())
        self.assertIn("this.pitch=-.95", (app / "pipeline-mesh.js").read_text())
        self.assertGreater(-math.sin(-.95), 0)
        self.assertIn("runtime_details", javascript)
        self.assertNotIn("SAM2+ProPainter", markup + javascript)
        self.assertIn("u(재구성 단위)", markup)
        self.assertIn("실제 미터 단위로 검증하지 않았습니다", markup)
        self.assertIn("verified_change_ids", javascript)
        publisher = (root / "spacewatch_pipeline/paper_viewer.py").read_text()
        self.assertNotIn("from .viewer_service import launch", publisher)
        self.assertIn('"viewer_launch": False', publisher)
        self.assertIn('coordinate_metadata(job)', publisher)


if __name__ == "__main__":
    unittest.main()
