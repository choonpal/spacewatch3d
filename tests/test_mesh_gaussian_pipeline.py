import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.gaussian_change.inputs import load_manifest
from tools.gaussian_change.fusion import fuse
from tools.mesh_simsac.core import camera_transform, transform
from tools.mesh_simsac.structural import rigid_icp, residuals, error_summary


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'mesh.npz').write_bytes(b'mesh')
        (self.root/'map.ply').write_bytes(b'ply')
        (self.root/'cameras.json').write_text(json.dumps({'cameras': [{'id': 'k1', 'camera_to_world': np.eye(4).tolist()}]}))
        side = dict(mesh='mesh.npz', gaussian='map.ply', cameras='cameras.json', camera_convention='opencv',
                    **{k: np.eye(4).tolist() for k in ['world_from_mesh', 'world_from_gaussian', 'world_from_camera']})
        self.doc = dict(schema='mesh-gaussian-change-input-v1', world_up='+Y', units='m', before=side.copy(), after=side.copy())

    def save(self):
        p = self.root/'input.json'; p.write_text(json.dumps(self.doc)); return p

    def test_missing_coordinate_bridge_is_rejected(self):
        del self.doc['after']['world_from_gaussian']
        with self.assertRaisesRegex(ValueError, 'world_from_gaussian'):
            load_manifest(self.save())

    def test_saved_render_request_pose_and_scaled_camera_bridge(self):
        p = np.eye(4); p[:3, 3] = [2, 3, 4]
        (self.root/'cameras.json').write_text(json.dumps({'requests': [{'viewpoint_id': 'vp1', 'reference_from_camera': p.tolist()}]}))
        bridge = np.diag([2., 2., 2., 1.]); bridge[0, 3] = 7
        self.doc['before']['world_from_camera'] = bridge.tolist()
        _, sides, _, hashes = load_manifest(self.save())
        camera = sides[0]['views'][0]['pose']
        np.testing.assert_allclose(camera[:3, :3], np.eye(3))
        np.testing.assert_allclose(camera[:3, 3], [11, 6, 8])
        self.assertEqual(len(hashes['before.cameras']), 64)

    def test_duplicate_keypoint_id_and_reflection_are_rejected(self):
        (self.root/'cameras.json').write_text(json.dumps({'cameras': [{'id': 'x', 'camera_to_world': np.eye(4).tolist()}]*2}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_manifest(self.save())
        bridge = np.diag([-1., 1., 1., 1.])
        with self.assertRaises(ValueError):
            camera_transform(np.eye(4), bridge)


class GeometryIntegrationTests(unittest.TestCase):
    def test_json_registration_matrix_is_usable_after_round_trip(self):
        matrix = np.eye(4); matrix[:3, 3] = [2, 3, 4]
        restored = json.loads(json.dumps(matrix.tolist()))
        np.testing.assert_allclose(transform([[0., 0., 0.]], restored), [[2, 3, 4]])

    def test_regional_icp_reduces_known_offset_and_residual_has_full_rank(self):
        rng = np.random.default_rng(8)
        a = rng.uniform(-1, 1, (2100, 3)); n = np.zeros_like(a)
        for axis in range(3):
            mask = np.arange(len(a)) % 3 == axis
            a[mask, axis] = [-1., 1.][axis % 2]
            n[mask, axis] = 1.
        delta = np.array([.04, -.03, .025]); b = a+delta
        matrix, trace = rigid_icp(b, a, n, n, np.eye(4), 2.7, .02)
        self.assertTrue(trace)
        self.assertLess(np.median(np.linalg.norm(transform(b, matrix)-a, axis=1)), .003)
        classes = np.where(abs(n[:, 1]) > .8, 3, 0)
        field = error_summary(residuals(a, transform(b, matrix), n, n @ matrix[:3, :3].T, classes, classes), 2.7)
        self.assertTrue(field['supported'])
        self.assertEqual(field['normal_rank'], 3)

    def test_same_change_from_two_keypoints_is_one_3d_candidate(self):
        q = np.array([[x, y, z] for x in [.01, .11, .21] for y in [.01, .11] for z in [.01, .11]])
        samples = [{'points': q, 'phase': 0, 'view': 0}, {'points': q+.002, 'phase': 1, 'view': 1}]
        candidates, points, labels = fuse(samples, .1, 5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['observations'], [0, 1])
        self.assertEqual(candidates[0]['type'], 'unclassified')
        self.assertEqual(len(points), len(labels))
        self.assertEqual(fuse([], .1, 5)[0], [])


class SystemAdapterTests(unittest.TestCase):
    def test_dispatcher_preserves_lineage_and_links_candidates_without_inventing_object_events(self):
        from spacewatch3d.task_api import TaskRequest
        from spacewatch3d.tasks.task_5a_change_detection.pipeline import run
        from spacewatch3d.contracts import validate_artifact
        root = Path(__file__).resolve().parents[1]
        inputs = {p: root/'examples/fixtures'/p/'objects.json' for p in ['before', 'after']}
        lineage = {p: json.loads(path.read_text()) for p, path in inputs.items()}
        with tempfile.TemporaryDirectory(dir=root, prefix='.test-task5a-') as temp:
            folder = Path(temp); manifest = folder/'pair.json'
            manifest.write_text(json.dumps({'scene_id': lineage['after']['scene_id'],
                                           **{p: {'capture_id': lineage[p]['capture_id']} for p in inputs}}))
            def fake_gpu(pair, output, runtime):
                output.mkdir(parents=True)
                (output/'changes.json').write_text(json.dumps({'candidates': [{'id': 'C1'}]}))
                (output/'registration.json').write_text(json.dumps({'global_after_to_before': np.eye(4).tolist()}))
                return output/'changes.json'
            with patch('spacewatch3d.tasks.task_5a_change_detection.gaussian_backend.analyze', side_effect=fake_gpu):
                result = run(TaskRequest(inputs, folder/'result', {'backend': 'mesh_gaussian_simsac', 'pair_manifest': str(manifest)}))
            data = validate_artifact(result, check_files=True)
            self.assertTrue(data['synthetic'])
            self.assertEqual(data['changes'], [])
            self.assertEqual(data['parameters']['candidate_count'], 1)


if __name__ == '__main__':
    unittest.main()
