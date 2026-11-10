import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from spacewatch_pipeline.change_viewer import publish_surface_comparison
from spacewatch_pipeline.viewer import mesh_bytes
from spacewatch_pipeline.io import save, read


class SurfaceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [1, 3, 2]])
        self.tf = np.eye(4)
        # Nontrivial rotation and translation exercise the same baseline frame.
        self.tf[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.tf[:3, 3] = [10, 20, 1]
        deps = {}
        for epoch, name in enumerate(['before', 'after']):
            points = self.vertices if epoch == 0 else (self.vertices-self.tf[:3, 3])@self.tf[:3, :3]
            arrays = self.root / (name+'.npz')
            np.savez(arrays, vertices=points, faces=self.faces, rgb=np.full((4, 3), 100, np.uint8), face_class_id=[0, 0])
            vocab = self.root / (name+'-classes.json')
            save(vocab, [dict(id=0, rgb=[100, 100, 100])])
            deps[name+'.semantic'] = dict(arrays=str(arrays), vocabulary=str(vocab))
            deps[name+'.reconstruct'] = dict(metric_scale_verified=False)
        scene = self.root/'scene.json'
        save(scene, dict(floor=dict(height_m=0)))
        deps['before.scene'] = dict(scene=str(scene))
        source = self.root/'changes.json'
        deps['pair.changes'] = dict(changes=str(source))
        self.job = dict(config=dict(captures=[dict(id='before'), dict(id='after')]), dependencies=deps, run_root=str(self.root))
        self.tour = dict(alignment=dict(status='underconstrained', baseline_from_current=self.tf.tolist(), registration_error_m=.2), events=[
            dict(id='a', epoch=0, face_ids=[0], bounds=[[0, 0, 0], [1, 1, 0]], status='deferred', reason='alignment_underconstrained', own_views=[dict(frame=1, source_image='/private/path')]),
            dict(id='b', epoch=1, face_ids=[1], bounds=[[0, 0, 0], [1, 1, 0]], status='deferred', reason='alignment_underconstrained')])

    def publish(self):
        save(self.job['dependencies']['pair.changes']['changes'], self.tour)
        assets = []
        publish_surface_comparison(self.tour, self.job, self.root, Path, assets, mesh_bytes)
        return read(self.root/'surface-comparison/comparison.json'), assets

    def test_exact_frame_face_order_and_decisions_preserved(self):
        data, assets = self.publish()
        self.assertEqual(data['counts'], dict(candidates=0, deferred=2))
        self.assertEqual(data['alignment']['status'], 'underconstrained')
        self.assertNotIn('source_image', data['events'][0]['own_views'][0])
        self.assertEqual([e['face_ids'] for e in data['events']], [[0], [1]])
        self.assertEqual(len(assets), 3)
        for epoch in range(2):
            raw = (self.root/f'surface-comparison/mesh-{epoch}.bin').read_bytes()
            xyz = np.frombuffer(raw, '<f4', count=18, offset=16).reshape(2, 3, 3)
            np.testing.assert_allclose(xyz, self.vertices[self.faces], atol=1e-6)

    def test_invalid_face_and_mismatched_transform_rejected(self):
        self.tour['events'][0]['face_ids'] = [2]
        with self.assertRaisesRegex(ValueError, 'face IDs'):
            self.publish()
        self.tour['events'][0]['face_ids'] = [0]
        self.tour['alignment']['baseline_from_current'][0][3] += 1
        with self.assertRaisesRegex(ValueError, 'bounds'):
            self.publish()

    def test_duplicate_faces_and_invalid_epoch_rejected(self):
        self.tour['events'].append(copy.deepcopy(self.tour['events'][0]))
        with self.assertRaisesRegex(ValueError, 'face IDs'):
            self.publish()
        self.tour['events'].pop()
        self.tour['events'][0]['epoch'] = 3
        with self.assertRaisesRegex(ValueError, 'epoch'):
            self.publish()

    def test_unavailable_has_no_identity_substitute_or_fake_mesh(self):
        self.tour = dict(alignment=dict(status='unavailable', baseline_from_current=None), events=[])
        data, assets = self.publish()
        self.assertFalse(data['available'])
        self.assertIsNone(data['bounds'])
        self.assertEqual(data['maps'], [])
        self.assertEqual(assets, ['surface-comparison/comparison.json'])

    def test_empty_comparison_and_candidate_status_remain_distinct(self):
        self.tour['events'] = []
        data, _ = self.publish()
        self.assertTrue(data['available'])
        self.assertEqual(data['counts'], dict(candidates=0, deferred=0))
        self.tour['alignment']['status'] = 'accepted'
        self.tour['events'] = [dict(id='a', epoch=0, face_ids=[0], bounds=[[0, 0, 0], [1, 1, 0]], status='candidate')]
        data, _ = self.publish()
        self.assertEqual(data['counts'], dict(candidates=1, deferred=0))
