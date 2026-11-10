import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_jobs import TrajectoryJobs, digest_file
try:
    import numpy as np
    from estimate_trajectory import fit_floor
except ImportError:
    fit_floor = None


class JobTests(unittest.TestCase):
    def test_upload_hash_and_path_boundary(self):
        with TemporaryDirectory() as directory, patch('trajectory_jobs.subprocess.run') as run, patch.object(TrajectoryJobs, 'inspect_video', return_value={'duration': 12, 'width': 3840, 'height': 1920}):
            run.return_value.returncode = 1
            jobs = TrajectoryJobs(directory)
            key = jobs.register_upload(io.BytesIO(b'video bytes'), 11, '../../video.mp4')
            source = jobs.sources[key]
            self.assertEqual(source['name'], 'video.mp4')
            self.assertTrue(source['path'].is_relative_to(Path(directory)))
            self.assertEqual(key, digest_file(source['path']))
            self.assertEqual(jobs.status(key)['state'], 'idle')
            with self.assertRaises(ValueError):
                jobs.start(key)
            with self.assertRaises(ValueError):
                jobs.register_upload(io.BytesIO(b'x'), jobs.MAX_UPLOAD + 1, 'large.mp4')
            with self.assertRaises(ValueError):
                jobs.register_upload(io.BytesIO(b'x'), 1, 'program.py')
            with self.assertRaises(ValueError):
                jobs.register_upload(io.BytesIO(b'x'), 100, 'broken.mp4')
            self.assertFalse(list(jobs.data.glob('*.part')))
            restored = TrajectoryJobs(directory)
            self.assertIn(key, restored.sources)

    def test_unknown_source_and_interrupted_job(self):
        with TemporaryDirectory() as directory, patch('trajectory_jobs.subprocess.run') as run, patch.object(TrajectoryJobs, 'inspect_video', return_value={'duration': 12, 'width': 3840, 'height': 1920}):
            run.return_value.returncode = 1
            jobs = TrajectoryJobs(directory)
            with self.assertRaises(KeyError):
                jobs.status('a' * 64)
            self.assertIsNone(jobs.result_path('a' * 64))
            key = jobs.register_upload(io.BytesIO(b'x'), 1, 'a.mp4')
            (jobs.cache / key / 'status.json').write_text(json.dumps({'state': 'running'}))
            self.assertEqual(jobs.status(key)['state'], 'interrupted')


@unittest.skipIf(fit_floor is None, 'Trajectory analysis dependencies not installed')
class FloorTests(unittest.TestCase):
    def test_observed_floor_estimation_rejects_vertical_wall_outliers(self):
        rng = np.random.default_rng(4)
        floor = np.column_stack([rng.uniform(-4,4,600), rng.normal(1.7,.01,600), rng.uniform(-4,4,600)])
        walls = np.column_stack([rng.normal(4,.01,400), rng.uniform(-1,1.7,400), rng.uniform(-4,4,400)])
        cameras = np.column_stack([np.linspace(-2,2,20),np.zeros(20),np.zeros(20)])
        result, up = fit_floor(np.vstack([floor,walls]),cameras,np.repeat(np.eye(3)[None],20,axis=0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['cameraHeight'],1.7,delta=.03)
        self.assertGreater(result['support'],500)
        self.assertGreater(np.dot(up,[0,-1,0]),.99)

    def test_no_floor_is_reported_when_only_vertical_structure_is_available(self):
        rng = np.random.default_rng(8)
        points = np.column_stack([np.ones(300)*3,rng.uniform(-2,2,300),rng.uniform(-5,5,300)])
        result,_ = fit_floor(points,np.zeros((12,3)),np.repeat(np.eye(3)[None],12,axis=0))
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
