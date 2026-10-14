import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migrate_legacy import migrate
from storage import StoragePaths
from trajectory_jobs import TrajectoryJobs, write_json


class StorageTests(unittest.TestCase):
    def test_repo_defaults_and_relative_overrides(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "apps" / "autotour360"
            storage = StoragePaths.for_app(app)
            self.assertEqual(storage.data, root / "data" / "virtual_tour")
            self.assertEqual(storage.analysis, root / "outputs" / "virtual_tour" / "analysis")
            custom = StoragePaths.for_app(app, "../../media", "../../results")
            self.assertEqual(custom.data, root / "media")
            self.assertEqual(custom.tours, root / "results" / "tours")

    def test_upload_result_and_restart_use_separate_directories(self):
        with TemporaryDirectory() as directory, patch('trajectory_jobs.subprocess.run') as run:
            run.return_value.returncode = 1
            root = Path(directory)
            kwargs = {"data_dir": root / "uploads", "output_dir": root / "results"}
            jobs = TrajectoryJobs(root / "app", **kwargs)
            with patch.object(jobs, 'inspect_video', return_value={"duration": 12, "width": 3840, "height": 1920}):
                key = jobs.register_upload(io.BytesIO(b"movie"), 5, 'movie.mp4')
            self.assertEqual(jobs.sources[key]['path'], root / 'uploads' / key / 'input.mp4')
            self.assertFalse((jobs.cache / key / 'input.mp4').exists())
            write_json(jobs.cache / key / 'trajectory.json', {"test": True})
            restored = TrajectoryJobs(root / 'app', **kwargs)
            self.assertEqual(restored.status(key)['state'], 'complete')
            self.assertEqual(restored.result_path(key), root / 'results' / 'analysis' / key / 'trajectory.json')
            self.assertFalse((root / 'app' / '.data').exists())

    def test_migration_preserves_identity_is_repeatable_and_rejects_conflicts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'legacy'
            key = hashlib.sha256(b'movie').hexdigest()
            old = legacy / '.data' / 'trajectory' / key
            old.mkdir(parents=True)
            (old / 'input.mp4').write_bytes(b'movie')
            write_json(old / 'source.json', {'file': 'input.mp4', 'name': 'movie.mp4', 'size': 5})
            write_json(old / 'trajectory.json', {'source': {'sha256': key}})
            write_json(legacy / 'exports' / 'saved.tour.json', {'version': 1})
            storage = StoragePaths(root / 'videos', root / 'results')
            report, report_path = migrate(legacy, storage)
            self.assertEqual(len(report['videos']), 1)
            self.assertTrue(report_path.exists())
            self.assertEqual((storage.data / key / 'input.mp4').read_bytes(), b'movie')
            self.assertTrue((storage.analysis / key / 'trajectory.json').exists())
            self.assertTrue((storage.tours / 'saved.tour.json').exists())
            self.assertTrue((old / 'input.mp4').exists())
            migrate(legacy, storage)
            (storage.data / key / 'input.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'no overwrite'):
                migrate(legacy, storage)
            self.assertEqual((storage.data / key / 'input.mp4').read_bytes(), b'changed')

    def test_migration_rejects_wrong_source_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / '.data' / 'trajectory' / ('a' * 64)
            old.mkdir(parents=True)
            (old / 'input.mp4').write_bytes(b'movie')
            write_json(old / 'source.json', {'file': 'input.mp4', 'name': 'movie.mp4', 'size': 5})
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                migrate(root, StoragePaths(root / 'videos', root / 'results'))
