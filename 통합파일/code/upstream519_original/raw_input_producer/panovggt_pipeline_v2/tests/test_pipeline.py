from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import (read, save, record, verify, validate_config, make_runtime, windows,
                    check_boundary_policy, retry_optimization, select_match_files)
from masks import correct_mask
from run import Executor


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read(HERE/'configs/dorm3.json')

    def tearDown(self):
        self.temp.cleanup()

    def test_window_coverage_and_tail(self):
        ws = windows(248, 10, 5)
        self.assertEqual(len(ws), 49)
        self.assertEqual(ws[-1], list(range(238, 248)))
        self.assertEqual(set(sum(ws, [])), set(range(248)))

    def test_mask_periodic_seam_and_explicit_frame_rule(self):
        rules = {'coordinate_size': [20, 10], 'periodic_dilation_kernel': [5, 3],
                 'max_fraction': .9, 'polygons': {'1': [[[8, 3], [11, 3], [11, 6], [8, 6]]]}}
        base = np.zeros((10, 20), np.uint8)
        base[5, 0] = 255
        a, b = correct_mask(base, 0, rules), correct_mask(base, 1, rules)
        self.assertTrue(a[5, 19] > 0)
        self.assertEqual(a[4, 9], 0)
        self.assertEqual(b[4, 9], 255)
        self.assertTrue(np.array_equal(base[:, 1:], np.zeros((10, 19))))

    def test_unknown_boundary_fails_instead_of_guessing(self):
        policy = self.config['boundary']
        self.assertEqual(check_boundary_policy([list(range(246)), [246, 247]], policy), [[245, 246]])
        with self.assertRaisesRegex(ValueError, 'Unlisted'):
            check_boundary_policy([list(range(20)), list(range(20, 248))], policy)
        with self.assertRaises(ValueError):
            check_boundary_policy([[0, 2], [1, 3]], policy)
        with self.assertRaises(ValueError):
            check_boundary_policy([[0], [1]], {**policy, 'policy': 'fail'})

    def test_retry_warmstarts_and_stops_at_success(self):
        seen, saved = [], []
        def solve(value, options):
            seen.append(value.copy())
            return value+1, {'success': len(seen) == 2, 'seconds': 0}
        result, reports = retry_optimization(np.array([0.]), [{'max_nfev': 1}]*3, solve,
                                             lambda *args: saved.append(args))
        self.assertEqual([x.item() for x in seen], [0, 1])
        self.assertEqual(result.item(), 2)
        self.assertEqual(len(saved), 2)

    def test_retry_exhaustion_cannot_export_failed_solution(self):
        with self.assertRaises(RuntimeError):
            retry_optimization(np.zeros(1), [{}], lambda p, o: (p, {'success': False}), lambda *args: None)

    def test_recovery_observations_use_only_finished_sources(self):
        for source in ['matches', 'hires_matches']:
            (self.root/source).mkdir()
            np.savez(self.root/source/'000_001.npz', identity=source)
        before = select_match_files(self.root, ['matches'])
        after = select_match_files(self.root, ['matches', 'hires_matches'])
        self.assertEqual(before['000_001.npz'].parent.name, 'matches')
        self.assertEqual(after['000_001.npz'].parent.name, 'hires_matches')
        self.assertNotEqual(record(before['000_001.npz'])['sha256'], record(after['000_001.npz'])['sha256'])

    def test_stage_waits_then_seals_and_resume_detects_mutation(self):
        output = self.root/'done.txt'
        command = [sys.executable, '-c', 'import time,pathlib;time.sleep(.05);pathlib.Path('+repr(str(output))+').write_text("complete")']
        ex = Executor(self.root, 'identity')
        ex.stage('test', command, ['done.txt'])
        self.assertEqual(output.read_text(), 'complete')
        ex2 = Executor(self.root, 'identity', lambda *a, **k: self.fail('Completed stage was rerun'))
        ex2.stage('test', command, ['done.txt'])
        output.write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            Executor(self.root, 'identity').stage('test', command, ['done.txt'])

    def test_failed_stage_has_no_completion_receipt(self):
        with self.assertRaises(subprocess.CalledProcessError):
            Executor(self.root, 'x').stage('failed', [sys.executable, '-c', 'raise RuntimeError("failed")'], ['missing'])
        self.assertFalse((self.root/'stages/failed.json').exists())

    def test_snapshot_restores_correct_phase_on_resume(self):
        p = self.root/'graph.json'
        command = [sys.executable, '-c', 'from pathlib import Path;Path('+repr(str(p))+').write_text("phase1")']
        Executor(self.root, 'x').stage('obs', command, ['graph.json'], snapshot='phases/one')
        p.write_text('later-phase')
        Executor(self.root, 'x').stage('obs', command, ['graph.json'], snapshot='phases/one')
        self.assertEqual(p.read_text(), 'phase1')

    def test_symlink_target_change_is_detected(self):
        a, b, link = self.root/'a', self.root/'b', self.root/'link'
        a.write_text('a'); b.write_text('b'); link.symlink_to(a)
        sealed = record(link)
        link.unlink(); link.symlink_to(b)
        with self.assertRaises(ValueError):
            verify(sealed)

    def test_config_rejects_wrong_mask_dimensions(self):
        validate_config(self.config)
        self.config['mask']['coordinate_size'] = [2072, 1036]
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_numerical_camera_pose_solver_on_known_synthetic_geometry(self):
        save(self.root/'config.json', make_runtime(self.config, self.root))
        save(self.root/'input/manifest.json', {'frames': [{'timestamp_s': i*.5} for i in range(248)]})
        result = subprocess.run([sys.executable, str(HERE/'workers/refine.py'), 'preflight'],
                 env={**os.environ, 'PANOVGGT_RUN_ROOT': str(self.root), 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'},
                 text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(read(self.root/'optimizer_preflight.json')['passed'])


if __name__ == '__main__':
    unittest.main()
