"""Mocked process-boundary tests; never import/run either NN implementation."""
import builtins
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from spacewatch_pipeline import mesh_rgb
from spacewatch_pipeline.io import read, save


def fixture(root,timeout=40):
    prepared=root/'prepared';prepared.mkdir()
    (prepared/'fixture.npz').write_bytes(b'immutable tiny byte fixture, not an array')
    out=root/'infer';out.mkdir()
    return dict(output=str(out),dependencies={'pair.rgb_prepare':dict(root=str(prepared))},
        config=dict(runtime=dict(change_python='/existing/change-python',change_root='/existing/change-models',
            mesh_rgb_checkpoint='/existing/changesim.pth',stage_timeout_seconds=timeout)))


class MeshRgbLifecycleTests(unittest.TestCase):
    def test_native_clis_separate_shared_deadline_same_environment_and_worker_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);job=fixture(root);calls=[]
            original_import=builtins.__import__
            def guarded_import(name,*args,**kwargs):
                if name in {'tools.mesh_simsac.run','tools.mesh_simsac.appearance','torch'}:
                    raise AssertionError('NN implementation imported into worker')
                return original_import(name,*args,**kwargs)
            def run(command,**options):
                calls.append((command,options))
                destination=Path(command[command.index('--root')+1]);(destination/'results').mkdir(exist_ok=True)
                if command[2]=='tools.mesh_simsac.run':
                    save(destination/'inference-status.json',dict(state='complete',pairs=7))
                    save(destination/'results/model.json',dict(model='mocked-only'))
                else:save(destination/'results/object-appearance.json',dict(mocked=True))
                return subprocess.CompletedProcess(command,0)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-existing-lease','TORCH_HOME':'/existing/cache',
                                        'CAPSTONE_GPU_THREAD_ID':'existing-lease'}), \
                 patch.object(mesh_rgb.time,'monotonic',side_effect=[100.,104.,120.]), \
                 patch.object(mesh_rgb.subprocess,'run',side_effect=run), \
                 patch('builtins.__import__',side_effect=guarded_import):
                mesh_rgb.rgb_infer(job)
            self.assertEqual([command[2] for command,_ in calls],
                ['tools.mesh_simsac.run','tools.mesh_simsac.appearance'])
            self.assertEqual([options['timeout'] for _,options in calls],[36.,20.])
            for command,options in calls:
                self.assertEqual(command[0],'/existing/change-python');self.assertEqual(command[1],'-m')
                self.assertEqual(command[command.index('--simsac-source')+1],'/existing/change-models/simsac')
                self.assertEqual(command[command.index('--checkpoint')+1],'/existing/changesim.pth')
                self.assertEqual(command[-2:],['--device','cuda'])
                self.assertTrue(options['check']);self.assertFalse(options['start_new_session'])
                self.assertNotIn('preexec_fn',options);self.assertNotIn('process_group',options)
                self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'],'GPU-existing-lease')
                self.assertEqual(options['env']['CAPSTONE_GPU_THREAD_ID'],'existing-lease')
                self.assertEqual(options['env']['TORCH_HOME'],'/existing/cache')
                self.assertEqual(options['cwd'],Path(mesh_rgb.__file__).resolve().parents[1])
            result=read(Path(job['output'])/'result.json')
            self.assertEqual(result['pairs'],7);self.assertTrue(result['fresh_predictions'])
            self.assertTrue(result['source_rgb']);self.assertFalse(result['gaussian_used'])
            self.assertEqual(result['model'],str(Path(job['output'])/'detection/results/model.json'))
            self.assertEqual((root/'prepared/fixture.npz').read_bytes(),b'immutable tiny byte fixture, not an array')

    def test_shared_deadline_exhaustion_prevents_appearance(self):
        with tempfile.TemporaryDirectory() as temporary:
            job=fixture(Path(temporary))
            with patch.object(mesh_rgb.time,'monotonic',side_effect=[0.,1.,45.]), \
                 patch.object(mesh_rgb.subprocess,'run') as native:
                with self.assertRaises(subprocess.TimeoutExpired):mesh_rgb.rgb_infer(job)
            self.assertEqual(native.call_count,1)
            self.assertEqual(native.call_args.args[0][2],'tools.mesh_simsac.run')
            self.assertFalse((Path(job['output'])/'result.json').exists())

    def test_nn_failure_does_not_start_appearance_or_write_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            job=fixture(Path(temporary))
            with patch.object(mesh_rgb.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['mocked-NN'])) as native:
                with self.assertRaises(subprocess.CalledProcessError):mesh_rgb.rgb_infer(job)
            self.assertEqual(native.call_count,1)
            self.assertFalse((Path(job['output'])/'result.json').exists())

    def test_appearance_interruption_preserves_complete_nn_without_success_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            job=fixture(Path(temporary));calls=[]
            def run(command,**options):
                calls.append(command[2]);destination=Path(command[command.index('--root')+1])
                if command[2]=='tools.mesh_simsac.run':
                    save(destination/'inference-status.json',dict(state='complete',pairs=7))
                    return subprocess.CompletedProcess(command,0)
                raise subprocess.CalledProcessError(-15,command)
            with patch.object(mesh_rgb.subprocess,'run',side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):mesh_rgb.rgb_infer(job)
            self.assertEqual(calls,['tools.mesh_simsac.run','tools.mesh_simsac.appearance'])
            self.assertEqual(read(Path(job['output'])/'detection/inference-status.json')['state'],'complete')
            self.assertFalse((Path(job['output'])/'result.json').exists())

    def test_invalid_timeout_rejected_before_clone_or_process(self):
        for value in [0,-1,True,float('inf'),float('nan'),'40']:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as temporary:
                job=fixture(Path(temporary),value)
                with patch.object(mesh_rgb.subprocess,'run') as native:
                    with self.assertRaises(ValueError):mesh_rgb.rgb_infer(job)
                native.assert_not_called();self.assertFalse((Path(job['output'])/'detection').exists())


if __name__=='__main__':unittest.main()
