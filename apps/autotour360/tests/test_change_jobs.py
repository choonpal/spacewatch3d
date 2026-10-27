import io
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from change_jobs import ChangeJobs, FIELDS


class ChangeJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root/'runtime.local.json'
        self.runtime.write_text(json.dumps({'simsac_source': '/admin/models', 'simsac_checkpoint': '/admin/weights', 'person_checkpoint': '/admin/person', 'local_python': sys.executable}))
        self.jobs = ChangeJobs(self.root/'jobs', self.runtime)

    def upload_all(self):
        job = self.jobs.create('upload'); key = job['id']
        for field in FIELDS:
            suffix = '.npz' if field.endswith('mesh') else '.ply' if field.endswith('gaussian') else '.json'
            self.jobs.upload(key, field, io.BytesIO(b'input'), 5, 'user'+suffix)
        return key

    def test_six_uploaded_files_become_a_new_cli_input_and_admin_runtime_is_preserved(self):
        key = self.upload_all()
        descriptor = {'units': 'm', 'before': {'mesh': '/untrusted', 'camera_convention': 'opencv'}, 'after': {'camera_convention': 'opencv'}}
        process = Mock(); process.pid = 123; process.poll.return_value = None
        with patch('change_jobs.subprocess.Popen', return_value=process) as launch:
            result = self.jobs.start(key, descriptor)
        p = self.jobs.folder(key); manifest = json.loads((p/'input.local.json').read_text())
        self.assertEqual(result['state'], 'running')
        self.assertTrue(manifest['before']['mesh'].startswith(str(p/'inputs')))
        config = json.loads((p/'runtime.local.json').read_text())
        self.assertEqual(config['input_location'], 'local')
        self.assertEqual(config['simsac_checkpoint'], '/admin/weights')
        self.assertIn('tools.gaussian_change.pipeline', launch.call_args.args[0])

    def test_remote_completion_is_not_shown_until_the_local_result_has_arrived(self):
        key = self.upload_all(); p = self.jobs.folder(key)
        data = json.loads((p/'job.json').read_text()); data['state'] = 'running'; (p/'job.json').write_text(json.dumps(data))
        (p/'result').mkdir(); (p/'result/status.json').write_text(json.dumps({'state': 'complete'}))
        process = Mock(); process.poll.return_value = None; self.jobs.processes[key] = process
        self.assertEqual(self.jobs.status(key)['stage'], 'receiving_results')
        (p/'result/changes.json').write_text('{}'); process.poll.return_value = 0
        self.assertEqual(self.jobs.status(key)['state'], 'complete')
        with self.assertRaises(KeyError): self.jobs.asset(key, '../runtime.local.json')
        with self.assertRaises(KeyError): self.jobs.asset(key, 'source.zip')

    def test_incomplete_inputs_and_duplicate_uploads_cannot_launch(self):
        key = self.jobs.create('upload')['id']
        self.jobs.upload(key, 'before-cameras', io.BytesIO(b'{}'), 2, 'poses.json')
        with self.assertRaises(ValueError): self.jobs.upload(key, 'before-cameras', io.BytesIO(b'{}'), 2, 'poses.json')
        with patch('change_jobs.subprocess.Popen') as launch:
            with self.assertRaises(ValueError): self.jobs.start(key, {})
            launch.assert_not_called()

    def test_explicit_4k_profile_preserves_the_supplied_detection_threshold(self):
        key=self.upload_all();proc=Mock();proc.pid=124;proc.poll.return_value=None
        with patch('change_jobs.subprocess.Popen',return_value=proc):
            self.jobs.start(key,{'units':'m','before':{},'after':{},'settings':{'probability_threshold':.55},'analysis_profile':'4k-perspective'})
        manifest=json.loads((self.jobs.folder(key)/'input.local.json').read_text());settings=manifest['settings']
        self.assertEqual((settings['width'],settings['height'],settings['cube_size']),(4096,2048,2048))
        self.assertEqual(settings['inference_mode'],'perspective_tiles');self.assertEqual(settings['probability_threshold'],.55)

    def test_run_button_binds_registered_instance_observations_by_mesh_hash(self):
        mesh=b'new-mesh';mesh_hash=hashlib.sha256(mesh).hexdigest()
        packet=self.root/'instances.json';packet.write_text(json.dumps({'schema':'mesh-instance-observations-v1','mesh_sha256':mesh_hash,'observations':[{'id':'a'}],'cameras':[{'frame':0}]}))
        registry=self.root/'registry.json';registry.write_text(json.dumps({'meshes':{mesh_hash:{'file':str(packet)}}}))
        runtime=json.loads(self.runtime.read_text());runtime['instance_observations_registry']=str(registry);self.runtime.write_text(json.dumps(runtime))
        key=self.jobs.create('upload')['id']
        for field in FIELDS:
            suffix='.npz' if field.endswith('mesh') else '.ply' if field.endswith('gaussian') else '.json'
            content=mesh if field.endswith('mesh') else b'gs' if field.endswith('gaussian') else b'{"cameras":[]}'
            self.jobs.upload(key,field,io.BytesIO(content),len(content),'new'+suffix)
        proc=Mock();proc.pid=123;proc.poll.return_value=None
        with patch('change_jobs.subprocess.Popen',return_value=proc):
            status=self.jobs.start(key,{'units':'m','before':{},'after':{}})
        folder=self.jobs.folder(key);manifest=json.loads((folder/'input.local.json').read_text())
        self.assertEqual(len(status['instance_bindings']),2)
        self.assertIn('instance_segmentation',json.loads(Path(manifest['before']['cameras']).read_text()))
        self.assertNotIn('instance_segmentation',json.loads((folder/'inputs/before-cameras.json').read_text()))


if __name__ == '__main__': unittest.main()
