import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_change_jobs as jobs
from tools.mesh_simsac.bundle import unpack


def bundle(setting=1, image=b'original RGB'):
    doc = dict(schema='mesh-simsac-input-v1', settings={'frame_stride': setting})
    for phase in ['before', 'after']:
        doc[phase] = dict(mesh='mesh.npz', frames=[dict(id='0', image='rgb.jpg', world_from_camera=[])])
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        z.writestr('manifest.json', json.dumps(doc))
        z.writestr('mesh.npz', b'fixture mesh')
        z.writestr('rgb.jpg', image)
    return b.getvalue()


class RuntimeJobs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root/'runtime.json'
        jobs.write(self.runtime, dict(simsac_source='fixture', simsac_checkpoint='fixture'))
        self.service = jobs.MeshChangeJobs(self.root/'jobs', self.runtime)

    def tearDown(self):
        self.service.executor.shutdown(wait=True)
        self.temp.cleanup()

    def submit(self, payload):
        status = self.service.submit(io.BytesIO(payload), len(payload))
        deadline = time.monotonic()+10
        while status['state'] not in ['complete', 'failed', 'interrupted']:
            if time.monotonic() > deadline:
                self.fail('worker did not terminate')
            time.sleep(.01)
            status = self.service.status(status['id'])
        return status

    @staticmethod
    def fake_worker(command, **kwargs):
        # Explicit unit-test fixture; production never substitutes model inference.
        root = Path(command[command.index('--output')+1]); public = root/'detection'
        jobs.write(public/'viewer-data.json', dict(counts={'moved': 1}, job_provenance={'fresh_inference': True}))
        (public/'index.html').write_text('test viewer')
        jobs.write(root/'status.json', dict(state='complete'))
        return type('Process', (), {'returncode': 0})()

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    @patch.object(jobs.subprocess, 'run', side_effect=fake_worker)
    def test_first_input_computes_repeat_reuses_and_changes_recompute(self, run, identity):
        first = self.submit(bundle())
        self.assertEqual(first['state'], 'complete'); self.assertFalse(first['cached'])
        second = self.submit(bundle())
        self.assertTrue(second['cached']); self.assertEqual(run.call_count, 1)
        # Persisted cache also works in a new service instance after a restart.
        self.service.executor.shutdown(wait=True)
        self.service = jobs.MeshChangeJobs(self.root/'jobs', self.runtime)
        self.assertTrue(self.submit(bundle())['cached'])
        self.assertFalse(self.submit(bundle(setting=2))['cached'])
        self.assertFalse(self.submit(bundle(image=b'changed image'))['cached'])
        identity.return_value = {'models': {}, 'source': 'revision2'}
        self.assertFalse(self.submit(bundle())['cached'])
        self.assertEqual(run.call_count, 4)
        self.assertTrue(self.service.asset(first['id'], 'index.html').is_file())
        with self.assertRaises(KeyError):
            self.service.asset(first['id'], '../../runtime.json')

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    @patch.object(jobs.subprocess, 'run', side_effect=fake_worker)
    def test_corrupt_cache_recomputes(self, run, identity):
        first = self.submit(bundle())
        self.service.asset(first['id'], 'index.html').write_text('corrupt')
        again = self.submit(bundle())
        self.assertEqual(again['state'], 'complete'); self.assertFalse(again['cached'])
        self.assertEqual(run.call_count, 2)

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    def test_presentation_updates_completed_job_without_mutating_inference_cache(self, identity):
        def worker(command, **kwargs):
            result=self.fake_worker(command,**kwargs)
            public=Path(command[command.index('--output')+1])/'detection'
            (public/'events-viewer.js').write_text('immutable historical renderer')
            return result
        with patch.object(jobs.subprocess,'run',side_effect=worker) as run:
            status=self.submit(bundle())
            folder=self.service.folder(status['id']);data=jobs.read(folder/'job.json')
            cache=self.service.root/'cache'/data['fingerprint'];marker=cache/'complete.json'
            receipt=marker.read_bytes();generation=cache/jobs.read(marker)['generation']
            self.assertEqual((generation/'events-viewer.js').read_text(),'immutable historical renderer')
            for name,source in jobs.PRESENTATION.items():
                self.assertEqual(self.service.asset(status['id'],name),jobs.REPO/'tools/mesh_simsac'/source)
            self.assertEqual(marker.read_bytes(),receipt)
            self.assertEqual((generation/'events-viewer.js').read_text(),'immutable historical renderer')
            self.assertEqual(run.call_count,1)
            self.assertEqual(self.service.status(status['id'])['counts'],{'moved':1})

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    def test_reprocess_preserves_parent_and_requires_bound_probability_provenance(self, identity):
        configuration=jobs.read(self.runtime);configuration['remote_session']='test-session'
        jobs.write(self.runtime,configuration)
        def worker(command,**kwargs):
            runtime=jobs.read(command[command.index('--runtime')+1])
            root=Path(command[command.index('--output')+1]);public=root/'detection'
            if runtime.get('reprocess_source'):
                provenance=dict(fresh_inference=False,reused_probabilities=True,
                    parent_viewer_sha256=runtime['reprocess_parent_viewer_sha256'],new_model_predictions=0)
                jobs.write(public/'viewer-data.json',dict(counts={'unresolved':7},job_provenance=provenance))
                (public/'index.html').write_text('reaggregated fixture')
                jobs.write(root/'status.json',dict(state='complete'))
                return type('Process',(),{'returncode':0})()
            return self.fake_worker(command,**kwargs)
        with patch.object(jobs.subprocess,'run',side_effect=worker) as run:
            parent=self.submit(bundle());parent_file=self.service.asset(parent['id'],'viewer-data.json')
            original=parent_file.read_bytes()
            jobs.write(self.service.folder(parent['id'])/'attempt/result/remote.local.json',dict(directory='/test/rgb-parent'))
            status=self.service.submit_reprocess(parent['id']);deadline=time.monotonic()+10
            while status['state'] not in ['complete','failed']:
                if time.monotonic()>deadline:self.fail('reaggregation did not terminate')
                time.sleep(.01);status=self.service.status(status['id'])
            self.assertEqual(status['state'],'complete',status.get('message'))
            self.assertEqual(status['counts'],{'unresolved':7})
            self.assertEqual(parent_file.read_bytes(),original)
            result=jobs.read(self.service.asset(status['id'],'viewer-data.json'))
            self.assertFalse(result['job_provenance']['fresh_inference'])
            self.assertEqual(result['job_provenance']['new_model_predictions'],0)
            self.assertEqual(run.call_count,2)

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    @patch.object(jobs.subprocess, 'run', side_effect=RuntimeError('model unavailable'))
    def test_failure_has_no_viewer_and_is_not_cached(self, run, identity):
        result = self.submit(bundle())
        self.assertEqual(result['state'], 'failed'); self.assertIsNone(result['result_url'])
        self.assertIn('model unavailable', result['message'])
        self.assertFalse(list((self.root/'jobs').rglob('complete.json')))

    def test_invalid_and_traversing_bundles_fail_before_inference(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr('../escape', 'bad')
        with self.assertRaises(ValueError):
            unpack(io.BytesIO(b.getvalue()), self.root/'unsafe')
        self.assertFalse((self.root/'escape').exists())
        result = self.submit(b'not a zip')
        self.assertEqual(result['state'], 'failed')
        self.assertIsNone(result['result_url'])

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    @patch.object(jobs.subprocess, 'run', side_effect=fake_worker)
    def test_server_selection_prepares_current_inputs_then_infers_and_caches(self, run, identity):
        catalog = self.root/'catalog.json'
        jobs.write(catalog, {'captures': [dict(id='dorm4', candidate_regions=98, mesh='/private/a'),
                                          dict(id='dorm5', candidate_regions=93, mesh='/private/b')]})
        runtime = jobs.read(self.runtime); runtime['capture_catalog'] = str(catalog); jobs.write(self.runtime, runtime)
        self.assertNotIn('mesh', self.service.sources()[0])
        for request in [{'before':'dorm4','after':'dorm4'}, {'before':'dorm4','after':'/private/other'},
                        {'before':'dorm4','after':'dorm5','mesh':'injected'}]:
            with self.assertRaises(ValueError):
                self.service.submit_sources(request)

        def prepare(runtime, request, folder):
            self.assertEqual(request, {'before':'dorm4','after':'dorm5'})
            (folder/'upload.zip').write_bytes(bundle())

        with patch('mesh_capture_sources.prepare', side_effect=prepare) as importer:
            for expected_cached in [False, True]:
                status = self.service.submit_sources({'before':'dorm4','after':'dorm5'})
                deadline = time.monotonic()+10
                while status['state'] not in ['complete','failed'] and time.monotonic()<deadline:
                    time.sleep(.01); status = self.service.status(status['id'])
                self.assertEqual(status['state'], 'complete', status)
                self.assertEqual(status['cached'], expected_cached)
                self.assertEqual(status['source_selection']['before'], 'dorm4')
            self.assertEqual(importer.call_count, 2)
            self.assertEqual(run.call_count, 1)

    @patch.object(jobs, 'model_identity', return_value={'models': {}, 'source': 'revision1'})
    @patch.object(jobs.subprocess, 'run', side_effect=fake_worker)
    def test_http_upload_starts_job_and_only_serves_completed_public_assets(self, run, identity):
        from server import make_handler
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler([], [], mesh_change_jobs=self.service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection('127.0.0.1', server.server_port)
            connection.request('POST', '/api/mesh-change-jobs', body=bundle(), headers={'Content-Type': 'application/zip'})
            response = connection.getresponse()
            self.assertEqual(response.status, 202)
            key = json.loads(response.read())['id']
            deadline = time.monotonic()+5
            while self.service.status(key)['state'] not in ['complete', 'failed'] and time.monotonic()<deadline:
                time.sleep(.01)
            connection.request('GET', '/api/mesh-change-jobs/'+key)
            response = connection.getresponse(); result = json.loads(response.read())
            self.assertEqual(result['state'], 'complete')
            connection.request('GET', result['result_url'])
            response = connection.getresponse(); self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b'test viewer')
            for name in ['../../runtime.json', 'manifest.json', 'results/model.json', 'views/E0.npz']:
                connection.request('GET', '/api/mesh-change-jobs/'+key+'/files/'+name)
                response = connection.getresponse(); self.assertEqual(response.status, 404); response.read()
            connection.request('POST', '/api/mesh-change-jobs', body=bundle(), headers={'Origin': 'https://unrelated.invalid'})
            response = connection.getresponse(); self.assertEqual(response.status, 403); response.read()
            connection.close()
        finally:
            server.shutdown(); server.server_close()


if __name__ == '__main__':
    unittest.main()
