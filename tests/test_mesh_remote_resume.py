import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.mesh_simsac.remote_worker import execute,viewer_archive_script
import subprocess
import sys


class RemoteResumeTests(unittest.TestCase):
    def test_status_writer_survives_a_temporary_windows_reader_lock(self):
        from tools.mesh_simsac.core import write
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'status.json';replace=Path.replace;attempts=[]
            def locked(path,destination):
                attempts.append(path)
                if len(attempts)<3:raise PermissionError('temporary Windows reader lock')
                return replace(path,destination)
            with patch.object(Path,'replace',locked),patch('tools.mesh_simsac.core.time.sleep'):
                write(target,{'state':'connected'})
            self.assertEqual(json.loads(target.read_text()),{'state':'connected'})
            self.assertEqual(len(attempts),3)
            self.assertFalse(list(Path(temp).glob('*.tmp')))

    def test_temporary_disconnect_rejoins_the_same_worker_without_relaunch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            receipt=dict(directory='/fixture/existing-worker',pid=123)
            (root/'remote.local.json').write_text(json.dumps(receipt))
            calls=[]
            def call(**request):
                calls.append(request)
                if len(calls)==1:
                    raise ConnectionError('temporary relay delay')
                if len(calls)==2:
                    self.assertEqual(json.loads((root/'status.json').read_text())['stage'],'waiting_for_connection')
                    return json.dumps(dict(state='complete',counts={}))
                if request.get('get'):
                    with zipfile.ZipFile(request['get'][0][1],'w') as z:
                        z.writestr('index.html','explicit test viewer')
                return ''
            with patch('tools.mesh_simsac.remote_worker.Relay') as relay, \
                 patch('tools.mesh_simsac.remote_worker.time.sleep'):
                relay.return_value.call.side_effect=call
                execute(root/'unused.zip',root,dict(remote_session='fixture',remote_root='/fixture',python='python'))
            self.assertEqual(json.loads((root/'remote.local.json').read_text()),receipt)
            self.assertEqual(json.loads((root/'status.json').read_text())['state'],'complete')
            self.assertFalse(any('upload' in c for c in calls))
            self.assertFalse(any('Popen' in c.get('command','') for c in calls))

    def test_rgb_result_archive_does_not_require_legacy_mesh_viewer_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'result';root.mkdir()
            for name in ['index.html','rgb-viewer.js','rgb-viewer.css','validation.json']:
                (root/name).write_text('fixture')
            (root/'viewer-data.json').write_text(json.dumps({'backend':'simsac_rgb'}))
            archive=Path(temp)/'viewer.zip'
            subprocess.run([sys.executable,'-c',viewer_archive_script(root,archive)],check=True)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('rgb-viewer.js',z.namelist())
                self.assertNotIn('events-viewer.js',z.namelist())


if __name__ == '__main__':
    unittest.main()
