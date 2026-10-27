import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.mesh_simsac.remote_worker import execute


class RemoteResumeTests(unittest.TestCase):
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
                    with zipfile.ZipFile(root/'viewer.zip','w') as z:
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


if __name__ == '__main__':
    unittest.main()
