import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import make_handler

class PipelineServerTests(unittest.TestCase):
    def test_only_allowlisted_public_files_with_range_and_default_pipeline_page(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);public=root/'public';public.mkdir()
            (root/'private.txt').write_text('secret');(public/'outside.bin').symlink_to(root/'private.txt')
            (public/'mesh.bin').write_bytes(bytes(range(64)))
            (public/'weights.pt').write_text('private')
            (public/'manifest.json').write_text(json.dumps(dict(bundle_id='test',assets=['mesh.bin','outside.bin','../private.txt'])))
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],pipeline_dir=public))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                c=HTTPConnection('127.0.0.1',server.server_port)
                c.request('GET','/');r=c.getresponse();self.assertEqual(r.status,200);self.assertIn(b'survey.js',r.read())
                c.request('GET','/api/pipeline');r=c.getresponse();self.assertEqual(json.loads(r.read())['bundle_id'],'test')
                c.request('GET','/pipeline-assets/mesh.bin',headers={'Range':'bytes=8-15'});r=c.getresponse();self.assertEqual(r.status,206);self.assertEqual(r.read(),bytes(range(8,16)))
                for route in ['/pipeline-assets/weights.pt','/pipeline-assets/%2e%2e/private.txt','/pipeline-assets/outside.bin','/server.py','/manifest.json']:
                    c.request('GET',route);r=c.getresponse();self.assertEqual(r.status,404,route);r.read()
                c.request('POST','/api/videos',body=b'anything');r=c.getresponse();self.assertEqual(r.status,503);r.read();c.close()
            finally:server.shutdown();server.server_close()
