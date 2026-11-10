import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

spec = importlib.util.spec_from_file_location('final_demo', Path(__file__).with_name('demo.py'))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'demo').mkdir()
        (root / 'code/apps/autotour360').mkdir(parents=True)
        (root / 'RELEASE.json').write_text(json.dumps({'demo': {'app_assets': ['pipeline.html']}}))
        (root / 'demo/manifest.json').write_text(json.dumps({'captures': [{'id': 'dorm5'}], 'assets': ['image.png']}))
        (root / 'demo/image.png').write_bytes(b'0123456789')
        (root / 'code/apps/autotour360/pipeline.html').write_text('<html>demo</html>')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), demo.make_handler(root))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_manifest_and_actual_asset(self):
        with urlopen(self.base + '/api/pipeline') as response:
            self.assertEqual(json.load(response)['captures'][0]['id'], 'dorm5')
        with urlopen(self.base + '/pipeline-assets/image.png') as response:
            self.assertEqual(response.read(), b'0123456789')

    def test_head_and_range(self):
        with urlopen(Request(self.base + '/pipeline-assets/image.png', method='HEAD')) as response:
            self.assertEqual(response.headers['Content-Length'], '10')
            self.assertEqual(response.read(), b'')
        with urlopen(Request(self.base + '/pipeline-assets/image.png', headers={'Range': 'bytes=2-5'})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'2345')
        self.assertEqual(demo.file_range('bytes=-3', 10), (7, 9))

    def test_bad_range_rejected(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base + '/pipeline-assets/image.png', headers={'Range': 'bytes=30-40'}))
        self.assertEqual(error.exception.code, 416)

    def test_non_allowlisted_and_traversal_rejected(self):
        for path in ['/RELEASE.json', '/code/spacewatch.py', '/pipeline-assets/../RELEASE.json', '/download.zip']:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
