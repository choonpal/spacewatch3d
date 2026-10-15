import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import make_handler, parse_range


class RangeTests(unittest.TestCase):
    def test_bonn_routes_serve_viewer_assets_without_exposing_raw_depth(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'changes.json').write_text('{"dataset_id":"bonn-removal"}')
            (root/'frame-0-000040.jpg').write_bytes(b'original-bonn-frame')
            (root/'frame-0-000040.raw').write_bytes(b'private-depth')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],rgbd_dir=root))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                c=HTTPConnection('127.0.0.1',server.server_port)
                c.request('GET','/datasets/bonn-removal/changes.json');r=c.getresponse()
                self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['dataset_id'],'bonn-removal')
                c.request('GET','/datasets/bonn-removal/frame-0-000040.jpg');r=c.getresponse()
                self.assertEqual(r.status,200);self.assertEqual(r.read(),b'original-bonn-frame')
                for path in ['/datasets/bonn-removal/frame-0-000040.raw','/datasets/bonn-removal/%2e%2e/server.py','/datasets/other/changes.json']:
                    c.request('GET',path);r=c.getresponse();self.assertEqual(r.status,404);r.read()
                c.close()
            finally:server.shutdown();server.server_close()

    def test_independent_change_routes_keep_raw_inputs_and_private_fields_private(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'changes.json').write_text('{"kind":"m3c2_independent_changes"}')
            (root / '0-region-ids.bin').write_bytes(bytes(range(20)))
            (root / 'frame-0-000015.jpg').write_bytes(b'original-frame-fixture')
            (root / 'candidate-fields.npz').write_bytes(b'private')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler([], [], independent_dir=root))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                c = HTTPConnection('127.0.0.1', server.server_port)
                c.request('GET', '/m3c2-independent/0-region-ids.bin', headers={'Range':'bytes=2-9'})
                r = c.getresponse()
                self.assertEqual(r.status, 206); self.assertEqual(r.read(), bytes(range(2,10)))
                c.request('GET', '/m3c2-independent/frame-0-000015.jpg')
                r = c.getresponse(); self.assertEqual(r.status, 200); r.read()
                for path in ['/m3c2-independent/candidate-fields.npz','/m3c2-independent/frame-0-000015.raw','/m3c2-independent/%2e%2e/server.py']:
                    c.request('GET', path); r = c.getresponse(); self.assertEqual(r.status, 404); r.read()
                c.close()
            finally:
                server.shutdown(); server.server_close()

    def test_mesh_comparison_only_serves_public_viewer_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'comparison.json').write_text('{"kind":"geometry_method_comparison"}')
            (root / '0-comparison.bin').write_bytes(bytes(range(60)))
            (root / 'core-fields.npz').write_bytes(b'private computation data')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler([], [], mesh_comparison_dir=root))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                c = HTTPConnection('127.0.0.1', server.server_port)
                c.request('GET', '/mesh-comparison/0-comparison.bin', headers={'Range': 'bytes=28-43'})
                r = c.getresponse()
                self.assertEqual(r.status, 206)
                self.assertEqual(r.read(), bytes(range(28, 44)))
                for url in ['/mesh-comparison/core-fields.npz', '/mesh-comparison/../server.py', '/mesh-comparison/%2e%2e/server.py', '/mesh-comparison/2-comparison.bin']:
                    c.request('GET', url)
                    r = c.getresponse()
                    self.assertEqual(r.status, 404)
                    r.read()
                c.request('GET', '/mesh-comparison/comparison.json')
                r = c.getresponse()
                self.assertEqual(json.loads(r.read())['kind'], 'geometry_method_comparison')
                c.request('GET', '/mesh-comparison.html')
                r = c.getresponse()
                self.assertEqual(r.status, 200)
                self.assertIn(b'M3C2', r.read())
                c.close()
            finally:
                server.shutdown()
                server.server_close()

    def test_ranges(self):
        self.assertEqual(parse_range('bytes=0-9', 100), (0, 9))
        self.assertEqual(parse_range('bytes=50-', 100), (50, 99))
        self.assertEqual(parse_range('bytes=-10', 100), (90, 99))
        self.assertEqual(parse_range('bytes=90-999', 100), (90, 99))
        for invalid in ['bytes=100-', 'bytes=9-0', 'bytes=-0', 'bytes=0-1,4-5', 'bytes=-', 'bad']:
            with self.assertRaises(ValueError):
                parse_range(invalid, 100)

    def test_stream_and_private_file_boundary(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'sample.mp4'
            video.write_bytes(bytes(range(100)))
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler([video], [{'name': 'sample.mp4'}]))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                c = HTTPConnection('127.0.0.1', server.server_port)
                c.request('GET', '/media/0', headers={'Range': 'bytes=20-29'})
                r = c.getresponse()
                self.assertEqual(r.status, 206)
                self.assertEqual(r.getheader('Content-Range'), 'bytes 20-29/100')
                self.assertEqual(r.read(), bytes(range(20, 30)))
                c.request('HEAD', '/media/0')
                r = c.getresponse()
                self.assertEqual(r.getheader('Content-Length'), '100')
                self.assertEqual(r.read(), b'')
                for path in ['/../README.md', '/.local.json', '/server.py', '/media/1', '/assets/../../.local.json']:
                    c.request('GET', path)
                    r = c.getresponse()
                    self.assertEqual(r.status, 404)
                    r.read()
                c.request('GET', '/media/0', headers={'Range': 'bytes=1000-'})
                r = c.getresponse()
                self.assertEqual(r.status, 416)
                r.read()
                c.request('GET', '/api/config')
                r = c.getresponse()
                self.assertEqual(json.loads(r.read())['media'][0]['name'], 'sample.mp4')
                c.request('GET', '/keyframe-transition.js')
                r = c.getresponse()
                self.assertEqual(r.status, 200)
                self.assertIn(b'export class KeyframeTransition', r.read())
                c.close()
            finally:
                server.shutdown()
                server.server_close()

    def test_evidence_downloads_do_not_expose_arbitrary_output_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evidence.json').write_text('{"version":1}')
            (root / 'region-001.bin').write_bytes(bytes(range(24)))
            (root / 'private.json').write_text('private')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler([], [], evidence_dir=root))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                c = HTTPConnection('127.0.0.1', server.server_port)
                c.request('GET', '/evidence/region-001.bin', headers={'Range': 'bytes=12-23'})
                response = c.getresponse()
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), bytes(range(12, 24)))
                for url in ['/evidence/private.json', '/evidence/../private.json', '/evidence/%2e%2e/private.json', '/evidence/region-1.bin']:
                    c.request('GET', url)
                    response = c.getresponse()
                    self.assertEqual(response.status, 404)
                    response.read()
                c.request('GET', '/evidence/evidence.json')
                self.assertEqual(json.loads(c.getresponse().read()), {'version': 1})
                c.close()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
