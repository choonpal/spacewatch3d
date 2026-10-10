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
                c.close()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
