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
    def test_retired_render_comparison_routes_are_not_exposed(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[]))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            connection=HTTPConnection('127.0.0.1',server.server_port)
            for path in ['/api/gaussian-change/status','/api/change-jobs/config','/change-analysis.html',
                         '/gaussian-change/gaussian-map-comparison-20261025/index.html']:
                connection.request('GET',path);response=connection.getresponse()
                self.assertEqual(response.status,404,path);response.read()
            connection.close()
        finally:
            server.shutdown();server.server_close()

    def test_dorm_tour_only_exposes_adapted_public_assets_and_preserves_other_datasets(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);dorm=root/'dorm';rescan=root/'rescan';dorm.mkdir();rescan.mkdir()
            (dorm/'tour.json').write_text('{"kind":"dorm_change_tour"}')
            (dorm/'map-1.bin').write_bytes(bytes(range(60)))
            (dorm/'Dorm3-00107.jpg').write_bytes(b'original-photo')
            (dorm/'object-E001-1.bin').write_bytes(b'original-objects')
            (dorm/'object-P0022-1.bin').write_bytes(b'changed-triangles')
            (dorm/'changes-1.bin').write_bytes(b'only-change-surfaces')
            (dorm/'review-P0022.jpg').write_bytes(b'original-photo-comparison')
            (dorm/'ghost-P0027-0-1-00.png').write_bytes(b'object-only-png')
            (dorm/'surface-audit.json').write_text('{"private":true}')
            (dorm/'import-audit.json').write_text('{"private":true}')
            (rescan/'changes.json').write_text('{"original":true}')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],dorm_tour_dir=dorm,rescan_dir=rescan))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                c=HTTPConnection('127.0.0.1',server.server_port)
                c.request('GET','/dorm-tour/tour.json');r=c.getresponse();self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['kind'],'dorm_change_tour')
                c.request('GET','/dorm-tour/map-1.bin',headers={'Range':'bytes=12-23'});r=c.getresponse();self.assertEqual(r.status,206);self.assertEqual(r.read(),bytes(range(12,24)))
                for path in ['/dorm-tour/Dorm3-00107.jpg','/dorm-tour/object-E001-1.bin','/dorm-tour/object-P0022-1.bin','/dorm-tour/changes-1.bin','/dorm-tour/review-P0022.jpg','/dorm-tour/ghost-P0027-0-1-00.png']:
                    c.request('GET',path);r=c.getresponse();self.assertEqual(r.status,200);r.read()
                for path in ['/dorm-tour/import-audit.json','/dorm-tour/surface-audit.json','/dorm-tour/%2e%2e/server.py','/dorm-tour/map-2.bin','/dorm-tour/Dorm3-00107.raw']:
                    c.request('GET',path);r=c.getresponse();self.assertEqual(r.status,404);r.read()
                c.request('GET','/rescan/changes.json');r=c.getresponse();self.assertEqual(json.loads(r.read()),{'original':True})
                c.close()
            finally:server.shutdown();server.server_close()

    def test_dorm_published_assets_are_isolated_from_other_captures_and_private_arrays(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);dorm=root/'dorm';legacy=root/'legacy';dorm.mkdir();legacy.mkdir()
            (dorm/'comparison.json').write_text('{"dataset_id":"dorm2-dorm3"}')
            (dorm/'0-comparison.bin').write_bytes(bytes(range(32)))
            (dorm/'alignment-view.json').write_text('{"kind":"dorm_structural_alignment_view"}')
            (dorm/'1-alignment.bin').write_bytes(bytes(range(60)))
            (dorm/'detection.npz').write_bytes(b'private')
            (legacy/'comparison.json').write_text('{"dataset_id":"rescan"}')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],dorm_dir=dorm,residual_dir=legacy))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                connection=HTTPConnection('127.0.0.1',server.server_port)
                for route,dataset in [('/residual-comparison/dorm2-dorm3/comparison.json','dorm2-dorm3'),('/residual-comparison/comparison.json','rescan')]:
                    connection.request('GET',route);response=connection.getresponse()
                    self.assertEqual(response.status,200);self.assertEqual(json.loads(response.read())['dataset_id'],dataset)
                connection.request('GET','/residual-comparison/dorm2-dorm3/0-comparison.bin',headers={'Range':'bytes=2-5'})
                response=connection.getresponse();self.assertEqual(response.status,206);self.assertEqual(response.read(),bytes(range(2,6)))
                connection.request('GET','/residual-comparison/dorm2-dorm3/alignment-view.json')
                response=connection.getresponse();self.assertEqual(response.status,200);self.assertEqual(json.loads(response.read())['kind'],'dorm_structural_alignment_view')
                connection.request('GET','/residual-comparison/dorm2-dorm3/1-alignment.bin',headers={'Range':'bytes=12-23'})
                response=connection.getresponse();self.assertEqual(response.status,206);self.assertEqual(response.read(),bytes(range(12,24)))
                for route in ['/residual-comparison/dorm2-dorm3/detection.npz','/residual-comparison/dorm2-dorm3/%2e%2e/comparison.json']:
                    connection.request('GET',route);response=connection.getresponse();self.assertEqual(response.status,404);response.read()
                connection.close()
            finally:server.shutdown();server.server_close()

    def test_motion_assessment_is_separate_and_only_the_public_report_is_served(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);legacy=root/'rescan';legacy.mkdir()
            (legacy/'changes.json').write_text('{"original":true}')
            report=root/'motion-assessment.json';report.write_text('{"kind":"local_background_motion_assessment"}')
            (root/'support-audit.json').write_text('{"private":true}')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],rescan_dir=legacy,motion_assessment_path=report))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                c=HTTPConnection('127.0.0.1',server.server_port)
                c.request('GET','/rescan/motion-assessment.json');r=c.getresponse()
                self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['kind'],'local_background_motion_assessment')
                c.request('GET','/rescan/changes.json');r=c.getresponse();self.assertTrue(json.loads(r.read())['original'])
                for url in ('/rescan/support-audit.json','/rescan/../support-audit.json'):
                    c.request('GET',url);r=c.getresponse();self.assertEqual(r.status,404);r.read()
                c.close()
            finally:server.shutdown();server.server_close()

    def test_frozen_transfer_route_keeps_previous_result_and_private_data_separate(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);original=root/'original';transfer=root/'transfer'
            original.mkdir();transfer.mkdir()
            (original/'comparison.json').write_text('{"capture":"rescan"}')
            (transfer/'comparison.json').write_text('{"capture":"unused-bonn"}')
            (transfer/'frame-0-000005.jpg').write_bytes(b'original-photo')
            (transfer/'prediction.npz').write_bytes(b'private')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],residual_dir=original,transfer_dir=transfer))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                connection=HTTPConnection('127.0.0.1',server.server_port)
                for route,value in [('/residual-comparison/comparison.json','rescan'),
                    ('/residual-comparison/bonn-placement/comparison.json','unused-bonn')]:
                    connection.request('GET',route);response=connection.getresponse()
                    self.assertEqual(response.status,200);self.assertEqual(json.loads(response.read())['capture'],value)
                connection.request('GET','/residual-comparison/bonn-placement/frame-0-000005.jpg')
                response=connection.getresponse();self.assertEqual(response.read(),b'original-photo')
                for route in ['/residual-comparison/bonn-placement/prediction.npz',
                    '/residual-comparison/bonn-placement/%2e%2e/comparison.json',
                    '/residual-comparison/unknown/comparison.json']:
                    connection.request('GET',route);response=connection.getresponse();self.assertEqual(response.status,404);response.read()
                connection.close()
            finally:server.shutdown();server.server_close()

    def test_residual_experiment_is_separate_and_does_not_replace_default_results(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);experiment=root/'experiment';legacy=root/'legacy'
            experiment.mkdir();legacy.mkdir()
            (experiment/'comparison.json').write_text('{"kind":"reviewed_roi_residual_comparison"}')
            (experiment/'0-empirical.bin').write_bytes(bytes(range(32)))
            (experiment/'measurements.npz').write_bytes(b'private')
            (legacy/'changes.json').write_text('{"fixed_sigma":0.02}')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],independent_dir=legacy,residual_dir=experiment))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                c=HTTPConnection('127.0.0.1',server.server_port)
                c.request('GET','/m3c2-independent/changes.json');r=c.getresponse();self.assertEqual(json.loads(r.read()),{'fixed_sigma':.02})
                c.request('GET','/residual-comparison/comparison.json');r=c.getresponse();self.assertEqual(r.status,200);r.read()
                c.request('GET','/residual-comparison/0-empirical.bin',headers={'Range':'bytes=4-11'});r=c.getresponse();self.assertEqual(r.status,206);self.assertEqual(r.read(),bytes(range(4,12)))
                for url in ('/residual-comparison/measurements.npz','/residual-comparison/%2e%2e/server.py','/residual-comparison/frame-0-000001.raw'):
                    c.request('GET',url);r=c.getresponse();self.assertEqual(r.status,404);r.read()
                c.close()
            finally:server.shutdown();server.server_close()

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
