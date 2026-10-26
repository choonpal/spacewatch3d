import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_assets import TOUR,SHARED,asset_path,status
from server import make_handler


class GaussianAssets(unittest.TestCase):
    def fixture(self,root):
        folder=root/TOUR;folder.mkdir();shared=root/SHARED;(shared/'models').mkdir(parents=True);(shared/'vendor').mkdir()
        data={'zones':[{'id':'one'}],'models':{'before':[{'file':f'../{SHARED}/models/before.ply'}],'after':[{'file':f'../{SHARED}/models/after-R01.ply'}]}}
        for name in ['index.html','tour-routes.json','tour-candidates.json','object-evidence.json']:(folder/name).write_text('{}')
        (folder/'gaussian-data.json').write_text(json.dumps(data))
        (folder/'change-regions.json').write_text(json.dumps({'modes':{'filtered':{'counts':{'regions':1,'unique_events':1}}}}))
        for name in ['before.ply','after-R01.ply']:(shared/'models'/name).write_bytes(bytes(range(32)))
        for name in ['three.module.js','spark.module.js']:(shared/'vendor'/name).write_text('// vendor')
        return folder,shared

    def test_status_uses_real_counts_and_missing_models_are_not_reported_ready(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);self.assertFalse(status(root)['available']);self.assertNotIn('counts',status(root))
            folder,shared=self.fixture(root);result=status(root)
            self.assertTrue(result['available']);self.assertEqual(result['keyframes'],1);self.assertEqual(result['counts']['unique_events'],1)
            (shared/'models/before.ply').unlink();self.assertFalse(status(root)['available'])

    def test_only_public_gaussian_assets_are_served_with_ranges_and_old_routes_survive(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder,shared=self.fixture(root);(folder/'object-meshes').mkdir();(folder/'object-meshes/S0-010@I0-005.json').write_text('{}')
            (folder/'protocol.json').write_text('{"private_input_paths":true}');(folder/'depth.npz').write_bytes(b'private')
            rescan=root/'legacy';rescan.mkdir();(rescan/'changes.json').write_text('{"original":true}')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],gaussian_change_dir=root,rescan_dir=rescan))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                client=HTTPConnection('127.0.0.1',server.server_port)
                client.request('GET','/api/gaussian-change/status');response=client.getresponse();self.assertEqual(response.status,200);self.assertTrue(json.loads(response.read())['available'])
                client.request('GET',f'/gaussian-change/{SHARED}/models/before.ply',headers={'Range':'bytes=4-9'});response=client.getresponse();self.assertEqual(response.status,206);self.assertEqual(response.read(),bytes(range(4,10)))
                client.request('GET',f'/gaussian-change/{TOUR}/object-meshes/S0-010%40I0-005.json');response=client.getresponse();self.assertEqual(response.status,200);response.read()
                for path in [f'/gaussian-change/{TOUR}/protocol.json',f'/gaussian-change/{TOUR}/depth.npz',f'/gaussian-change/{TOUR}/%2e%2e/server.py',f'/gaussian-change/{SHARED}/vendor/../../server.py']:
                    client.request('GET',path);response=client.getresponse();self.assertEqual(response.status,404,path);response.read()
                client.request('GET','/rescan/changes.json');response=client.getresponse();self.assertEqual(json.loads(response.read()),{'original':True})
                client.request('HEAD',f'/gaussian-change/{SHARED}/vendor/three.module.js');response=client.getresponse();self.assertEqual(response.status,200);self.assertEqual(response.read(),b'')
                client.close()
            finally:server.shutdown();server.server_close()

    def test_symbolic_link_outside_published_root_is_not_served(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder,shared=self.fixture(root);outside=root.parent/(root.name+'-private')
            outside.write_text('private')
            try:
                link=shared/'vendor/escape.js'
                try:link.symlink_to(outside)
                except OSError:self.skipTest('Symlink creation is unavailable')
                self.assertIsNone(asset_path(root,f'/gaussian-change/{SHARED}/vendor/escape.js'))
            finally:outside.unlink()


if __name__=='__main__':unittest.main()
