import io
import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from http.client import HTTPConnection
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_jobs import TrajectoryJobs
from server import make_handler

class AutoTourTests(unittest.TestCase):
    def test_tour_export_round_trip_and_directory_boundary(self):
        with TemporaryDirectory() as directory, patch('server.ROOT', Path(directory)):
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],object(),export_dir=Path(directory)/'custom-tours'))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                tour={'version':1,'title':'../한글 투어','assets':[],'scenes':[]}
                conn=HTTPConnection('127.0.0.1',server.server_port)
                conn.request('POST','/api/tours',body=json.dumps(tour).encode(),headers={'Content-Type':'application/json'})
                response=conn.getresponse();self.assertEqual(response.status,201);saved=json.loads(response.read())
                export=Path(directory)/'custom-tours'/saved['fileName']
                self.assertEqual(export.parent,Path(directory)/'custom-tours')
                self.assertEqual(json.loads(export.read_text()),tour)
                conn.request('GET',saved['url']);response=conn.getresponse()
                self.assertEqual(response.status,200);self.assertIn('attachment',response.getheader('Content-Disposition'))
                self.assertEqual(json.loads(response.read()),tour)
                conn.request('GET','/exports/%2e%2e/server.py');response=conn.getresponse()
                self.assertEqual(response.status,404);response.read()
                conn.request('POST','/api/tours',body=b'{}');response=conn.getresponse()
                self.assertEqual(response.status,400);response.read();conn.close()
                self.assertEqual(len(list(export.parent.iterdir())),1)
            finally:
                server.shutdown();server.server_close()

    def test_upload_starts_analysis_and_video_reopens_by_stable_id(self):
        with TemporaryDirectory() as directory, patch('trajectory_jobs.subprocess.run') as run, patch.object(TrajectoryJobs,'inspect_video',return_value={'duration':12,'width':3840,'height':1920}):
            run.return_value.returncode=1
            jobs=TrajectoryJobs(directory);jobs.available=True
            with patch.object(jobs,'start',return_value={'state':'queued'}) as start:
                server=ThreadingHTTPServer(('127.0.0.1',0),make_handler([],[],jobs));threading.Thread(target=server.serve_forever,daemon=True).start()
                try:
                    conn=HTTPConnection('127.0.0.1',server.server_port)
                    conn.request('POST','/api/videos?name=test.mp4',body=b'abc123',headers={'Content-Type':'application/octet-stream'})
                    response=conn.getresponse();self.assertEqual(response.status,201);body=json.loads(response.read());key=body['key'];start.assert_called_once_with(key,force=False)
                    conn.request('GET',body['media']['url'],headers={'Range':'bytes=1-3'});response=conn.getresponse();self.assertEqual(response.status,206);self.assertEqual(response.read(),b'bc1')
                    conn.request('GET','/api/library');response=conn.getresponse();self.assertEqual(json.loads(response.read())['media'][0]['trajectoryKey'],key)
                    conn.close()
                finally:server.shutdown();server.server_close()
            restored=TrajectoryJobs(directory);self.assertEqual(restored.media_catalog()[0]['trajectoryKey'],key);self.assertEqual(restored.sources[key]['duration'],12)

    def test_invalid_video_is_rejected_before_registration(self):
        with TemporaryDirectory() as directory, patch('trajectory_jobs.subprocess.run') as run:
            run.return_value.returncode=1;jobs=TrajectoryJobs(directory)
            with patch.object(jobs,'inspect_video',side_effect=ValueError('2:1 video required')):
                with self.assertRaisesRegex(ValueError,'2:1'):jobs.register_upload(io.BytesIO(b'invalid'),7,'bad.mp4')
            self.assertEqual(jobs.sources,{});self.assertFalse(list(jobs.data.glob('*.part')))

if __name__=='__main__':unittest.main()
