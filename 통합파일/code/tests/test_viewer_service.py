import os
import tempfile
import time
import unittest
from pathlib import Path
from spacewatch_pipeline import viewer_service,engine
from spacewatch_pipeline.io import save

class ViewerServiceTests(unittest.TestCase):
    def test_detached_server_health_reuse_and_owned_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);public=run/'public';public.mkdir()
            save(public/'manifest.json',dict(bundle_id='test-service',assets=[],captures=[],changes=None))
            first=viewer_service.launch(run,public)
            try:
                self.assertTrue(viewer_service.health(first['url'],'test-service'))
                self.assertFalse(viewer_service.health(first['url'],'wrong-bundle'))
                self.assertEqual(viewer_service.launch(run,public)['pid'],first['pid'])
                self.assertNotEqual(os.getsid(first['pid']),os.getsid(0))
            finally:
                self.assertTrue(viewer_service.stop(run)['termination_requested'])
                for _ in range(30):
                    if not viewer_service.health(first['url'],'test-service'):break
                    time.sleep(.05)
            self.assertFalse(viewer_service.health(first['url'],'test-service'))
    def test_viewer_runs_last_without_gpu_and_can_be_disabled(self):
        config=dict(captures=[dict(id='one',source_run='existing')],segmentation=dict(source='provided'))
        stages=engine.stages(config)
        self.assertEqual(stages[-1]['name'],'viewer');self.assertFalse(stages[-1]['gpu'])
        self.assertIn('one.render',stages[-1]['dependencies']);self.assertIn('publish',stages[-1]['dependencies'])
        config['viewer']={'enabled':False};self.assertEqual(engine.stages(config)[-1]['name'],'publish')
