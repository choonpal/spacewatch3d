import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil
from spacewatch_pipeline.florence import cached_inference,CODE
from spacewatch_pipeline.io import save,sha,record

class CacheTests(unittest.TestCase):
    def fixture(self,base):
        old=base/'old';out=base/'new';out.mkdir();data=old/'data';data.mkdir(parents=True)
        view=dict(key='one/0',sha256='image',source_sha256='source',frame=0,view=dict(yaw=0,pitch=0))
        save(data/'prepared.json',dict(scenes=[dict(views=[view])]))
        save(data/'inference_summary.json',dict(views=1));save(data/'result.json',dict(data_root=str(data)))
        weights=old/'weights';weights.write_bytes(b'pinned-model')
        save(old/'identity.json',dict(config=dict(discovery={'models':{'florence':'pinned'}}),code={
            'vendor/florence_only/'+name:sha(CODE/name) for name in ['infer.py','repair.py','florence_core.py','florence_image.py']}))
        save(old/'stages/one.infer/complete.json',dict(result_path=str(data/'result.json'),artifacts=[record(data/'inference_summary.json'),record(data/'prepared.json')]))
        save(old/'stages/one.prepare/complete.json',dict(artifacts=[record(weights)]))
        job=dict(capture={'id':'one'},config=dict(reuse_florence_run=str(old),discovery={'models':{'florence':'pinned'}}))
        return job,out,data/'prepared.json',weights
    def test_changed_weights_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            job,out,prep,weights=self.fixture(Path(td));weights.write_bytes(b'other-model')
            with self.assertRaises(ValueError):cached_inference(job,False,out,prep)
    def test_verified_cache_and_changed_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            job,out,prep,weights=self.fixture(Path(td))
            self.assertTrue(cached_inference(job,False,out,prep)['inference_reused'])
            changed=Path(td)/'changed.json';save(changed,dict(scenes=[dict(views=[])]))
            with self.assertRaises(ValueError):cached_inference(job,False,out,changed)

    def test_changed_visual_cache_helper_rejects_prediction_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            job,out,prep,weights=self.fixture(Path(td))
            altered=Path(td)/'altered_code';altered.mkdir()
            for name in ['infer.py','repair.py','florence_core.py','florence_image.py']:
                shutil.copyfile(CODE/name,altered/name)
            with (altered/'florence_image.py').open('a') as f:f.write('\n# changed helper\n')
            with patch('spacewatch_pipeline.florence.CODE',altered):
                with self.assertRaisesRegex(ValueError,'producing code changed'):
                    cached_inference(job,False,out,prep)

    def test_legacy_producer_without_helper_identity_is_rejected(self):
        from spacewatch_pipeline.io import read
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);job,out,prep,weights=self.fixture(base)
            path=base/'old/identity.json';identity=read(path)
            del identity['code']['vendor/florence_only/florence_image.py'];save(path,identity)
            with self.assertRaisesRegex(ValueError,'producing code changed'):
                cached_inference(job,False,out,prep)
