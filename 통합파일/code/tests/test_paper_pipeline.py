import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from spacewatch_pipeline.paper_pipeline import adapt_stages,validate,resource_guard


def config():
    return dict(paper_integration=dict(enabled=True,schema='spacewatch.paper.20261110.v1',
        selection=dict(max_views=15,deadline_seconds=35.)),
        media=dict(mode='original_erp',candidate_interval_s=.25),
        runtime=dict(overlap_cpu_gpu=False,cpu_threads=1),
        captures=[dict(id='before',video='/video.mp4',importance_cache='/scores.json')])


class Contracts(unittest.TestCase):
    def test_valid(self):validate(config())
    def test_no_optional_quality_gate(self):
        c=config();c['media']['candidate_interval_s']=.5
        with self.assertRaises(ValueError):validate(c)
    def test_current_plan_before_rgb(self):
        c=config();c['captures'].append(dict(id='after'))
        rows=[dict(name='pair.rgb_inputs',action='rgb_inputs',dependencies=[]),
              dict(name='before.trajectory',action='trajectory',dependencies=['before.reconstruct'])]
        rows=adapt_stages(rows,c)
        self.assertIn('before.plan',rows[0]['dependencies'])
        self.assertIn('after.plan',rows[0]['dependencies'])
        self.assertIn('before.quality',rows[1]['dependencies'])
    def test_resource_hold(self):
        c=config()
        with patch('spacewatch_pipeline.paper_pipeline.memory_snapshot',return_value=dict(
                mem_available_bytes=1024,pswpout=0)),patch('spacewatch_pipeline.paper_pipeline.save'),\
                patch('spacewatch_pipeline.paper_pipeline.time.sleep'):
            with self.assertRaises(RuntimeError):resource_guard(c,'plan',Path('unused.json'))


if __name__=='__main__':unittest.main()
