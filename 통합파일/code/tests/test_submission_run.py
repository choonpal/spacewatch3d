import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spacewatch_pipeline.submission_run import run


class WholePipelineBinding(unittest.TestCase):
    def fixture(self,base):
        (base/'coords.json').write_text('{}')
        (base/'classes.json').write_text('[]')
        config=dict(captures=[dict(id='A',source_run='OLD',planning_scene='OLD')],
                    paper_integration={},segmentation={})
        (base/'template.json').write_text(json.dumps(config))
        (base/'prepare.json').write_text('{}')
        request=dict(schema='spacewatch.integrated_fresh.v1',pipeline_config='template.json',
                     captures=[dict(id='A',prepare_request='prepare.json')])
        path=base/'request.json';path.write_text(json.dumps(request))
        ready=dict(capture_id='A',source_run=str(base/'NEW'),coordinates=str(base/'coords.json'),
            semantic_arrays=str(base/'NEW_ARRAYS.npz'),vocabulary=str(base/'classes.json'),
            planning_profile=dict(path=str(base/'PROFILE'),sha256='a'*64),
            captured_prediction=dict(predictions=str(base/'PREDICTIONS')))
        return path,ready

    def test_fresh_outputs_replace_old_inputs_before_native_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);request,ready=self.fixture(base)
            with patch('spacewatch_pipeline.upstream519.run',return_value=ready),\
                 patch('spacewatch_pipeline.upstream519.source_snapshot'),\
                 patch('spacewatch_pipeline.engine.validate'),\
                 patch('spacewatch_pipeline.engine.execute') as execute:
                run(request,base/'output')
            bound=json.loads((base/'output/pipeline-config.json').read_text())
            capture=bound['captures'][0]
            self.assertEqual(capture['source_run'],ready['source_run'])
            self.assertNotIn('planning_scene',capture)
            self.assertEqual(capture['captured_prediction'],ready['captured_prediction'])
            self.assertEqual(bound['paper_integration']['upstream_mode'],'fresh_original_producer_in_this_submission')
            execute.assert_called_once_with(base/'output/pipeline-config.json',base/'output/pipeline')

    def test_missing_own_frame_predictions_fails_before_native_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);request,ready=self.fixture(base);ready.pop('captured_prediction')
            with patch('spacewatch_pipeline.upstream519.run',return_value=ready),\
                 patch('spacewatch_pipeline.upstream519.source_snapshot'),\
                 patch('spacewatch_pipeline.engine.execute') as execute:
                with self.assertRaises(ValueError):run(request,base/'output')
            execute.assert_not_called()
            self.assertEqual(json.loads((base/'output/submission-state.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
