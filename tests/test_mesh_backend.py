import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spacewatch3d.contracts import ContractError, validate_artifact
from spacewatch3d.task_api import TaskRequest
from spacewatch3d.tasks.task_5a_change_detection.mesh_backend import analyze
from spacewatch3d.tasks.task_5a_change_detection.pipeline import run


class MeshSystemTests(unittest.TestCase):
    def test_manifest_validation_does_not_import_or_run_gpu_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'mesh.npz').write_bytes(b'explicit manifest-only fixture')
            (root/'rgb.jpg').write_bytes(b'explicit manifest-only fixture')
            side=dict(id='capture',mesh='mesh.npz',frames=[dict(id='0',image='rgb.jpg',world_from_camera=[])])
            manifest=root/'pair.local.json'
            manifest.write_text(json.dumps(dict(schema='mesh-simsac-input-v1',before=side,after=side)))
            result=json.loads(analyze(manifest,root/'checked',validate_only=True).read_text())
            self.assertFalse(result['inference_performed'])
            self.assertFalse(result['geometry_validated'])
            self.assertIn('pair.local.json',result['source_files'])

    def test_lineage_dispatch_links_local_candidates_without_fabricating_contract_events(self):
        repo=Path(__file__).resolve().parents[1]
        inputs={p:repo/'examples/fixtures'/p/'objects.json' for p in ['before','after']}
        lineage={p:json.loads(f.read_text()) for p,f in inputs.items()}
        with tempfile.TemporaryDirectory(dir=repo,prefix='.test-mesh-backend-') as temp:
            root=Path(temp);manifest=root/'pair.json'
            doc=dict(scene_id=lineage['after']['scene_id'],**{p:dict(id=d['capture_id']) for p,d in lineage.items()})
            manifest.write_text(json.dumps(doc))
            def fake_gpu(pair,output,runtime):
                result=output/'detection';(result/'results').mkdir(parents=True)
                (result/'results/events.json').write_text(json.dumps(dict(events=[{'id':'EV-001'}])))
                (result/'registration.json').write_text(json.dumps(dict(common_from_world=[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]*2)))
                (result/'viewer-data.json').write_text('{}')
                return result/'results/events.json'
            config=dict(backend='mesh_rgb_simsac',pair_manifest=str(manifest))
            with patch('spacewatch3d.tasks.task_5a_change_detection.mesh_backend.analyze',side_effect=fake_gpu) as execute:
                result=validate_artifact(run(TaskRequest(inputs,root/'output',config)),check_files=True)
                self.assertEqual(result['parameters']['candidate_count'],1)
                self.assertEqual(result['changes'],[])
                self.assertEqual(execute.call_count,1)
                doc['after']['id']='wrong-capture';manifest.write_text(json.dumps(doc))
                with self.assertRaises(ContractError):
                    run(TaskRequest(inputs,root/'other',config))
                self.assertEqual(execute.call_count,1)


if __name__ == '__main__':
    unittest.main()
