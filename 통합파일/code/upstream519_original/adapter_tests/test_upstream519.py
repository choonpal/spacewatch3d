"""Small source-contract checks; no geometry reconstruction or model execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from spacewatch_pipeline import upstream519 as m


class Tests(unittest.TestCase):
    def test_context_plan_is_sanitized_and_capture_bound(self):
        plan={'capture_id':'new_capture','viewpoints':[{'old_frame':999}],
              'provenance':{'config':{'camera_height_m':1.8}},
              'route_graph':{'grid_origin_xy_m':[1,2],'nodes':['old']},
              'coverage_report':{'surface_sampling':{'fallback_tiles':{'origin_xy_m':[3,4]}},'old_VQ':[999]}}
        context=m.context_fields(plan,'new_capture')
        self.assertNotIn('viewpoints',context)
        self.assertNotIn('nodes',context['route_graph'])
        self.assertNotIn('old_VQ',context['coverage_report'])
        with self.assertRaises(ValueError):m.context_fields(plan,'other_capture')

    def test_streamed_arrays_match_original_profile_digest(self):
        import hashlib
        arrays={'vertices':np.arange(18,dtype='f4').reshape(6,3),
                'faces':np.array([[0,1,2],[2,3,4]],dtype='u4'),
                'rgb':np.ones((6,3),dtype='u1'),'face_class_id':np.array([1,2],dtype='u2')}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mesh.npz'
            np.savez_compressed(path,**arrays)
            actual=m.npz_array_binding(path,arrays)
            for key,a in arrays.items():
                expected=hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
                self.assertEqual(actual[key]['sha256'],expected)

    def test_only_accepted_original_upstream_stages(self):
        self.assertEqual(m.ACTIONS,('inputs','infer','person_masks','pano','coordinates','reconstruct','mesh','semantic','scene'))
        for forbidden in ('plan','gs_train','render','priority','changes','filter'):
            self.assertNotIn(forbidden,m.ACTIONS)

    def test_person_binding_does_not_mask_umbrella(self):
        import runpy
        from PIL import Image
        worker=runpy.run_path(str(m.PRODUCERS/'offline_worker.py'),run_name='offline_boundary_test')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cid='test'
            (root/'source').mkdir();(root/'source/test.MP4').write_bytes(b'fixture-only')
            target=root/'inputs/test';(target/'frames').mkdir(parents=True)
            template=root/'code/v5/vendor/panovggt_pipeline_v2/configs/dorm2.json'
            template.parent.mkdir(parents=True);template.write_text(json.dumps({'input':{},'mask':{}}))
            pred=root/'predictions';pred.mkdir();rows=[]
            for i in range(4):
                image=target/f'frames/frame_{i:05d}.jpg';Image.new('RGB',(1036,518),'white').save(image)
                rows.append({'file':f'frames/{image.name}'})
                labels=np.zeros((512,1024),'u1');labels[100:110,100:110]=15;labels[100:110,500:510]=16
                np.savez_compressed(pred/f'{i:05d}.npz',labels=labels)
            (target/'manifest.json').write_text(json.dumps({'source_sha256':m.sha(root/'source/test.MP4'),'frames':rows}))
            vocabulary=root/'classes.json';vocabulary.write_text(json.dumps([{'id':1,'name':'floor'},{'id':2,'name':'ceiling'},{'id':15,'name':'person'},{'id':16,'name':'umbrella'}]))
            ns,binding=worker['raw_namespace'](m.PRODUCERS/'raw_input_producer/prepare_new.py',root,vocabulary)
            ns['masks'](cid,str(pred))
            mask=np.asarray(Image.open(target/'base_masks/frame_00000.png'))
            self.assertTrue(mask[106,106])
            self.assertFalse(mask[106,510])
            self.assertTrue(mask[500,510])
            self.assertEqual(len(binding['class_binding_only']),2)
            self.assertFalse(binding['mask_algorithm_parameters_changed'])

    def test_hash_reuse_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'binding.json';path.write_text('{}')
            ref=m.record(path)
            self.assertEqual(m.verify(ref),path)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):m.verify(ref)

    def test_fresh_model_is_not_implicitly_launched_by_inspection(self):
        with patch.object(m,'source_snapshot',return_value={'files':[]}),patch.object(m,'_profile_context',return_value=(Path(__file__),{})):
            with self.assertRaises(ValueError):m.preflight({'capture_id':'x','mode':'automatic_raw_video'})

    def test_cold_memory_guard_precedes_any_producer_execution(self):
        with patch.object(m,'memory_available_gib',return_value=6):
            with self.assertRaises(RuntimeError):m._run_fresh({'minimum_available_memory_gib':16},Path('/unused'),{},None)

    def test_prediction_receipt_is_accepted_by_native_consumer(self):
        from spacewatch_pipeline.capture519 import import_predictions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'input').mkdir(parents=True)
            pred=root/'pred';pred.mkdir()
            vocabulary=root/'vocabulary.json';vocabulary.write_text(json.dumps([{'id':0,'name':'wall'},{'id':1,'name':'floor'}]))
            np.savez_compressed(pred/'00000.npz',labels=np.zeros((2,4),'u1'),support_fraction=np.ones((2,4),'f2'),margin_fraction=np.ones((2,4),'f2'))
            (pred/'complete.json').write_text(json.dumps({'vocabulary_sha256':m.sha(vocabulary),'strict_load':True,'checkpoint_sha256':'fixture-checkpoint-only'}))
            (source/'input/manifest.json').write_text(json.dumps({'frames':[{'sha256':'fixture-source-rgb-hash','source_frame_index':0}]}))
            output=root/'receipts';output.mkdir()
            binding=m._captured_prediction(output,source,pred,vocabulary)
            imported=root/'native_import';imported.mkdir()
            import_predictions({'capture':{'id':'test','captured_prediction':binding},'output':str(imported),
                'dependencies':{'test.reconstruct':{'source_run':str(source)},'discovery':{'vocabulary':str(vocabulary)}}})
            result=json.loads((imported/'result.json').read_text())
            self.assertTrue(result['verified_prediction_import'])
            self.assertEqual(result['new_model_calls'],0)


if __name__=='__main__':unittest.main()
