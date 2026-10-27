import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_segmentation_review import ReviewAssets, build, camera_from_world, digest


class SegmentationInspection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.folder = Path(self.temp.name)
        self.inputs = self.folder/'inputs'; self.inputs.mkdir()
        Image.new('RGB', (64, 32), '#999999').save(self.inputs/'rgb.jpg')
        np.savez(self.inputs/'mesh.npz', vertices=np.array([[0.,0,2],[1,0,2],[0,1,2],[2,0,2]]),
                 faces=np.array([[0,1,2],[1,3,2]]), face_instance_id=np.array([1,2]))
        (self.inputs/'objects.json').write_text(json.dumps({'instance_records':[
            {'instance_id':1,'label':'chair'},{'instance_id':2,'label':'person'}]}))
        cls=np.zeros((32,64),int); cls[10:20,10:20]=1
        np.savez(self.inputs/'mask.npz', class_id=cls, instance_id=np.zeros_like(cls), person=cls==2)
        spec=dict(id='capture',mesh='mesh.npz',objects='objects.json',projection='erp',
                  world_from_mesh=[[1,0,0,3],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  frames=[dict(id='0',image='rgb.jpg',segmentation='mask.npz',
                               segmentation_rgb_sha256=digest(self.inputs/'rgb.jpg'),world_from_camera=np.eye(4).tolist())])
        self.doc=dict(before=spec,after=spec); self.save()

    def save(self):
        (self.inputs/'manifest.json').write_text(json.dumps(self.doc))

    def tearDown(self):
        self.temp.cleanup()

    def test_projection_uses_original_calibration_and_keeps_source_world_translation(self):
        output=self.folder/'out'; data=build(self.inputs,output); scene=data['scenes'][0]
        positions=np.frombuffer((output/'mesh-0.bin').read_bytes(),'<f4',count=scene['vertices']*3).reshape(-1,3)
        np.testing.assert_allclose(positions[0],[3,0,2])
        camera=np.array(camera_from_world(np.eye(4)))
        np.testing.assert_allclose(camera @ [3,1,2,1],[3,-1,-2,1])
        self.assertFalse(data['inference_rerun'])
        self.assertEqual(scene['labelled_faces'],1)
        self.assertEqual([o['label'] for o in scene['objects']],['chair'])

    def test_unmapped_original_mask_stays_visible_without_inventing_mesh_labels(self):
        output=self.folder/'out'; build(self.inputs,output)
        with Image.open(output/'mask-0-00000.png') as image:
            pixels=np.asarray(image)
        visible=pixels[pixels[...,3]>0]
        self.assertGreater(len(visible),0)
        np.testing.assert_array_equal(visible[0,:3],[255,110,211])

    def test_wrong_rgb_mask_binding_is_rejected(self):
        self.doc['before']['frames'][0]['segmentation_rgb_sha256']='bad';self.save()
        with self.assertRaisesRegex(ValueError,'RGB'):
            build(self.inputs,self.folder/'out')

    def test_assets_are_cached_and_cannot_expose_other_files(self):
        jobs=Mock();jobs.folder.return_value=self.folder;jobs.status.return_value={'state':'complete'}
        assets=ReviewAssets(jobs)
        first=assets.asset('job','manifest.json');mtime=first.stat().st_mtime_ns
        self.assertEqual(assets.asset('job','manifest.json'),first)
        self.assertEqual(first.stat().st_mtime_ns,mtime)
        with self.assertRaises(KeyError):assets.asset('job','../inputs/manifest.json')
        self.doc['before']['mesh']='../outside.npz';self.save()
        with self.assertRaises(ValueError):assets.asset('job','manifest.json')


if __name__ == '__main__':
    unittest.main()
