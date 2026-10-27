import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from tools.mesh_simsac.capturer import semantic_capturer
from tools.mesh_simsac.aux_inputs import box_face_instances
from tools.mesh_simsac.rgb_people import candidates


class BoxPersonInputs(unittest.TestCase):
    def test_photographer_does_not_remove_separate_bystander(self):
        mask = np.zeros((100, 200), bool)
        mask[70:, :8] = True
        mask[70:, -8:] = True
        mask[40:65, 80:90] = True
        excluded, audit = semantic_capturer(mask)
        self.assertTrue(excluded[80, 0])
        self.assertTrue(excluded[80, -1])
        self.assertFalse(excluded[50, 85])
        self.assertEqual(audit['retained_person_pixels'], 250)

    def test_box_requires_independent_mask_supported_frames(self):
        faces = np.array([[0,1,2],[1,2,3],[5,6,7]])
        ids = np.array([3,3,4])
        classes = np.array([0,0,0])
        hit = np.array([[0,0,1,1,2,2],[0,0,1,1,2,2]], 'u4')
        box = hit < 2
        unchanged = box_face_instances(faces, ids, classes, [(hit, box)], 17)
        np.testing.assert_array_equal(unchanged[0], ids)
        result, labels, records = box_face_instances(faces, ids, classes, [(hit,box),(hit,box)], 17)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['face_count'], 2)
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[2], 4)
        np.testing.assert_array_equal(labels, [17,17,0])

    def test_background_face_with_small_mask_coverage_is_not_box(self):
        faces = np.array([[0,1,2],[1,2,3]])
        hit = np.zeros((10,10), 'u4')
        mask = np.zeros_like(hit, bool); mask[:2,:2] = True
        _, _, records = box_face_instances(faces, np.zeros(2,int), np.zeros(2,int), [(hit,mask)]*2, 17)
        self.assertEqual(records, [])

    def test_mesh_missing_person_requires_repeatable_best_registered_rgb_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'views').mkdir();(root/'results/pairs').mkdir(parents=True)
            y,x=np.mgrid[:10,:10]
            points=np.stack([x,y,np.ones_like(x)],-1).astype(float)
            cls=np.zeros((10,10),int);cls[1:7,2:8]=1
            rows=[];pairs=[]
            for i in range(2):
                for epoch in [0,1]:
                    vid=f'E{epoch}-V{i}'
                    rows.append(dict(id=vid,epoch=epoch,frame_id=str(i),timestamp_s=i*3,
                        person_mask_policy='capturer_only',segmentation_labels=['person']))
                    np.savez(root/'views'/(vid+'.npz'),points=points,pose=np.eye(4),K=np.eye(3),
                        depth=np.ones((10,10)),person=np.zeros((10,10),bool),mesh_hit=np.ones((10,10),bool),
                        segmentation_class=cls if epoch else np.zeros_like(cls),segmentation_instance=np.zeros_like(cls))
                query=f'E1-V{i}';ref=f'E0-V{i}'
                pairs.append(dict(query=query,refs=[dict(id=ref)]))
                np.savez(root/'results/pairs'/(query+'--'+ref+'.npz'),probability=np.where(cls==1,.9,.1))
            (root/'pairs.json').write_text(json.dumps(pairs))
            prepared=dict(views=rows,settings=dict(probability_threshold=.5,min_object_pixels=16,min_frames=2))
            groups=candidates(root,prepared)
            self.assertEqual(len(groups),1)
            self.assertEqual({r['frame_id'] for r in groups[0]},{'0','1'})
            self.assertEqual(groups[0][0]['label'],'person')
            self.assertEqual(groups[0][0]['mesh_instance_ids'],[])
            np.savez(root/'results/pairs/E1-V1--E0-V1.npz',probability=np.zeros((10,10)))
            self.assertEqual(candidates(root,prepared),[])


if __name__ == '__main__':
    unittest.main()
