import hashlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
from tools.gaussian_change.instance_observations import rebuild_tracks, build_instances, packet
from tools.gaussian_change.lifecycle import objects_from_mesh


def two_connected_boxes():
    corners=np.array([[x,y,z] for x in [-.5,.5] for y in [0,1] for z in [-.5,.5]],float)
    triangles=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    vertices=[];faces=[];groups=[];index={}
    def vertex(point):
        key=tuple(np.round(point,8))
        if key not in index:index[key]=len(vertices);vertices.append(point)
        return index[key]
    for shift in [0.,1.]:
        raw=corners+np.array([shift,0,0]);samples=[]
        for ids in triangles:
            a,b,c=raw[ids];ab=(a+b)/2;ac=(a+c)/2;bc=(b+c)/2
            for tri in [[a,ab,ac],[ab,b,bc],[ac,bc,c],[ab,bc,ac]]:
                faces.append([vertex(p) for p in tri]);samples.append(np.mean(tri,axis=0))
        groups.append(np.array(samples))
    return np.asarray(vertices),np.asarray(faces),groups


class InstanceConnectionTests(unittest.TestCase):
    def test_masks_split_two_objects_that_semantic_connectivity_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,f,samples=two_connected_boxes();path=Path(tmp)/'mesh.npz'
            np.savez(path,p=p,faces=f,face_semantic=np.full(len(f),19),face_observations=np.full(len(f),3),rgb=np.full(p.shape,80))
            observations=[]
            for obj,points in enumerate(samples):
                for frame in [0,5,10]:
                    observations.append(dict(id=f'{obj}-{frame}',frame=frame,class_name='chair',class_id=56,confidence=.9,raw_points=points.tolist()))
            source=dict(schema='mesh-instance-observations-v1',mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),coordinate_frame='mesh',observations=observations,cameras=[])
            side={'mesh':path,'world_from_mesh':np.eye(4),'_instance_segmentation':source}
            registration={'height_u':2.,'centers':[[0,0,0]],'regions':[{'supported':True}],'global_after_to_before':np.eye(4).tolist()}
            old,_=objects_from_mesh(side,0,registration);new,audit=build_instances(side,0,registration)
            self.assertEqual(len(old),1)
            self.assertEqual(len(new),2)
            self.assertTrue(audit['used'])
            self.assertTrue(all(o['source_observation_count']==3 for o in new))
            self.assertFalse(set(new[0]['original_face_ids'])&set(new[1]['original_face_ids']))
            source['mesh_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'exact input mesh'):packet(side)

    def test_old_cross_capture_labels_do_not_affect_unary_tracking(self):
        points=np.array([[x,y,z] for x in [0,.2,.4] for y in [0,.2,.4] for z in [0,.2,.4]])
        observations=[dict(id=str(i),frame=i,class_name='chair',raw_points=points.tolist(),zone='old-room',common_center=[99,99,99],event='moved') for i in [0,1]]
        first=rebuild_tracks(observations,2.)
        for o in observations:o.update(zone='other',event='removed',common_center=[-99,-99,-99])
        second=rebuild_tracks(observations,2.)
        self.assertEqual(len(first),len(second));np.testing.assert_allclose(first[0]['center'],second[0]['center'])


if __name__=='__main__':unittest.main()
