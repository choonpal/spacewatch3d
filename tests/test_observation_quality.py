import unittest
import numpy as np

from tools.mesh_simsac.core import rays
from tools.mesh_simsac.observation_quality import object_pair_quality,reference_consensus
from tools.mesh_simsac.run import qualified_evidence


def camera(z=0):
    shape=(80,80);K=np.array([[40.,0,39.5],[0,40.,39.5],[0,0,1.]])
    points=rays(K,shape)*2
    pose=np.eye(4);pose[2,3]=z
    ids=np.zeros(shape,int);ids[20:60,20:60]=1
    return dict(valid=np.ones(shape,bool),person=np.zeros(shape,bool),instance=ids,
        points=points,pose=pose,K=K,depth=np.full(shape,2.-z))


class ObservationQuality(unittest.TestCase):
    def test_reference_tie_is_not_a_positive_vote(self):
        support=np.array([0,1,2,2,3,3]);positive=np.array([0,1,1,2,1,2])
        np.testing.assert_array_equal(reference_consensus(support,positive),[False,True,False,True,False,True])

    def test_same_size_object_can_be_compared_without_counterpart_label(self):
        query,reference=camera(),camera();reference['instance'][:]=0
        supported,audit=object_pair_quality(query,reference)
        self.assertTrue(supported[30,30]);self.assertTrue(audit[0]['accepted'])
        self.assertAlmostEqual(audit[0]['reference_scale_ratio'],1.)

    def test_distant_small_reference_cannot_vote_on_large_query_object(self):
        supported,audit=object_pair_quality(camera(),camera(-4.))
        self.assertFalse(supported[30,30])
        self.assertIn('object_scale_mismatch',audit[0]['reasons'])

    def test_rejected_object_observation_becomes_unknown_not_unchanged(self):
        query=camera();reference=camera(-4.);shape=query['valid'].shape
        raw=query['instance']>0;zero=np.zeros(shape,bool)
        masks=dict(raw=raw,candidate=raw,observable=raw,excluded_person=zero,
            unsupported_geometry=zero,occluded=zero,unobserved_counterpart=zero)
        result,_=qualified_evidence(query,reference,masks)
        self.assertFalse(result['candidate'].any());self.assertFalse(result['observable'].any())
        np.testing.assert_array_equal(result['unobserved_counterpart'],raw)
        np.testing.assert_array_equal(sum(result[k].astype(int) for k in
            ['candidate','excluded_person','unsupported_geometry','occluded','unobserved_counterpart']),raw.astype(int))
        np.testing.assert_array_equal(masks['candidate'],raw)


if __name__=='__main__':unittest.main()
