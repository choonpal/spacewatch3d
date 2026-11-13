import unittest,numpy as np
from core import parent_pixels,periodic_components,min_segment_distance,proposals

def rectangle(x0,x1,y0=0,y1=1,divisions=1):
 v=np.array([[x,y,0.] for y in np.linspace(y0,y1,divisions+1) for x in np.linspace(x0,x1,divisions+1)],float);f=[]
 for j in range(divisions):
  for i in range(divisions):
   a=j*(divisions+1)+i;b=a+1;c=a+divisions+1;d=c+1;f.extend([[a,b,d],[a,d,c]])
 return v,np.array(f,int)
def combine(meshes):
 vertices=[];lookup={};faces=[]
 for v,f in meshes:
  mapping=[]
  for point in v:
   key=tuple(np.round(point,12))
   if key not in lookup:lookup[key]=len(vertices);vertices.append(point)
   mapping.append(lookup[key])
  faces.extend(np.array(mapping)[f])
 return np.array(vertices),np.array(faces)
def group(v,f,signatures=None,positive=None):
 n=len(f);cc=np.ones((3,n),np.uint32) if signatures is None else signatures;pos=np.full(n,3,np.uint32) if positive is None else positive
 return proposals(v,f,np.ones(n,bool),cc,pos,.12)
class CoreTests(unittest.TestCase):
 def test_legacy_record_parent_pixels_gate(self):
  from unittest.mock import Mock
  from server_full import frame_record
  old = Mock()
  face=np.array([0,0],np.uint32);labels=np.array([3,0],np.uint8);weights=np.array([.8,.4],np.float32);eligible=np.ones(2,bool)
  old.frame_votes.return_value = np.array([(0, 3, .4)], dtype=[('index', 'i4'), ('label', 'u1'), ('weight', 'f4')])
  self.assertEqual(len(frame_record(face,labels,weights,eligible,1,old,False)),1);self.assertEqual(len(frame_record(face,labels,weights,eligible,1,old,True)),0)
 def test_child_margin_removed_not_background_margin(self):
  s=np.zeros((17,1,1),np.float32);s[3]=.40;s[4]=.39;s[0]=.21;labels=np.array([[3]],np.uint8);o=parent_pixels(s,labels,'hard_union');self.assertTrue(o['raw'][0,0]);self.assertTrue(o['parent_eligible'][0,0]);self.assertAlmostEqual(float(o['margin'][0,0]),.19,places=3)
 def test_sum_group_is_separate_expanding_arm(self):
  s=np.zeros((17,1,1),np.float32);s[0]=.42;s[3]=.35;s[4]=.23;p=np.array([[0]],np.uint8);self.assertFalse(parent_pixels(s,p,'hard_union')['raw'][0,0]);self.assertTrue(parent_pixels(s,p,'score_sum')['raw'][0,0])
 def test_periodic_erp(self):
  a=np.zeros((3,6),bool);a[1,0]=a[1,-1]=True;c=periodic_components(a);self.assertGreater(c[1,0],0);self.assertEqual(c[1,0],c[1,-1]);self.assertEqual(periodic_components(np.ones((2,2),bool)).max(),1)
 def test_exact_segment_distance_midpoints(self):
  a=np.array([[[0,0,0],[2,0,0]]],float);b=np.array([[[1,-1,.04],[1,1,.04]]],float);d,_,_=min_segment_distance(a,b);self.assertAlmostEqual(d,.04)
 def test_fragmented_object_supported_gap(self):
  v,f=combine([rectangle(0,1),rectangle(1.04,2.04)]);o=group(v,f);self.assertEqual(int(o['ids'].max()),1);self.assertEqual(o['merged_component_links'],1);self.assertEqual(int((o['ids']>0).sum()),len(f))
 def test_close_objects_different_view_components(self):
  v,f=combine([rectangle(0,1),rectangle(1.04,2.04)]);cc=np.tile(np.r_[np.ones(2),np.full(2,2)],(3,1)).astype('u4');o=group(v,f,cc);self.assertEqual(int(o['ids'].max()),2);self.assertFalse(any(x['merged'] for x in o['relations']))
 def test_wrong_mesh_bridge_cut_by_observations(self):
  v,f=combine([rectangle(0,1),rectangle(1,2)]);cc=np.tile(np.r_[np.ones(2),np.full(2,2)],(3,1)).astype('u4');o=group(v,f,cc);self.assertEqual(int(o['ids'].max()),2);self.assertTrue(any(x['reason']=='multiview_separation' for x in o['cut_edges']))
 def test_no_evidence_does_not_force_gap_merge(self):
  v,f=combine([rectangle(0,1),rectangle(1.04,2.04)]);cc=np.zeros((3,4),np.uint32);cc[0,:]=1;o=group(v,f,cc);self.assertEqual(int(o['ids'].max()),2);self.assertEqual(o['merged_component_links'],0)
 def test_single_frame_uncertain_and_preserved(self):
  v,f=rectangle(0,1);before=v.copy();o=group(v,f,positive=np.ones(len(f),np.uint32));self.assertFalse(o['ids'].any());self.assertTrue(o['uncertain'].all());self.assertEqual(sum(x['faces'] for x in o['records']),len(f));np.testing.assert_array_equal(v,before)
 def test_tiny_region_not_deleted_or_confirmed(self):
  v,f=rectangle(0,.01,0,.01);o=group(v,f);self.assertFalse(o['ids'].any());self.assertEqual(sum(x['faces'] for x in o['records']),2);self.assertEqual(o['records'][0]['identity_status'],'uncertain')
 def test_density_invariance_of_geometry_grouping(self):
  areas=[]
  for divisions in [1,4]:
   v,f=combine([rectangle(0,1,divisions=divisions),rectangle(1.04,2.04,divisions=divisions)]);o=group(v,f);self.assertEqual(int(o['ids'].max()),1);areas.append(sum(x['area'] for x in o['records']))
  self.assertAlmostEqual(*areas)
 def test_reproducible_and_face_order_independent_partition(self):
  v,f=combine([rectangle(0,1),rectangle(1.04,2.04)]);a=group(v,f);b=group(v,f);np.testing.assert_array_equal(a['ids'],b['ids']);order=np.array([3,1,0,2]);c=group(v,f[order]);self.assertEqual(len(np.unique(c['ids'])),1);self.assertEqual(int(c['ids'].max()),1)
class NonplanarPolicyTests(unittest.TestCase):
 def test_nonplanar_multiview_retained_only_as_provisional(self):
  # Smooth connected half cylinder: local normals agree but global normal covariance is nonplanar.
  theta=np.linspace(0,np.pi,9);v=np.array([[np.cos(t),np.sin(t),z] for t in theta for z in [0,1]],float);f=[]
  for i in range(8):f.extend([[2*i,2*i+2,2*i+1],[2*i+1,2*i+2,2*i+3]])
  f=np.array(f);cc=np.ones((2,len(f)),np.uint32);mask=np.ones(len(f),bool);count=np.full(len(f),2)
  old=proposals(v,f,mask,cc,count,.1);new=proposals(v,f,mask,cc,count,.1,nonplanar_policy='retain_provisional')
  self.assertEqual(int(old['ids'].max()),0);self.assertEqual(int(new['ids'].max()),1)
  self.assertTrue(new['records'][0]['shape_uncertainty']);self.assertFalse(new['records'][0]['physical_identity_verified'])
  self.assertEqual(len(new['records'][0]['source_face_ids']),len(f))
 def test_nonplanar_without_multiview_not_promoted(self):
  theta=np.linspace(0,np.pi,9);v=np.array([[np.cos(t),np.sin(t),z] for t in theta for z in [0,1]],float);f=np.array([q for i in range(8) for q in [[2*i,2*i+2,2*i+1],[2*i+1,2*i+2,2*i+3]]]);cc=np.ones((1,len(f)),np.uint32)
  r=proposals(v,f,np.ones(len(f),bool),cc,np.full(len(f),2),.1,nonplanar_policy='retain_provisional');self.assertEqual(int(r['ids'].max()),0)

if __name__=='__main__':unittest.main(verbosity=2)
