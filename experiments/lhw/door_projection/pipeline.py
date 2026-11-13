"""Standalone opt-in CPU prototype. server_full or explicitly limited local_samples."""
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import argparse,base64,hashlib,importlib.util,io,json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('_parent_prototype_core',HERE/'core.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
FRAMES=[10,12,15,17,20]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(v,f):
 h=hashlib.sha256()
 for a,dt in [(v,'<f8'),(f,'<i8')]:
  a=np.ascontiguousarray(a,dtype=dt);h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def mask_cc_sample(cc,pixels,shape):
 h,w=shape;x=pixels%w;y=pixels//w;return cc[np.minimum(((y+.5)*cc.shape[0]/h).astype(int),cc.shape[0]-1),np.minimum(((x+.5)*cc.shape[1]/w).astype(int),cc.shape[1]-1)]
def roi_faces(pixels,faces,box,width):
 a,b,c,d=box;mask=(pixels%width>=a)&(pixels%width<c)&(pixels//width>=b)&(pixels//width<d);return np.unique(faces[mask])
def object_metrics(mask,proposal,reference,area):
 ids=np.array(sorted(reference),int);a=area[ids];assigned=proposal[ids];positive=np.unique(assigned[assigned>0]);supports={int(i):float(a[assigned==i].sum()) for i in positive};dominant=max(supports,key=supports.get) if supports else 0
 return dict(reference_faces=len(ids),reference_area=float(a.sum()),parent_faces=int(mask[ids].sum()),parent_area=float(area[ids[mask[ids]]].sum()),assigned_faces=int((assigned>0).sum()),assigned_area=float(a[assigned>0].sum()),unassigned_faces=int((assigned==0).sum()),unassigned_area=float(a[assigned==0].sum()),proposal_ids=list(map(int,positive)),proposal_area_within_reference=supports,dominant_proposal_id=int(dominant),dominant_area=float(supports.get(dominant,0)),dominant_fraction_of_reference_area=float(supports.get(dominant,0)/max(a.sum(),1e-30)))
def evaluate(v,f,labels,old_ids,variants,references,scope):
 _,_,area=C.face_geometry(v,f);out=dict(scope=scope,units='mesh units and mesh units squared; metric scale unverified',full_object_GT=False,reference_policy='faces touched by manually chosen diagnostic image patches; partial and projection-dependent',variants={},references={})
 for name,ids in references.items():out['references'][name]=dict(faces=len(ids),area=float(area[list(ids)].sum()))
 for name,a in variants.items():
  mask=a['accepted'];pid=a['proposals']['ids'];o=dict(parent_faces=int(mask.sum()),parent_area=float(area[mask].sum()),proposal_faces=int((pid>0).sum()),proposal_area=float(area[pid>0].sum()),proposals=int(pid.max(initial=0)),unconfirmed_parent_faces=int((mask&(pid==0)).sum()),unconfirmed_parent_area=float(area[mask&(pid==0)].sum()),reference_objects={},source_semantic_of_assigned={},negative_patch_assigned={})
  for obj in ['hinged_door','elevator']:o['reference_objects'][obj]=object_metrics(mask,pid,references[obj],area)
  o['shared_proposal_ids_between_two_reference_objects']=sorted(set(o['reference_objects']['hinged_door']['proposal_ids'])&set(o['reference_objects']['elevator']['proposal_ids']))
  for cid in np.unique(labels[pid>0]):
   sel=(pid>0)&(labels==cid);o['source_semantic_of_assigned'][int(cid)]=dict(faces=int(sel.sum()),area=float(area[sel].sum()))
  for patch in ['wall_patch','floor_patch']:
   ids=np.array(sorted(references[patch]),int);sel=pid[ids]>0;o['negative_patch_assigned'][patch]=dict(faces=int(sel.sum()),area=float(area[ids[sel]].sum()),reference_faces=len(ids),reference_area=float(area[ids].sum()))
  out['variants'][name]=o
 base=np.isin(labels,[3,4]);out['source_233frame_parent_context']={obj:object_metrics(base,old_ids,references[obj],area) for obj in ['hinged_door','elevator']}
 return out

def local_samples(out):
 meta=read(ROOT/'review_projection/evidence/projection_manifest.json');mesh_path=ROOT/'review_instance_regression/dorm2_r1/evidence_v2.npz';mesh=np.load(mesh_path);v=mesh['vertices'];f=mesh['faces'];labels=mesh['face_class_id'];assert digest(v,f)==meta['mesh_digest'];n=len(f);gap=meta['thresholds']['depth_tolerance_mesh_units'];roi=read(HERE/'review_rois.json')
 with zipfile.ZipFile(ROOT.parent/'spacewatch_pipeline_v5_1_9_server_snapshot_20261027_133626.zip') as z:roles=np.load(io.BytesIO(z.read('spacewatch_pipeline_v5_1_9_server_snapshot/profiles/dorm2_statistical_scored/face_roles.npz')));old=roles['target_region_index'];assert np.array_equal(roles['source_class_id'],labels)
 positive=np.zeros((5,n),bool);fine=np.zeros((5,n),bool);negative=np.zeros((5,n),bool);cc_face=np.zeros((5,n),np.uint32);refs={k:set() for k in ['hinged_door','elevator','wall_patch','floor_patch']};frame_rows=[]
 (out/'frames').mkdir()
 for j,index in enumerate(FRAMES):
  path=ROOT/'review_projection/evidence/frames'/f'{index:05d}.npz';a=np.load(path);pix=a['sample_pixels'];faces=a['sample_faces'];lab=a['sample_labels'];good=np.isin(lab,[3,4]);qualified=good&(a['sample_states']==4);positive[j,np.unique(faces[good])]=True;fine[j,np.unique(faces[qualified])]=True;negative[j,np.unique(faces[~good])]=True
  image_path=ROOT/'review_ooops_panorama/evidence/original'/f'{index:05d}.jpg';mask_path=ROOT/'review_ooops_panorama/evidence/ids'/f'{index:05d}.png';ids=np.asarray(Image.open(mask_path));raw=np.isin(ids,[3,4]);cc=C.periodic_components(raw);pointcc=mask_cc_sample(cc,pix,(518,1036));cc_face[j]=C.dominant_face_component(faces[good],pointcc[good],n,minimum=1)
  for group in ['positive','negative']:
   for name,box in roi[group][str(index)].items():refs[name].update(map(int,roi_faces(pix,faces,box,1036)))
  # Preserve actual sample positions and mapping. Original arrays stay read-only.
  np.savez_compressed(out/'frames'/f'{index:05d}.npz',**{k:a[k] for k in a.files},raw_parent_sample=good,legacy_qualified_parent_sample=qualified,raw_parent_face_ids=np.flatnonzero(positive[j]).astype('u4'))
  Image.fromarray((raw*255).astype('u1')).save(out/'frames'/f'{index:05d}_raw_parent.png')
  frame_rows.append(dict(frame=index,source_shape=[518,1036],sample_count=len(pix),raw_parent_samples=int(good.sum()),legacy_qualified_parent_samples=int(qualified.sum()),parent_quality_status='unknown_without_full_scores; fine-qualified samples are a known lower bound',source_image=str(image_path.resolve()),source_mask=str(mask_path.resolve()),source_sample_sha256=sha(path),source_image_sha256=sha(image_path),source_mask_sha256=sha(mask_path)))
 # Evaluation ambiguities never affect clustering.
 overlap=set.intersection(refs['hinged_door'],refs['elevator']);negative_union=refs['wall_patch']|refs['floor_patch'];ref_conflicts=(refs['hinged_door']|refs['elevator'])&negative_union
 for name in refs:refs[name]-=overlap|ref_conflicts
 raw_count=positive.sum(0);fine_count=fine.sum(0);negative_count=negative.sum(0);candidate=(raw_count>=2)&(negative_count<2);lower=(fine_count>=2)&(negative_count<2)
 variants={}
 for name,mask,cc in [('raw_union_samples',candidate,cc_face),('fine_filtered_sample_control',lower,cc_face)]:
  p=C.proposals(v,f,mask,cc,raw_count,gap)
  for row in p['records']:row.update(quality_status='score_unverified_sample_only',subclass_status='not_assigned',observation_scope='stratified_sample_only')
  variants[name]=dict(accepted=mask,proposals=p)
  save(out/(name+'_proposals.json'),p['records']);save(out/(name+'_relations.json'),dict(links=p['relations'],cut_topology_edges=p['cut_edges'],settings=p['settings'],initial_components=p['initial_components'],initial_component_faces=p['initial_component_faces']))
 arrays={k:mesh[k] for k in ['vertices','faces','rgb','face_class_id','observations','geometric_observations','vote_agreement','mean_winning_support']};arrays.update(existing_profile_instance_id=old,raw_parent_observation_frames=raw_count.astype('u1'),known_fine_qualified_frames=fine_count.astype('u1'),nonparent_sample_frames=negative_count.astype('u1'))
 arrays['raw_parent_observation_matrix']=positive;arrays['fine_qualified_observation_matrix']=fine;arrays['frame_parent_component_id']=cc_face
 quality=np.where(raw_count>0,1,0).astype('u1');quality[fine_count>0]=2;quality[fine_count>=2]=3;arrays['parent_quality_status']=quality
 for name,a in variants.items():arrays[name+'_parent_mask']=a['accepted'];arrays[name+'_proposal_id']=a['proposals']['ids'];arrays[name+'_uncertain_fragment']=a['proposals']['uncertain']
 for name,ids in refs.items():arrays['review_'+name+'_faces']=np.array(sorted(ids),np.uint32)
 np.savez_compressed(out/'arrays.npz',**arrays);report=evaluate(v,f,labels,old,variants,refs,'LOCAL_SAMPLE_ONLY: exact stratified correspondences, not complete projection or score-qualified parent fusion');report.update(reference_conflicts_removed=len(overlap|ref_conflicts),samples_total=sum(x['sample_count'] for x in frame_rows),score_sum_status='not_executed_real_scores_only_on_unavailable_server',parent_quality='not_reconstructed; raw candidates remain score-unverified',mesh_digest=meta['mesh_digest'],new_instances_are='provisional_score_unverified, not physical GT')
 save(out/'metrics.json',report);save(out/'frames.json',frame_rows);save(out/'run_status.json',dict(mode='local_samples',full_server_run=False,GPU_used=False,new_model_inference=False,original_arrays_preserved=True,subclass_assignment=False,scope_limit='Only available sample correspondences; no synthesized scores',settings=read(HERE/'config.json')))
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['local_samples'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.relative_to(HERE.resolve());out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();report=local_samples(out);print(json.dumps(dict(output=str(out),seconds=time.perf_counter()-start,variants={k:{a:b for a,b in v.items() if a in ['parent_faces','proposal_faces','proposals','unconfirmed_parent_faces','shared_proposal_ids_between_two_reference_objects']} for k,v in report['variants'].items()}),ensure_ascii=False))
