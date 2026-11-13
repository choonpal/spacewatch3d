"""Prepared full 5-frame CPU replay. NOT executed while the server is offline."""
from pipeline import HERE,FRAMES,C,read,save,sha,digest,evaluate,roi_faces,mask_cc_sample
import os,sys,signal,time,json
from pathlib import Path
import numpy as np
from PIL import Image

def frame_record(face,labels,support,eligible,nfaces,legacy,minimum_parent_pixels=False):
 w=support.copy();w[~eligible]=0;votes=legacy.frame_votes(face,labels,w,.35);total=np.bincount(face[eligible],minlength=nfaces);keep=(votes['weight']>0)&(total[votes['index']]>=2)
 if minimum_parent_pixels:
  parent=np.bincount(face[eligible&(labels==3)],minlength=nfaces);keep&=(votes['label']!=3)|(parent[votes['index']]>=2)
 return votes[keep]

def main(output):
 import open3d as o3d
 source=Path('/workspace/spacewatch_pipeline_v2');run=Path('/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v2_20261024');score_root=Path('/workspace/lhw/ooops_door_b0b1_20261027_r1/attempt03/results/B0')
 assert HERE.resolve().is_relative_to(Path('/workspace/lhw').resolve());memory=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert int(memory.split()[1])>=4*1024**2
 signal.alarm(900);start=time.perf_counter();output=output.resolve();output.relative_to(HERE.resolve());output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir();(output/'cache').mkdir()
 sys.path[:0]=[str(source/'vendor/segmentation'),str(source)]
 from geometry import source_frames
 import core as legacy
 from spacewatch_pipeline.geometry import similarity,transform
 identity=read(run/'identity.json');protected={}
 for rel in ['spacewatch_pipeline/geometry.py','spacewatch_pipeline/early_semantics.py','vendor/segmentation/geometry.py','vendor/segmentation/core.py']:
  path=source/rel;protected[str(path)]=sha(path);assert protected[str(path)]==identity['code'][rel]
 results=read(run/'results.json');src=results['dorm2.filter'];mesh=results['dorm2.mesh'];sem=results['dorm2.semantic'];data=np.load(sem['arrays']);v=data['vertices'];f=data['faces'];labels=data['face_class_id'];n=len(f);coords=read(mesh['coordinates']);tf,scale=similarity(coords['source_to_mesh']);gap=.1*scale;roles=np.load('/workspace/spacewatch_pipeline_v5_1/profiles/dorm2_statistical_scored/face_roles.npz');old=roles['target_region_index'];assert np.array_equal(roles['source_class_id'],labels)
 for path in [Path(sem['arrays']),Path(mesh['coordinates']),Path('/workspace/spacewatch_pipeline_v5_1/profiles/dorm2_statistical_scored/face_roles.npz')]:protected[str(path)]=sha(path)
 index=read(Path(src['source_index'])/'index.json');sealed=read(Path(src['source_index'])/'complete.json')
 for path,record in sealed['files'].items():assert sha(path)==record['sha256']
 keys=[np.load(a['path'],mmap_mode='r') for a in index['buckets']];rowmap=np.load(index['unfiltered_to_target'],mmap_mode='r');poses=np.array([r['camera_to_world'] for r in read(Path(src['source_run'])/'result/camera_poses.json')]);centers=transform(poses[:,:3,3],tf);ray=o3d.t.geometry.RaycastingScene(nthreads=2);ray.add_triangles(o3d.core.Tensor(v.astype('f4')),o3d.core.Tensor(f.astype('u4')))
 expected=np.load(HERE/'expected_legacy_frames.npz');roi=read(HERE/'review_rois.json');refs={x:set() for x in ['hinged_door','elevator','wall_patch','floor_patch']};names=['legacy','hard_union','score_sum'];records={k:[] for k in names};positive={k:np.zeros((5,n),bool) for k in names};raw={k:np.zeros((5,n),bool) for k in names[1:]};facecc={k:np.zeros((5,n),np.uint32) for k in names[1:]};frame_rows=[];stats=[]
 for frame,xyz,mask,_ in source_frames(dict(source_run=src['source_run']),selected=set(FRAMES)):
  j=FRAMES.index(frame);cells=legacy.voxel_keys(xyz,index['voxel']);bucket=legacy.buckets(cells,len(keys));rows=np.empty(len(xyz),np.uint32)
  for k,base in enumerate(keys):
   select=bucket==k
   if select.any():rows[select]=legacy.exact_lookup(base,cells[select])+index['buckets'][k]['offset']
  kept=rowmap[rows]!=np.uint32(2**32-1);points=transform(xyz[kept],tf);origin=centers[frame];delta=points-origin;distance=np.linalg.norm(delta,axis=1);hit=ray.cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(origin,points.shape),delta/np.maximum(distance[:,None],1e-12)].astype('f4')),nthreads=2);depth=hit['t_hit'].numpy();fid=hit['primitive_ids'].numpy();valid=np.isfinite(depth)&(abs(depth-distance)<=gap)&(distance>0);pixels=np.flatnonzero(mask)[kept][valid];fid=fid[valid];point=points[valid];error=(depth[valid]-distance[valid]).astype('f4')
  pred=score_root/f'{frame:05d}.npz';base_report=read(score_root/'report.json');assert sha(pred)==next(x['prediction_sha256'] for x in base_report['frames'] if x['frame']==frame);protected[str(pred)]=sha(pred)
  with np.load(pred) as a:
   native=a['labels'];scores=a['all_class_scores'];lab,support=legacy.resize_labels_to_source(native,a['support_fraction'],*mask.shape);_,margin=legacy.resize_labels_to_source(native,a['margin_fraction'],*mask.shape);lab=lab.ravel()[pixels];support=support.ravel()[pixels];margin=margin.ravel()[pixels];elig=(support>=.35)&(margin>=.1)&(lab!=16);oldvote=frame_record(fid,lab,support,elig,n,legacy);records['legacy'].append(oldvote);positive['legacy'][j,oldvote['index'][np.isin(oldvote['label'],[3,4])]]=True;face_labels=np.full(n,255,np.uint8);face_labels[oldvote['index']]=oldvote['label'];assert np.array_equal(face_labels,expected[f'frame_{frame:05d}'])
   frame_data=dict(pixels=pixels.astype('u4'),face_ids=fid.astype('u4'),source_points=point.astype('f4'),depth_errors=error,camera=origin.astype('f4'),source_shape=np.array(mask.shape),source_class=lab,legacy_pixel_eligible=elig)
   for name in names[1:]:
    p=C.parent_pixels(scores,native,name);mapped={}
    for key in ['raw','labels','support','margin','eligible']:
     if key=='labels':x=legacy.resize_labels_to_source(p[key],p['support'],*mask.shape)[0]
     else:x=legacy.resize_labels_to_source(native,p[key].astype('f4'),*mask.shape)[1]
     mapped[key]=x.ravel()[pixels]
    mapped['raw']=mapped['raw']>0;mapped['eligible']=mapped['eligible']>0;vote=frame_record(fid,mapped['labels'],mapped['support'],mapped['eligible'],n,legacy,True);records[name].append(vote);positive[name][j,vote['index'][vote['label']==3]]=True;raw[name][j,np.unique(fid[mapped['raw']])]=True;cc=C.periodic_components(p['raw']);ccpoints=mask_cc_sample(cc,pixels,mask.shape);chosen=mapped['raw']&mapped['eligible'];facecc[name][j]=C.dominant_face_component(fid[chosen],ccpoints[chosen],n)
    for key,value in mapped.items():frame_data[name+'_'+key]=value
    Image.fromarray((p['raw']*255).astype('u1')).save(output/'frames'/f'{frame:05d}_{name}_raw.png');Image.fromarray((p['parent_eligible']*255).astype('u1')).save(output/'frames'/f'{frame:05d}_{name}_eligible.png')
    stats.append(dict(frame=frame,variant=name,raw_parent_pixels=int(p['raw'].sum()),eligible_parent_pixels=int(p['parent_eligible'].sum()),geometry_parent_pixels=int(mapped['raw'].sum()),geometry_eligible_parent_pixels=int(chosen.sum()),parent_voted_faces=int((vote['label']==3).sum())))
  np.savez_compressed(output/'cache'/f'{frame:05d}.npz',**frame_data)
  # Display-only exact correspondences: not the full calculation input.
  select=np.linspace(0,len(pixels)-1,min(2500,len(pixels))).astype(int);np.savez_compressed(output/'frames'/f'{frame:05d}.npz',sample_pixels=pixels[select].astype('u4'),sample_faces=fid[select].astype('u4'),sample_points=point[select].astype('f4'),sample_labels=lab[select],sample_states=np.where(elig[select],4,3).astype('u1'),sample_depth_error=error[select],camera=origin.astype('f4'),face_labels=face_labels)
  for group in ['positive','negative']:
   for obj,box in roi[group][str(frame)].items():refs[obj].update(map(int,roi_faces(pixels,fid,box,mask.shape[1])))
  original=Path(src['base_source_run'])/'input/frames'/f'frame_{frame:05d}.jpg';frame_rows.append(dict(frame=frame,source_shape=list(mask.shape),source_image=str(original),source_image_sha256=sha(original),geometry_correspondences=len(pixels),legacy_vote_replay_exact=True));print('CPU_PROJECTED',frame,len(pixels),flush=True)
 variants={};arrays={k:data[k] for k in data.files};arrays['existing_profile_instance_id']=old
 for name in names:
  rec=np.concatenate(records[name]);ids,win,agree,mean,obs=legacy.fuse_votes(rec,.55,.2,2);fused=np.full(n,255,np.uint16);fused[ids]=win;arrays[name+'_fused_class_id']=fused;arrays[name+'_positive_frames']=positive[name].sum(0).astype('u1')
  if name=='legacy':
   arrays['legacy5_parent_mask']=np.isin(fused,[3,4]);continue
  count=positive[name].sum(0);accepted=(fused==3)&(count>=2);prop=C.proposals(v,f,accepted,facecc[name],count,gap);variants[name]=dict(accepted=accepted,proposals=prop);arrays[name+'_parent_mask']=accepted;arrays[name+'_proposal_id']=prop['ids'];arrays[name+'_uncertain_fragment']=prop['uncertain'];arrays[name+'_raw_parent_observation_frames']=raw[name].sum(0).astype('u1');arrays[name+'_frame_component_id']=facecc[name];a=np.zeros(n,'f4');a[ids]=agree;arrays[name+'_parent_agreement']=a;a=np.zeros(n,'f4');a[ids]=mean;arrays[name+'_parent_mean_support']=a
  save(output/(name+'_proposals.json'),prop['records']);save(output/(name+'_relations.json'),dict(links=prop['relations'],cut_topology_edges=prop['cut_edges'],settings=prop['settings'],initial_components=prop['initial_components'],initial_component_faces=prop['initial_component_faces']))
 overlap=(refs['hinged_door']&refs['elevator'])|((refs['hinged_door']|refs['elevator'])&(refs['wall_patch']|refs['floor_patch']))
 for obj in refs:refs[obj]-=overlap;arrays['review_'+obj+'_faces']=np.array(sorted(refs[obj]),np.uint32)
 np.savez_compressed(output/'arrays.npz',**arrays);report=evaluate(v,f,labels,old,variants,refs,'server_full_5frame_complete_pixel_correspondence');report['matched_legacy5_reference']={obj:dict(faces=int(arrays['legacy5_parent_mask'][list(refs[obj])].sum()),area=float(C.face_geometry(v,f)[2][list(refs[obj])][arrays['legacy5_parent_mask'][list(refs[obj])]].sum())) for obj in ['hinged_door','elevator']};report['source_integrity_unchanged']=all(sha(p)==h for p,h in protected.items());assert report['source_integrity_unchanged'];report['seconds']=time.perf_counter()-start
 save(output/'metrics.json',report);save(output/'frames.json',frame_rows);save(output/'pixel_stage_statistics.json',stats);save(output/'input_hashes.json',protected);save(output/'run_status.json',dict(mode='server_full',GPU_used=False,new_model_inference=False,subclass_assignment=False,all_five_legacy_face_vote_labels_exact=True,original_arrays_preserved=True,seconds=time.perf_counter()-start));signal.alarm(0)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
