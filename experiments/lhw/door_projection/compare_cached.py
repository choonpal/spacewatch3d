"""Cache-only comparison. Retain nonplanar groups as provisional, never final IDs."""
from pipeline import C,read,save,evaluate
from pathlib import Path
import numpy as np,time
HERE=Path(__file__).resolve().parent
out=HERE/'comparison_r3';out.mkdir(exist_ok=False)
a=np.load(HERE/'full_r1/arrays.npz');v=a['vertices'];f=a['faces'];refs={x:set(map(int,a['review_'+x+'_faces'])) for x in ['hinged_door','elevator','wall_patch','floor_patch']};gap=read(HERE/'full_r1/hard_union_relations.json')['settings']['max_gap_mesh_units'];arrays={k:a[k] for k in a.files};variants={}
for name in ['hard_union','score_sum']:
 mask=a[name+'_parent_mask'];cc=a[name+'_frame_component_id'];count=a[name+'_positive_frames'];prop=C.proposals(v,f,mask,cc,count,gap,nonplanar_policy='retain_provisional');variants[name]=dict(accepted=mask,proposals=prop);arrays[name+'_proposal_id']=prop['ids'];arrays[name+'_uncertain_fragment']=prop['uncertain'];save(out/(name+'_proposals.json'),prop['records']);save(out/(name+'_relations.json'),dict(links=prop['relations'],cut_topology_edges=prop['cut_edges'],settings=prop['settings'],initial_components=prop['initial_components'],initial_component_faces=prop['initial_component_faces']))
np.savez_compressed(out/'arrays.npz',**arrays);report=evaluate(v,f,a['face_class_id'],a['existing_profile_instance_id'],variants,refs,'complete_5frame_correspondence_optional_nonplanar_provisional');report['baseline']=read(HERE/'full_r1/metrics.json');report['verified_physical_instances']=False;save(out/'metrics.json',report);print(report['variants'])
