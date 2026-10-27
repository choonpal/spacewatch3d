"""Compare completed runs without relabeling detections or counting disappearance as success."""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
from tools.mesh_simsac.core import read,write


def metrics(folder):
    events=read(folder/'object-events.json');changes=read(folder/'changes.json');inputs=read(folder/'inputs.json');model=read(folder/'model.json')
    count=len(events['events']);unknown=events['counts']['unresolved']
    return {'counts':events['counts'],'events':count,'raw_candidates':len(changes['candidates']),
            'unresolved_reasons':dict(Counter(e['reason'] for e in events['events'] if e['type']=='unresolved')),
            'unresolved_rate':unknown/max(count,1),'excluded_surfaces':len(events.get('excluded_surface_candidates',[])),
            'render_size':[inputs['settings']['width'],inputs['settings']['height']],
            'inference_mode':inputs['settings'].get('inference_mode','panorama'),
            'views':len(changes['observations']),'camera_counts':inputs['camera_counts'],
            'checkpoint_sha256':model['checkpoint_sha256'],'mesh_camera_hashes':{k:v for k,v in inputs['hashes'].items() if k.endswith(('.mesh','.cameras'))},
            'gaussian_hashes':{k:v for k,v in inputs['hashes'].items() if k.endswith('.gaussian')},
            'supported_regions':changes['registration_supported_regions'],'regions':changes['region_count']}


def instance_sets(event,objects):
    return [set(o.get('source_observation_ids',[])) for key in event.get('object_ids',[]) if (o:=objects.get(key)) and o.get('source_observation_ids')]


def overlap(a,b,tolerance):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a+np.array([-1,1])[:,None]*tolerance;b=b+np.array([-1,1])[:,None]*tolerance
    intersection=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0])).prod()
    union=np.maximum(0,a[1]-a[0]).prod()+np.maximum(0,b[1]-b[0]).prod()-intersection
    return float(intersection/max(union,1e-12))


def compare_events(baseline,current,baseline_candidates,current_candidates,height):
    old_objects={o['id']:o for o in baseline.get('objects',[])};new_objects={o['id']:o for o in current.get('objects',[])}
    raw={c['id']:c for c in current_candidates};excluded={c['candidate_id'] for c in current.get('excluded_surface_candidates',[])}
    rows=[]
    for old in baseline['events']:
        if old['type']!='unresolved':continue
        own=instance_sets(old,old_objects);matches=[]
        for new in current['events']:
            their=instance_sets(new,new_objects)
            score=max((len(a&b)/len(a|b) for a in own for b in their if a|b),default=0.)
            basis='shared_unary_instance_observations'
            if not own:
                bounds=[new['bounds']]+[new[k] for k in ['before_bounds','after_bounds'] if k in new]+[raw[c]['bounds'] for c in new.get('source_candidate_ids',[]) if c in raw]
                score=max((overlap(old['bounds'],b,.05*height) for b in bounds),default=0.);basis='spatial_region_only'
            if score>=(.5 if own else .1):matches.append({'id':new['id'],'type':new['type'],'match_basis':basis,'score':score,'reason':new['reason']})
        types={m['type'] for m in matches}
        if types and 'unresolved' not in types:outcome='classified'
        elif 'unresolved' in types:outcome='still_unresolved' if len(types)==1 else 'partly_classified'
        elif any(overlap(old['bounds'],raw[c]['bounds'],.05*height)>=.1 for c in excluded if c in raw):outcome='excluded_as_structural_surface'
        else:outcome='no_corresponding_detection'
        rows.append({'baseline_id':old['id'],'baseline_reason':old['reason'],'current_matches':matches,'outcome':outcome})
    return rows


def compare(baseline_folder,current_folder):
    baseline_folder=Path(baseline_folder);current_folder=Path(current_folder)
    state=read(current_folder/'status.json')
    if state.get('state')!='complete':raise ValueError('Compare completed output directories only')
    a,b=metrics(baseline_folder),metrics(current_folder)
    old,new=read(baseline_folder/'object-events.json'),read(current_folder/'object-events.json')
    old_changes,new_changes=read(baseline_folder/'changes.json'),read(current_folder/'changes.json')
    h=read(current_folder/'registration.json')['height_u']
    rows=compare_events(old,new,old_changes['candidates'],new_changes['candidates'],h)
    old_objects={o['id']:o for o in old.get('objects',[])};new_objects={o['id']:o for o in new.get('objects',[])};motions=[]
    for event in old['events']:
        if event['type']!='moved':continue
        own=instance_sets(event,old_objects)
        for current in new['events']:
            if current['type']!='moved':continue
            other=instance_sets(current,new_objects)
            score=min((max((len(a&b)/len(a|b) for b in other if a|b),default=0.) for a in own),default=0.)
            if score>=.5:motions.append({'baseline_id':event['id'],'current_id':current['id'],'shared_instance_score':score,
                'baseline_appearance':event['correspondence']['appearance'],'current_appearance':current['correspondence']['appearance'],
                'baseline_origin_hypotheses':max(1,len(event.get('origin_candidates',[]))),
                'current_origin_hypotheses':max(1,len(current.get('origin_candidates',[])))})
    old_cameras=read(baseline_folder/'cameras.json')['views'];new_cameras=read(current_folder/'cameras.json')['views']
    def key(folder):return folder.parent.name if re.fullmatch('[a-f0-9]{24}',folder.parent.name) else None
    result={'schema':'gaussian-change-run-comparison-v1','baseline_job_id':key(baseline_folder),'current_job_id':key(current_folder),
            'baseline':a,'current':b,'baseline_unresolved_regions':rows,'region_outcomes':dict(Counter(r['outcome'] for r in rows)),
            'matched_motion_events':motions,
            'same_meshes_and_camera_inputs':a['mesh_camera_hashes']==b['mesh_camera_hashes'],
            'same_camera_request_order':[v['id'] for v in old_cameras]==[v['id'] for v in new_cameras],
            'same_model_checkpoint':a['checkpoint_sha256']==b['checkpoint_sha256'],
            'same_classification_gates':old['mask_instance_matching_parameters']==new['mask_instance_matching_parameters'] and old['policy']==new['policy'],
            'registration_fitted_again':current_folder.resolve()!=baseline_folder.resolve() and all(s in state.get('stages_executed',[]) for s in ['global_registration','regional_correction']),
            'prior_detections_consumed':read(current_folder/'inputs.json').get('prior_detections_consumed'), 'manual_labels_changed':False,
            'unresolved_count_delta':b['counts']['unresolved']-a['counts']['unresolved'],
            'missing_detection_is_counted_as_resolved':False,'accuracy_evaluated':False,
            'interpretation':'Model maps, render resolution and inference partition changed together. Count/rate changes are not an accuracy benchmark; absence of a previous detection is not successful classification.'}
    write(current_folder/'comparison.json',result);new_changes['comparison']='comparison.json';write(current_folder/'changes.json',new_changes)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--current',type=Path,required=True);args=p.parse_args()
    result=compare(args.baseline,args.current)
    print({'baseline':result['baseline']['counts'],'current':result['current']['counts'],'region_outcomes':result['region_outcomes']})
