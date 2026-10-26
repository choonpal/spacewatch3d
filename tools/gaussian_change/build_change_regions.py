"""One change registry shared by all keyframe observations.

Spatial regions group navigation only; different objects remain separate events.
Original SimSaC boxes and registration are never edited or re-numbered.
"""
from pathlib import Path
import argparse
import numpy as np
from prepare import read,save,sha

POLICY={'same_event':'same source ID or identical automatic object correspondence; never keyframe ID',
        'area_grouping':'shared observing keyframe and AABB gap <= 3 * scene height',
        'maximum_area_gap_h':3.,'counts':'unique events, not observations',
        'representative':'most events visible, then best minimum two-epoch depth support',
        'source_geometry_changed':False}


def identity(box,evidence):
    event=evidence.get(box['id'],{}).get('event',{})
    match=event.get('automatic_match')
    if match:return ('object-motion',event['type'],event.get('class_name'),match['reference_track'],match['source_track'])
    pair=event.get('selected_pair')
    if pair and any(pair) and event['type']!='unresolved':return ('object',event['type'],event.get('class_name'),*pair)
    return ('source-candidate',box['id'])


def gap(a,b):
    return float(np.linalg.norm(np.maximum(0,np.maximum(np.asarray(a[0])-b[1],np.asarray(b[0])-a[1]))))


def make_mode(boxes,visibility,views,order,height,evidence):
    unique={}
    for b in boxes:
        if b['id'] in unique and b!=unique[b['id']]:raise ValueError('Conflicting data for source candidate '+b['id'])
        unique.setdefault(b['id'],b)
    events={};signatures={};candidate_to_event={}
    for b in unique.values():
        signature=identity(b,evidence);id=signatures.setdefault(signature,b['id'])
        candidate_to_event[b['id']]=id
        events.setdefault(id,{'id':id,'candidate_ids':[],'observing_keyframes':[]})['candidate_ids'].append(b['id'])
    positions={v['id']:np.asarray(v['camera_to_common_cv'])[:3,3] for v in views}
    def support(id,view):
        scores=[min(visibility[view][alias]['support_rays'].values()) for alias in events[id]['candidate_ids'] if visibility[view][alias]['visible']]
        return max(scores,default=0)
    def observes(id,view):return any(visibility[view][alias]['visible'] for alias in events[id]['candidate_ids'])
    def rank(view,ids):
        seen=[id for id in ids if observes(id,view)]
        distance=np.mean([np.linalg.norm(positions[view]-np.mean(events[id]['bounds'],axis=0)) for id in seen]) if seen else 1e9
        return (len(seen),min((support(id,view) for id in ids),default=0),sum(support(id,view) for id in ids),-float(distance),-order.index(view))
    for id,event in events.items():
        bounds=np.asarray([unique[alias]['bounds'] for alias in event['candidate_ids']])
        event['bounds']=[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()]
        event['observing_keyframes']=[v for v in order if observes(id,v)]
        event['best_view']=max(event['observing_keyframes'],key=lambda v:rank(v,[id]),default=None)
    # Connected groups of nearby changes sharing actual depth-supported views.
    pending=set(events);groups=[]
    while pending:
        seed=next(id for id in events if id in pending);pending.remove(seed);component=[seed];stack=[seed]
        while stack:
            a=stack.pop()
            neighbors=[b for b in events if b in pending and set(events[a]['observing_keyframes'])&set(events[b]['observing_keyframes'])
                       and gap(events[a]['bounds'],events[b]['bounds'])<=POLICY['maximum_area_gap_h']*height]
            for b in neighbors:pending.remove(b);component.append(b);stack.append(b)
        groups.append(component)
    groups.sort(key=lambda ids:min((order.index(v) for id in ids for v in events[id]['observing_keyframes']),default=len(order)))
    regions=[];event_to_region={}
    for index,ids in enumerate(groups,1):
        observed=[v for v in order if any(observes(id,v) for id in ids)];representative=max(observed,key=lambda v:rank(v,ids),default=None)
        region={'id':'area-'+ids[0],'label':'구역 '+str(index),'color_index':index-1,'event_ids':ids,
                'source_zones':sorted({unique[a]['zone'] for id in ids for a in events[id]['candidate_ids']}),
                'observing_keyframes':observed,'representative_view':representative}
        regions.append(region)
        for id in ids:event_to_region[id]=region['id']
    observations={}
    for view in order:
        ids=[id for id in events if observes(id,view)]
        observations[view]={'event_ids':ids,'region_ids':list(dict.fromkeys(event_to_region[id] for id in ids)),
                            'candidate_ids':[id for id in unique if visibility[view][id]['visible']],
                            'new_events_created':0}
    return {'events':events,'candidate_to_event':candidate_to_event,'event_to_region':event_to_region,
            'regions':regions,'observations':observations,
            'counts':{'source_candidates':len(unique),'unique_events':len(events),'regions':len(regions),
                      'observation_links':sum(len(v['event_ids']) for v in observations.values())}}


def compute(folder):
    fixed=read(folder/'tour-candidates.json');data=read(folder/'gaussian-data.json');routes=read(folder/'tour-routes.json');evidence=read(folder/'object-evidence.json')
    order=[id for group in routes['groups'] for segment in group['segments'] for id in segment['nodes']]
    result={'schema':'shared-spatial-change-registry-v1','policy':POLICY,'modes':{},
            'source_sha256':{name:sha(folder/name) for name in ['tour-candidates.json','object-evidence.json','tour-routes.json','gaussian-data.json']}}
    for mode,boxes in fixed['boxes'].items():
        visibility={id:v[mode] for id,v in fixed['visibility'].items()}
        result['modes'][mode]=make_mode(boxes,visibility,data['zones'],order,data['height_u'],evidence['regions'][mode])
    return result


def build(folder):
    result=compute(folder);save(folder/'change-regions.json',result);return result


def validate(folder):
    result=read(folder/'change-regions.json');assert result==compute(folder)
    for mode,index in result['modes'].items():
        ids=[id for region in index['regions'] for id in region['event_ids']]
        assert len(ids)==len(set(ids))==index['counts']['unique_events']
        assert set(ids)==set(index['events'])
        assert all(v['new_events_created']==0 for v in index['observations'].values())
    return {'passed':True,'counting_unit':'unique change event','repeated_views_create_events':False,
            'counts':{mode:index['counts'] for mode,index in result['modes'].items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-uncapped-tour-20261026'))
    args=parser.parse_args();build(args.directory);print(validate(args.directory))
