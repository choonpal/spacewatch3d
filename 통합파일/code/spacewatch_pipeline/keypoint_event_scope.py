"""Use original detector-to-keypoint links. Never infer links from distance."""
import copy
import numpy as np


def apply_keypoint_scope(overlay,comparison,plan,events):
    original_events={e['id']:e for e in events}
    related={row['id']:{e['id'] for e in row['related_events']} for row in comparison['keypoints']}
    by_view={row['viewpoint_id']:row for row in plan['keypoints']}
    if len(by_view)!=len(plan['keypoints']):raise ValueError('Ambiguous original keypoint mapping')
    if set(related)!={row['id'] for row in plan['keypoints']}:raise ValueError('Keypoint scope differs from source plan')
    out=copy.deepcopy(overlay);audit=[]
    for capture,group in out['captures'].items():
        for row in group['views']:
            keypoint=by_view.get(row['viewpoint_id'])
            allowed=related[keypoint['id']] if keypoint is not None and keypoint['anchor_capture']==capture else set()
            if not allowed<=set(original_events):raise ValueError('Scope contains unknown source event')
            merged={};dropped=[]
            for item in row['items']:
                if item['id'] not in allowed:
                    dropped.append(item['id']);continue
                event=original_events[item['id']]
                if item['type']!=event['type'] or item['label']!=event['label']:raise ValueError('Source event decision changed')
                if item['id'] not in merged:
                    merged[item['id']]=item
                else:
                    # One source event remains one annotation at a keypoint,
                    # not a new candidate for every disconnected mesh fragment.
                    first=merged[item['id']]
                    first['angular_samples']+=item['angular_samples']
                    first['pixels']=first.get('pixels',0)+item.get('pixels',0)
                    samples=np.radians(np.asarray(first['angular_samples']))
                    yaw=samples[:,0];pitch=-samples[:,1]
                    direction=np.array([np.mean(np.sin(yaw)*np.cos(pitch)),np.mean(np.sin(pitch)),np.mean(np.cos(yaw)*np.cos(pitch))])
                    first['yaw']=float(np.degrees(np.arctan2(direction[0],direction[2])))
                    first['pitch']=float(-np.degrees(np.arctan2(direction[1],np.hypot(direction[0],direction[2]))))
            row['items']=list(merged.values())
            audit.append(dict(capture=capture,viewpoint_id=row['viewpoint_id'],
                source_keypoint=keypoint['id'] if keypoint and keypoint['anchor_capture']==capture else None,
                source_allowed=sorted(allowed),displayed=sorted(merged),dropped_out_of_scope=sorted(set(dropped))))
    return out,audit
