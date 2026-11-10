"""Select actual camera frames near frozen tour keypoints in a common map."""
import numpy as np

def select_nearby(frames, target, radius, count=3):
    if not np.isfinite(radius) or radius <= 0 or count < 1:
        raise ValueError('Positive radius and frame budget required')
    target=np.asarray(target,float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('Finite common-frame target required')
    eligible=[]
    for f in frames:
        p=np.asarray(f['common_position'],float)
        if p.shape!=(3,) or not np.isfinite(p).all():continue
        distance=float(np.linalg.norm(p-target))
        if distance<=radius:
            eligible.append({**f,'distance_u':distance})
    if not eligible:return []
    sharp=np.asarray([max(0,float(f.get('sharpness',0))) for f in eligible])
    order=np.argsort(np.argsort(sharp,kind='stable'),kind='stable')/max(1,len(eligible)-1)
    for i,f in enumerate(eligible):
        f['selection_score']=float(.6*(1-f['distance_u']/radius)+.3*order[i]+.1*min(1,f.get('object_class_count',0)/5))
    chosen=[];used=set()
    while len(chosen)<count:
        candidates=[f for f in eligible if f['id'] not in used]
        if not candidates:break
        def rank(f):
            diversity=min((np.linalg.norm(np.asarray(f['common_position'])-np.asarray(g['common_position']))/radius for g in chosen),default=1)
            return (f['selection_score']+.12*min(1,diversity),-f['distance_u'],-int(f['id']))
        f=max(candidates,key=rank);chosen.append(f);used.add(f['id'])
    return chosen

def grouped_references(choices, keypoint_ids, max_refs=2):
    """Cover each keypoint using opposite frames of that same keypoint only."""
    refs={}
    for keypoint in keypoint_ids or [None]:
        frames=set();n=0
        for score,ref,overlap,distance in sorted(choices,key=lambda x:(-x[0],x[1]['id'])):
            if keypoint is not None and keypoint not in ref.get('keypoint_ids',[]):continue
            if ref['frame_id'] in frames:continue
            frames.add(ref['frame_id']);n+=1
            row=refs.setdefault(ref['id'],dict(id=ref['id'],overlap=overlap,camera_distance_u=distance,keypoint_ids=[]))
            if keypoint is not None:row['keypoint_ids'].append(keypoint)
            if n>=max_refs:break
    return list(refs.values())
