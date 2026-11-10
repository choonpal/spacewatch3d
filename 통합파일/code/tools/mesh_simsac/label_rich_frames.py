"""Rank nearby real frames by observed object-label variety, without label quotas."""
import numpy as np

def select_label_rich(frames,target,radius,count=3,min_label_pixels=32):
    target=np.asarray(target,float)
    if target.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(radius) or radius<=0 or count<1 or min_label_pixels<1:
        raise ValueError('Finite spatial target, positive radius and budgets required')
    rows=[]
    for f in frames:
        position=np.asarray(f['common_position'],float)
        if position.shape!=(3,) or not np.isfinite(position).all():continue
        distance=float(np.linalg.norm(position-target))
        if distance>radius:continue
        labels={str(k):int(v) for k,v in f.get('label_counts',{}).items() if k not in {'wall','floor','ceiling'} and v>=min_label_pixels}
        rows.append({**f,'distance_u':distance,'label_type_count':len(labels),'visible_labels':sorted(labels),
            'qualifying_label_counts':labels,'object_label_pixels':sum(labels.values())})
    # Two independent-frame votes remain a detector rule, not a per-label quota.
    rows.sort(key=lambda f:(-f['label_type_count'],-f['object_label_pixels'],-float(f.get('sharpness',0)),f['distance_u'],int(f['id'])))
    chosen=[];used=set()
    for f in rows:
        if f['id'] not in used:chosen.append(f);used.add(f['id'])
        if len(chosen)>=count:break
    return chosen
