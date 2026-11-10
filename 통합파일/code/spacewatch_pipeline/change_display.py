"""Display-only visibility/legibility. Never feeds detector evidence or labels."""
import numpy as np


def event_display_mask(hit_ids, distances, finite, photographer, event_faces,
                       potential_pixels, max_distance_m=6., min_pixels=64,
                       min_fraction=.2):
    ids=np.asarray(hit_ids);distance=np.asarray(distances)
    valid=np.asarray(finite,bool)&np.isfinite(distance)&(distance>0)
    if not np.isfinite(max_distance_m) or max_distance_m<=0:
        raise ValueError('Display distance must be positive')
    on_event=np.zeros(valid.shape,bool)
    on_event[valid]=np.asarray(event_faces,bool)[ids[valid]]
    unmasked=on_event&~np.asarray(photographer,bool)
    mask=unmasked&(distance<=max_distance_m)
    pixels=int(mask.sum());fraction=pixels/max(1,int(potential_pixels))
    passed=pixels>=min_pixels and fraction>=min_fraction
    return mask,dict(passed=bool(passed),pixels=pixels,visible_fraction=fraction,
        beyond_display_distance=int((unmasked&(distance>max_distance_m)).sum()),
        max_visible_distance_m=float(distance[mask].max()) if pixels else None,
        scope='viewer_only; not detector evidence')
