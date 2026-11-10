"""Retain RGB people and boxes when the reconstruction omitted their mesh.

Only existing mesh-registered RGB pairs and their fresh SimSaC probabilities are
used. Background ray hits are a grouping proxy, never a synthesized person mesh.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .core import read, project_in_view


def candidates(root, prepared):
    root = Path(root)
    metadata = {v['id']:v for v in prepared['views']}
    pairs = {p['query']:p for p in read(root/'pairs.json')}
    views = {}

    def view(vid):
        if vid not in views:
            views[vid] = dict(np.load(root/'views'/(vid+'.npz')))
        return views[vid]

    observations = []
    for vid, meta in metadata.items():
        if meta.get('person_mask_policy') != 'capturer_only':
            continue
        query = view(vid)
        if 'segmentation_class' not in query:
            continue
        components = np.zeros(query['person'].shape,'i4')
        count, component_labels = 0, {}
        for name in ['person','box']:
            if name not in meta.get('segmentation_labels', []):
                continue
            mask = (query['segmentation_class'] == meta['segmentation_labels'].index(name)+1) & ~query['person']
            own, number = label(mask)
            components[own > 0] = own[own > 0]+count
            component_labels.update({count+i:name for i in range(1,number+1)})
            count += number
        refs = pairs.get(vid, {}).get('refs', [])
        if not refs:
            continue
        # Preserve the geometry-ranked best reference; an unrelated second
        # reference cannot create a person candidate by maximum pooling.
        reference_id = refs[0]['id']
        reference = view(reference_id)
        path = root/'results/pairs'/(vid+'--'+reference_id+'.npz')
        probability = np.load(path)['probability']
        positive = (probability > prepared['settings']['probability_threshold']) & ~query['person']
        for component in range(1, count+1):
            mask = components == component
            pixels = int(mask.sum())
            changed = int((mask & positive).sum())
            if pixels < 32 or changed < prepared['settings']['min_object_pixels'] or changed/pixels < .02:
                continue
            supported = mask & query['mesh_hit']
            if supported.sum() < 16:
                continue  # Cannot locate this person in the registered opposite view.
            points = query['points'][supported]
            regions = query['point_region'][supported] if 'point_region' in query else None
            uv, z, inside = project_in_view(points, reference, regions)
            uv = uv[inside]
            if len(uv) < 16:
                continue
            y, x = np.nonzero(mask)
            observations.append(dict(view=vid, reference=reference_id, epoch=meta['epoch'], frame_id=meta['frame_id'],label=component_labels[component],
                timestamp_s=meta['timestamp_s'], mask=mask, pixels=pixels, changed_pixels=changed,
                bbox=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                reference_bbox=[int(uv[:,0].min()), int(uv[:,1].min()), int(uv[:,0].max()+1), int(uv[:,1].max()+1)],
                mesh_instance_ids=[int(i) for i in np.unique(query['segmentation_instance'][mask]) if i>0],
                location_proxy=np.median(points,axis=0)))
    groups = []
    for row in sorted(observations, key=lambda r:(r['epoch'],r['timestamp_s'], -r['changed_pixels'])):
        matches = [g for g in groups if g[0]['epoch'] == row['epoch'] and g[0]['label'] == row['label']
                   and abs(g[-1]['timestamp_s']-row['timestamp_s']) <= 15
                   and np.linalg.norm(np.median([r['location_proxy'] for r in g],axis=0)-row['location_proxy']) <= 1.0]
        if matches:
            matches[0].append(row)
        else:
            groups.append([row])
    retained = []
    for group in groups:
        best = {}
        for row in group:
            key = row['frame_id']
            if key not in best or row['changed_pixels'] > best[key]['changed_pixels']:
                best[key] = row
        if len(best) >= prepared['settings']['min_frames']:
            retained.append(sorted(best.values(), key=lambda r:-r['changed_pixels']))
    return retained
