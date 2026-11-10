"""Receipted input adapter to the exact requested nearby-label-rich RGB method.

The selection and detector implementations are copied from the user-designated
server producer. Only current plan/dependency/file bindings are adapted here.
"""
import copy
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .io import read,save,result,sha,verify

SOURCE='/workspace/dorm45_keypoint_labels_20261028_vf6q7euv'

def validate_options(config):
    o=config['changes']['keypoint_nearby']
    if not isinstance(o,dict) or set(o)!={'anchor_capture','registration','registration_inputs'}:
        raise ValueError('Explicit anchor capture and unchanged-input registration receipt required')
    if o['anchor_capture'] not in {c['id'] for c in config['captures']}:
        raise ValueError('Missing keypoint anchor capture')
    verify(o['registration'])
    verify(o['registration_inputs'])
    r=read(o['registration']['path'])
    if not r.get('fresh_structural') or r.get('state')!='passed_geometric_gate':
        raise ValueError('Invalid original structural registration; no nearest-time fallback')

def select_plan(catalog, plans, registration, anchor_capture, work):
    from tools.mesh_simsac.label_rich_frames import select_label_rich
    from tools.mesh_simsac.regional import FreshRegionalAlignment
    regional=FreshRegionalAlignment(registration['fresh_structural'])
    U=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
    radius=.65*registration['fresh_structural']['height_u']
    epoch=next(i for i,p in enumerate(plans) if p['capture_id']==anchor_capture)
    selected=[{},{}];rows=[]
    views=sorted(plans[epoch]['viewpoints'],key=lambda v:v['source_frame']['pose_index'])
    for i,view in enumerate(views):
        kid=f'{anchor_capture}-K{i+1:02d}';native=U@np.asarray(view['reference_from_camera'])
        region=int(regional.assign(native[None,:3,3],epoch)[0])
        matrix=np.asarray(registration['region_matrices'][epoch][region])@np.asarray(registration['common_from_world'][epoch])
        target=(matrix@native)[:3,3]
        row=dict(id=kid,anchor_capture=anchor_capture,keypoint=i+1,viewpoint_id=view['viewpoint_id'],region=f'R{region:02d}',
                 common_position=target.tolist(),native_position=np.asarray(view['reference_from_camera'])[:3,3].tolist(),
                 nearby_radius_u=radius,registration_supported=registration['fresh_structural']['regions'][region-1]['supported'],sides=[])
        for e,records in enumerate(catalog):
            matrix=np.asarray(registration['region_matrices'][e][region])@np.asarray(registration['common_from_world'][e])
            candidates=[dict(r,common_position=(matrix@np.asarray(r['world_from_camera']))[:3,3].tolist()) for r in records]
            chosen=select_label_rich(candidates,target,radius,3)
            for r in chosen:
                rec=selected[e].setdefault(r['id'],dict(records[int(r['id'])],keypoint_ids=[]))
                rec['keypoint_ids'].append(kid)
            row['sides'].append(dict(capture=plans[e]['capture_id'],status='selected' if len(chosen)>=2 else 'insufficient_nearby_frames',
                frames=chosen,nearest_available_distance_u=min(np.linalg.norm(np.asarray(r['common_position'])-target) for r in candidates)))
        rows.append(row)
    return dict(schema='keypoint-nearby-rgb-v1',work=str(work),selection='nearby camera positions; object label type count first, then object label pixels, sharpness, distance; no per-label quota',
        frame_budget_per_capture_per_keypoint=3,nearby_radius_u=radius,units='map units; metric scale not independently verified',
        anchor_count=len(rows),keypoints=rows,unique_frames=[len(s) for s in selected],selected_frames=[list(s.values()) for s in selected],
        time_stride_used=False,known_box_reference_used=False,selection_policy='label_type_count_first',min_label_pixels=32,
        photographer_excluded_from_label_counts=True,source_producer=SOURCE)

def rgb_inputs(job):
    from . import mesh_rgb
    validate_options(job['config']);out=Path(job['output']);cfg=job['config'];deps=job['dependencies'];o=cfg['changes']['keypoint_nearby']
    # Existing exact original export associates own same-frame labels to first-hit
    # original mesh surfaces. Retain all anchors for the source frame selector.
    catalogjob=copy.deepcopy(job);catalogjob['output']=str(out/'catalog');catalogjob['config']['changes'].pop('keypoint_nearby')
    catalogjob['config']['changes'].update(frame_stride=1,native_rgb=False)
    (out/'catalog').mkdir();mesh_rgb.rgb_inputs(catalogjob)
    folder=out/'catalog/inputs';manifest=read(folder/'manifest.json');reg=read(o['registration']['path'])
    # The producer's frozen registration is permitted ONLY for identical meshes,
    # pointclouds AND input transforms. A new video/geometry requires fresh fitting.
    registration_inputs=Path(o['registration_inputs']['path']);source=registration_inputs.parent
    old=read(registration_inputs)
    binding={};catalog=[]
    for e,phase in enumerate(['before','after']):
        spec=manifest[phase];prior=old[phase];binding[spec['id']]={}
        if spec['id']!=prior['id']:raise ValueError('Registration capture mismatch')
        for key in ['mesh','pointcloud']:
            if sha(folder/spec[key])!=sha(source/prior[key]):raise ValueError('Registration input geometry changed')
            binding[spec['id']][key]=sha(folder/spec[key])
        for key in ['world_from_mesh','world_from_pointcloud','segmentation_labels']:
            if spec[key]!=prior[key]:raise ValueError('Registration coordinate/class binding changed')
        records=[]
        for f in spec['frames']:
            with np.load(folder/f['segmentation']) as z:cls=z['class_id'];person=z['person']
            counts=np.bincount(cls[~person].ravel(),minlength=256)
            rgb=np.asarray(Image.open(folder/f['image']).convert('L'),dtype=float)
            lap=-4*rgb+np.roll(rgb,1,0)+np.roll(rgb,-1,0)+np.roll(rgb,1,1)+np.roll(rgb,-1,1)
            records.append(dict(f,label_counts={name:int(counts[i+1]) for i,name in enumerate(spec['segmentation_labels'])},
                                sharpness=float(lap.var()),source_image=str(folder/f['image']),source_image_sha256=f['sha256']))
        catalog.append(records)
    plans=[read(deps[c['id']+'.plan']['plan']) for c in cfg['captures']]
    plan=select_plan(catalog,plans,reg,o['anchor_capture'],out)
    plan['source_plan_sha256']={c['id']:sha(deps[c['id']+'.plan']['plan']) for c in cfg['captures']}
    save(out/'keypoint-plan.json',plan)
    for e,phase in enumerate(['before','after']):
        selected={r['id']:r for r in plan['selected_frames'][e]}
        manifest[phase]['frames']=[dict(f,keypoint_ids=selected[f['id']]['keypoint_ids']) for f in manifest[phase]['frames'] if f['id'] in selected]
    shutil.copy2(o['registration']['path'],folder/'frozen-registration.json')
    manifest['settings'].update(registration='provided',frame_stride=1)
    manifest['frozen_structural_registration']=dict(path='frozen-registration.json',sha256=sha(folder/'frozen-registration.json'),input_hashes=binding)
    manifest['keypoint_plan']=dict(schema=plan['schema'],sha256=sha(out/'keypoint-plan.json'),anchor_count=plan['anchor_count'],
        keypoint_ids=[k['id'] for k in plan['keypoints']],nearby_radius_u=plan['nearby_radius_u'],canonical_anchor_capture=o['anchor_capture'])
    manifest['scope']='Current selected keypoints; exact original nearby camera frames; shared keypoint-only opposite references'
    save(folder/'manifest.json',manifest)
    if cfg['changes'].get('native_rgb',True):
        manifest=mesh_rgb.materialize_native_frames(folder,out/'catalog/adapter.json',cfg['runtime'])
    save(out/'input-lineage.json',dict(source_producer=SOURCE,source_registration=o['registration'],
        unchanged_geometry_and_transforms_verified=True,anchor_plan_sha256=plan['source_plan_sha256'],
        gaussian_inputs_used=False,old_predictions_imported=False,source_rgb_modified=False,new_pose_inference=False,
        class_specific_quotas=False,known_change_ids_used=False))
    result(out,manifest=str(folder/'manifest.json'),lineage=str(out/'input-lineage.json'),keypoint_plan=str(out/'keypoint-plan.json'),
        source_mode='original_rgb',frames=plan['unique_frames'],keypoints=plan['anchor_count'],source_producer=SOURCE)
