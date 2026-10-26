"""Fresh SimSaC inference at every supplied tour viewpoint, on both maps.

No baseline-positive mesh gates, no nearest-image pairing, no change-dependent
view selection. Geometry is used only for registration, observability and 3D
localization; the pretrained change model receives rendered RGB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from infer_simsac import clusters,person_mask,rays_for,read,save,sha
from prepare import FrozenPartition


def main(root):
    protocol=read(root/'protocol.json');out=root/'published'
    state=read(root/'render-status.json')
    if state.get('state')!='complete':
        raise RuntimeError('Render job must complete before inference')
    data=read(out/'gaussian-data.json')
    (out/'predictions').mkdir(exist_ok=True)
    save(root/'inference-status.json',{'state':'loading'})
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    sys.path.insert(0,'/workspace/dorm_registration_benchmark_20261019/image-lifecycle-v4/deps')
    from ultralytics import YOLO
    pair_root=Path('/workspace/pano_change_models_20261025')
    sys.path.insert(0,str(pair_root))
    from run_pair_models import SimSaC
    setup=read(pair_root/'setup-status.json')
    model=SimSaC(setup)
    assert model.audit['loaded_strict'] and model.audit['checkpoint_sha256']=='55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a'
    detector_path=Path('/workspace/dorm_registration_benchmark_20261019/image-lifecycle-v4/yolo11l-seg.pt')
    detector=YOLO(str(detector_path))
    save(out/'simsac-model.json',{**model.audit,'input_source':'Fresh same-pose renders of both Gaussian maps at all saved Task 5b tour viewpoints',
                                'person_checkpoint_sha256':sha(detector_path),'person_policy':'Six cube views; person segmentation OR 10%-margin box, union of both epochs',
                                'existing_change_candidates_used_for_selection':False})
    classifier=FrozenPartition(dict(np.load(root/'partition.npz')))
    weak={z for z,e in protocol['alignment_context']['methods']['structure']['errors']['regions'].items() if not e['supported']}
    W,H=protocol['decisions']['render_size'];rays=rays_for(W,H);height=protocol['height_u']
    result={'schema':'gaussian-keyframe-simsac-v1','source':'gaussian','threshold':.5,'alpha_threshold':.85,
            'zones':{},'boxes':{'raw':[],'filtered':[]},'self_controls':{},'frame_hashes':{},'view_selection_uses_change_candidates':False}
    start=time.time()
    for number,view in enumerate(protocol['views'],1):
        vid=view['id'];directory=out/'renders'/vid;pose=np.asarray(view['camera_to_common_cv']);rgb={};geom={};points={};people={};detections={}
        for phase in ['before','after']:
            record=data['renders'][vid][phase]
            np.testing.assert_allclose(record['camera_to_common_cv'],pose,atol=1e-12)
            path=out/record['image'];assert sha(path)==record['image_sha256']
            rgb[phase]=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            gpath=out/record['geometry'];assert sha(gpath)==record['geometry_sha256'];geom[phase]=dict(np.load(gpath))
            points[phase]=pose[:3,3]+(rays@pose[:3,:3].T)*geom[phase]['depth'][...,None]
            people[phase],detections[phase]=person_mask(detector,directory,phase,W,H,rays)
            result['frame_hashes'][record['image']]=record['image_sha256']
        person=people['before']|people['after'];visible=np.ones((H,W),bool)
        for g in geom.values():visible&=(g['alpha']>=.85)&np.isfinite(g['depth'])&(g['depth']>.02*height)&(g['depth']<40*height)
        stats={};selected={'raw':{},'filtered':{}}
        for phase,other in [('before','after'),('after','before')]:
            probability=model.predict(rgb[phase],rgb[other]);assert np.isfinite(probability).all()
            path=out/'predictions'/f'{vid}-{phase}.npz';np.savez_compressed(path,prob=probability.astype('f4'))
            raw=probability>.5;observable=raw&visible;filtered=observable&~person
            stats[phase]={'raw_pixels':int(raw.sum()),'outside_region_pixels':0,'unobserved_pixels':int((raw&~visible).sum()),
                          'removed_person_pixels':int((observable&person).sum()),'filtered_pixels':int(filtered.sum()),
                          'observable_without_person_filter':int(observable.sum()),
                          'probability_file':str(path.relative_to(out)),'probability_sha256':sha(path)}
            assert stats[phase]['raw_pixels']==stats[phase]['unobserved_pixels']+stats[phase]['removed_person_pixels']+stats[phase]['filtered_pixels']
            selected['raw'][phase]=points[phase][observable];selected['filtered'][phase]=points[phase][filtered]
            packed=np.zeros((H,W,4),'u1');packed[...,0]=raw*255;packed[...,1]=person*255;packed[...,2]=visible*255;packed[...,3]=255
            cv2.imwrite(str(directory/f'{phase}-candidates.png'),cv2.cvtColor(packed,cv2.COLOR_RGBA2BGRA))
        for phase in ['before','after']:
            control=model.predict(rgb[phase],rgb[phase])
            result['self_controls'][f'{vid}-{phase}']={'candidate_pixels':int((control>.5).sum()),'max_probability':float(control.max())}
        zone_boxes={}
        for mode in ['raw','filtered']:
            boxes=clusters(selected[mode]['before'],selected[mode]['after'],vid,height)
            for box in boxes:
                region='R%02d'%int(classifier.assign(np.asarray(box['center'])[None])[0])
                box.update(viewpoint_id=view['viewpoint_id'],registration_zone=region,registration_supported=region not in weak)
            result['boxes'][mode].extend(boxes);zone_boxes[mode]=len(boxes)
        result['zones'][vid]={'stats':stats,'boxes':zone_boxes,'same_camera_pose':True,'valid_overlap_pixels':int(visible.sum()),
                              'person_pixels':int(person.sum()),'person_detections':detections,'plan_group':view['plan_group'],
                              'anchor_registration_zone':view['registration_zone'],'anchor_registration_supported':view['registration_supported']}
        save(root/'inference-status.json',{'state':'inference','completed':number,'total':len(protocol['views']),'view':vid})
        print('KEYFRAME',vid,'valid',int(visible.sum()),'candidates',[stats[p]['filtered_pixels'] for p in ['before','after']],'boxes',zone_boxes,flush=True)
    save(out/'simsac-candidates.json',result)
    save(out/'keyframe-validation.json',{'views':len(protocol['views']),'same_pose_pairs':len(protocol['views']),
                                       'self_controls':len(result['self_controls']),
                                       'self_candidate_pixels':sum(c['candidate_pixels'] for c in result['self_controls'].values()),
                                       'keyframe_camera_checks':protocol['keyframe_camera_checks'],
                                       'view_selection_uses_change_candidates':False,'original_plan_publication_ready':False,
                                       'registration_fitted_again':False,'registration_unsupported_zones':sorted(weak),
                                       'all_gaussians_retained':[s['gaussians'] for s in data['scenes']],
                                       'exact_raw_probability_saved':True,'ground_truth_accuracy_claimed':False})
    save(root/'inference-status.json',{'state':'complete','seconds':time.time()-start,'views':len(protocol['views'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/workspace/gaussian_keyframes_20261026_v2'))
    args=parser.parse_args()
    try:main(args.root)
    except Exception:
        import traceback
        save(args.root/'inference-status.json',{'state':'error','traceback':traceback.format_exc()});raise
