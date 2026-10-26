"""Audit requested poses, same-pose RGB inputs and exact prediction accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare import read,sha,transform
from prepare_keyframes import transform_camera


def validate(directory,metadata_only=False):
    out=Path(directory);data=read(out/'gaussian-data.json');pred=read(out/'simsac-candidates.json')
    protocol=read(out/'protocol.json');model=read(out/'simsac-model.json')
    views=data['zones'];expected=sum(s['count'] for s in data['keyframe_sources'])
    assert len(views)==expected and len({v['id'] for v in views})==expected
    assert data['viewpoint_mode']=='tour_keyframes' and not data['baseline_boxes']
    assert not protocol['decisions']['existing_change_candidates_used_for_view_selection']
    assert not pred['view_selection_uses_change_candidates']
    assert model['loaded_strict'] and model['checkpoint_sha256']=='55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a'
    assert data['scenes'][0]['checkpoint_sha256']!=data['scenes'][1]['checkpoint_sha256']
    assert data['scenes'][0]['video_sha256']!=data['scenes'][1]['video_sha256']
    regions={z['id']:z for z in data['registration_zones']}
    source_requests={};axis_bridges={}
    for s in data['keyframe_sources']:
        if s.get('source_requests_file'):
            p=out/s['source_requests_file'];assert sha(p)==s['render_requests_sha256']
            source_requests[s['group']]=read(p)['requests']
    if not metadata_only:
        for s in data['keyframe_sources']:
            p=Path(s['render_requests']);assert sha(p)==s['render_requests_sha256']
            assert sha(s['tour_plan'])==s['tour_plan_sha256']
            source_requests[s['group']]=read(p)['requests']
            folder=p.parent.parent if p.parent.name=='plan' else p.parent
            assert sha(folder/'scene.json')==s['scene_sha256']
            capture=s['capture']
            bridge=next(Path(name) for name in data['source_hashes'] if Path(name).name=='mesh_to_glb.npy' and capture in Path(name).parts)
            cameras=bridge.with_name('original_poses_glb.npy')
            axis_bridges[capture]=np.load(bridge)
            planner=np.asarray(read(folder/'scene.json')['camera']['positions_m'])
            np.testing.assert_allclose(transform(planner,axis_bridges[capture]),np.load(cameras)[:,:3,3],atol=1e-7)
        for name,digest in data['source_hashes'].items():assert sha(name)==digest,name
    for epoch,phase in enumerate(['before','after']):
        assert sum(a['gaussians'] for a in data['models'][phase])==data['scenes'][epoch]['gaussians']
        for a in data['models'][phase]:
            if not metadata_only:assert sha(out/a['file'])==a['sha256']
    groups={s['group']:{'views':0,'positive_views':0,'before_pixels':0,'after_pixels':0,'overlap':[]} for s in data['keyframe_sources']}
    for view in views:
        vid=view['id'];pose=np.asarray(view['camera_to_common_cv'])
        matrix=np.asarray(regions[view['registration_zone']]['native_to_common'][view['anchor_epoch']])@np.asarray(view['source_reference_to_gaussian'])
        np.testing.assert_allclose(pose,transform_camera(view['source_reference_from_camera'],matrix),atol=1e-10)
        if view['plan_group'] in source_requests:
            request=source_requests[view['plan_group']][view['source_request_index']]
            assert request['viewpoint_id']==view['viewpoint_id']
            np.testing.assert_array_equal(request['reference_from_camera'],view['source_reference_from_camera'])
        if not metadata_only:
            np.testing.assert_array_equal(view['source_reference_to_gaussian'],axis_bridges[view['capture_id']])
        size=None;geometry=[]
        for phase in ['before','after']:
            record=data['renders'][vid][phase]
            np.testing.assert_allclose(record['camera_to_common_cv'],pose,atol=1e-12)
            assert sha(out/record['image'])==record['image_sha256']
            with Image.open(out/record['image']) as image:assert list(image.size)==record['size']
            packed=np.asarray(Image.open(out/f'renders/{vid}/{phase}-candidates.png'))
            raw,person,visible=[packed[...,i]>127 for i in range(3)]
            size=raw.size;s=pred['zones'][vid]['stats'][phase]
            assert raw.sum()==s['raw_pixels']
            assert (raw&~visible).sum()==s['unobserved_pixels']
            assert (raw&visible&person).sum()==s['removed_person_pixels']
            assert (raw&visible&~person).sum()==s['filtered_pixels']
            assert (raw&visible).sum()==s['observable_without_person_filter']
            assert visible.sum()==pred['zones'][vid]['valid_overlap_pixels']
            assert s['outside_region_pixels']==0
            if not metadata_only:
                p=out/s['probability_file'];assert sha(p)==s['probability_sha256']
                probability=np.load(p)['prob'];assert np.isfinite(probability).all()
                np.testing.assert_array_equal(probability>.5,raw)
                gpath=out/record['geometry'];assert sha(gpath)==record['geometry_sha256']
                geometry.append(dict(np.load(gpath)))
            assert pred['self_controls'][f'{vid}-{phase}']['candidate_pixels']==0
        if geometry:
            expected_visible=np.ones(visible.shape,bool)
            for g in geometry:
                expected_visible&=(g['alpha']>=.85)&np.isfinite(g['depth'])&(g['depth']>.02*data['height_u'])&(g['depth']<40*data['height_u'])
            np.testing.assert_array_equal(expected_visible,visible)
        for mode,key in [('raw','observable_without_person_filter'),('filtered','filtered_pixels')]:
            boxes=[b for b in pred['boxes'][mode] if b['zone']==vid]
            for phase in ['before','after']:
                assert sum(b[phase+'_pixels'] for b in boxes)==pred['zones'][vid]['stats'][phase][key]
        g=groups[view['plan_group']];s=pred['zones'][vid]
        g['views']+=1;g['before_pixels']+=s['stats']['before']['filtered_pixels'];g['after_pixels']+=s['stats']['after']['filtered_pixels']
        g['positive_views']+=s['stats']['before']['filtered_pixels']+s['stats']['after']['filtered_pixels']>0
        g['overlap'].append(s['valid_overlap_pixels']/size)
    for g in groups.values():
        overlap=g.pop('overlap');g.update(mean_valid_overlap=float(np.mean(overlap)),minimum_valid_overlap=float(min(overlap)))
    result={'passed':True,'metadata_only':metadata_only,'views':expected,'same_pose_pairs':expected,
            'source_request_poses_verified':len(source_requests)==len(data['keyframe_sources']),'all_input_image_hashes_verified':True,
            'exact_probabilities_verified':not metadata_only,'self_controls_zero':len(pred['self_controls']),
            'candidates_used_for_view_selection':False,'groups':groups,
            'filtered_view_regions':len(pred['boxes']['filtered']),'physical_changed_object_count_verified':False,
            'all_source_plans_publication_ready':False,'original_registration_unchanged':True}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');p.add_argument('--save',type=Path)
    a=p.parse_args();r=validate(a.directory,a.metadata_only)
    if a.save:a.save.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
