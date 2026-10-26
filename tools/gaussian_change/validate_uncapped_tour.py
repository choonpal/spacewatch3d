"""Audit route/pose/visibility links while preserving original SimSaC detections."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from prepare import read,sha,save
from prepare_keyframes import transform_camera
from build_tour_candidates import support
from link_segmented_objects import validate as validate_objects
from build_change_regions import validate as validate_change_regions


def validate(folder,metadata_only=False):
    data=read(folder/'gaussian-data.json');fixed=read(folder/'tour-candidates.json');routes=read(folder/'tour-routes.json')
    prediction=read(folder/'simsac-candidates.json');original=folder/fixed['source_prediction']
    assert sha(original)==fixed['source_prediction_sha256']==sha(folder/'simsac-candidates.json')
    assert fixed['boxes']==prediction['boxes']==read(original)['boxes']
    assert not fixed['new_inference'] and not fixed['candidate_geometry_changed']
    assert data['decisions']['new_simsac_inference'] is False
    assert data['keyframe_sources'][0]['tour_plan_sha256']=='a705fc726dea288bdc04566d725a760c802b9aa2a8ae22b6f8f06fed73d5663c'
    requests=read(folder/'source-requests/U3.json')
    assert sha(folder/'source-requests/U3.json')==data['keyframe_sources'][0]['render_requests_sha256']
    views=data['zones'];by_id={v['id']:v for v in views};assert set(by_id)=={f'K{i}' for i in range(1,22)}
    assert routes['total_nodes']==21 and routes['total_links']==20 and not routes['new_links_created']
    assert len(routes['groups'])==1 and len(routes['groups'][0]['segments'])==1
    assert len(routes['groups'][0]['segments'][0]['nodes'])==21
    for edge in routes['groups'][0]['links']:
        for i,key in [(0,'from'),(-1,'to')]:
            np.testing.assert_allclose(edge['polyline'][i],np.asarray(by_id[edge[key]]['camera_to_common_cv'])[:3,3],atol=1e-7)
    regions={r['id']:r for r in data['registration_zones']}
    visibility_counts={}
    for view in views:
        request=requests['requests'][view['source_request_index']]
        assert request['viewpoint_id']==view['viewpoint_id']
        np.testing.assert_array_equal(request['reference_from_camera'],view['source_reference_from_camera'])
        bridge=np.asarray(regions[view['registration_zone']]['native_to_common'][1])@np.asarray(view['source_reference_to_gaussian'])
        np.testing.assert_allclose(transform_camera(request['reference_from_camera'],bridge),view['camera_to_common_cv'],atol=1e-10)
        geometry={}
        for phase in ['before','after']:
            record=data['renders'][view['id']][phase]
            np.testing.assert_array_equal(record['camera_to_common_cv'],view['camera_to_common_cv'])
            assert sha(folder/record['image'])==record['image_sha256']
            with Image.open(folder/record['image']) as image:assert list(image.size)==record['size']
            if not metadata_only:
                assert sha(folder/record['geometry'])==record['geometry_sha256']
                with np.load(folder/record['geometry']) as archive:geometry[phase]={k:archive[k] for k in ['depth','alpha']}
        for mode in ['raw','filtered']:
            for box in fixed['boxes'][mode]:
                record=fixed['visibility'][view['id']][mode][box['id']]
                assert record['visible']==(max(record['support_rays'].values())>=3)
                if geometry:
                    counts={phase:support(box['bounds'],view['camera_to_common_cv'],g,data['height_u']) for phase,g in geometry.items()}
                    assert counts==record['support_rays']
        visibility_counts[view['id']]=[id for id,v in fixed['visibility'][view['id']]['filtered'].items() if v['visible']]
    reference=read(folder/fixed['source_asset_prefix']/'gaussian-data.json')
    for phase in ['before','after']:
        assert [(a['sha256'],a['matrix']) for a in data['models'][phase]]==[(a['sha256'],a['matrix']) for a in reference['models'][phase]]
        if not metadata_only:
            for asset in data['models'][phase]:assert sha(folder/asset['file'])==asset['sha256']
    assert [s['checkpoint_sha256'] for s in data['scenes']]==[s['checkpoint_sha256'] for s in reference['scenes']]
    result={'passed':True,'metadata_only':metadata_only,'keyframes':21,'stored_links':20,'connected_segments':1,
        'exact_source_keyframe_poses':True,'same_pose_render_pairs':21,'render_images_verified':42,
        'original_filtered_candidates':len(fixed['boxes']['filtered']),'candidate_id_bounds_pixels_unchanged':True,
        'new_simsac_inference':False,'registration_fitted_again':False,'gaussian_maps_unchanged':True,
        'visibility':visibility_counts,'depth_visibility_recomputed':not metadata_only,
        'physical_change_accuracy_verified':False}
    if (folder/'object-evidence.json').exists():
        result['object_evidence']=validate_objects(folder,metadata_only)
    if (folder/'change-regions.json').exists():
        result['shared_change_regions']=validate_change_regions(folder)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');p.add_argument('--save',type=Path)
    a=p.parse_args();r=validate(a.directory,a.metadata_only)
    if a.save:save(a.save,r)
    print(r)
