"""Audit the real render/probability/mask bridge; no model execution required."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def validate(directory,metadata_only=False):
    out=Path(directory);data=read(out/'gaussian-data.json');pred=read(out/'simsac-candidates.json');model=read(out/'simsac-model.json')
    assert len(data['scenes'])==2 and data['scenes'][0]['video_sha256']!=data['scenes'][1]['video_sha256']
    assert data['scenes'][0]['checkpoint_sha256']!=data['scenes'][1]['checkpoint_sha256']
    assert len(data['zones'])==13 and len(data['baseline_boxes'])==2
    assert pred['source']=='gaussian' and pred['threshold']==.5
    assert model['loaded_strict'] and model['checkpoint_sha256']=='55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a'
    for epoch,phase in enumerate(['before','after']):
        assert sum(a['gaussians'] for a in data['models'][phase])==data['scenes'][epoch]['gaussians']
        assert sum(data['region_membership_counts'][phase].values())==data['scenes'][epoch]['gaussians']
        for asset in data['models'][phase]:
            m=np.array(asset['matrix']);s=np.cbrt(np.linalg.det(m[:3,:3]));R=m[:3,:3]/s
            assert s>0 and np.allclose(R.T@R,np.eye(3),atol=1e-6)
            if not metadata_only:assert sha(out/asset['file'])==asset['sha256']
    for zone in data['zones']:
        zid=zone['id'];cams=[]
        for phase in ['before','after']:
            record=data['renders'][zid][phase];assert sha(out/record['image'])==record['image_sha256']
            cams.append(np.array(record['camera_to_common_cv']))
            stats=pred['zones'][zid]['stats'][phase]
            packed=np.array(Image.open(out/f'renders/{zid}/{phase}-candidates.png'))
            raw,person,observable=(packed[...,i]>127 for i in range(3))
            assert int(raw.sum())==stats['raw_pixels']
            assert int((raw&observable).sum())==stats['observable_without_person_filter']
            assert int((raw&observable&person).sum())==stats['removed_person_pixels']
            assert int((raw&observable&~person).sum())==stats['filtered_pixels']
            assert stats['raw_pixels']==sum(stats[k] for k in ['outside_region_pixels','unobserved_pixels','removed_person_pixels','filtered_pixels'])
            if not metadata_only:
                p=out/stats['probability_file'];assert sha(p)==stats['probability_sha256'];prob=np.load(p)['prob'];assert np.isfinite(prob).all();assert np.array_equal(prob>.5,raw)
        np.testing.assert_allclose(cams[0],cams[1],atol=1e-12)
        np.testing.assert_allclose(cams[0],zone['camera_to_common_cv'],atol=1e-12)
        for mode,key in [('raw','observable_without_person_filter'),('filtered','filtered_pixels')]:
            boxes=[b for b in pred['boxes'][mode] if b['zone']==zid]
            for phase in ['before','after']:assert sum(b[phase+'_pixels'] for b in boxes)==pred['zones'][zid]['stats'][phase][key]
        assert pred['self_controls'][zid]['candidate_pixels']==0
    for box in [*data['baseline_boxes'],*pred['boxes']['raw'],*pred['boxes']['filtered']]:
        bounds=np.array(box['bounds']);assert bounds.shape==(2,3) and np.isfinite(bounds).all() and np.all(bounds[1]>=bounds[0])
    if not metadata_only:
        for path,digest in data['source_hashes'].items():assert sha(path)==digest,path
        ctx_path=next(Path(p) for p in data['source_hashes'] if p.endswith('alignment-context.json'))
        base=ctx_path.parent;ctx=read(ctx_path);review=read(base/'automatic-lifecycle-v2/review.json');audit=read(base/'all-object-review-v1/face-audit.json')
        expected={entry for group in review['event_groups'] for entry in group['entry_ids']}
        assert {b['source_entry'] for b in data['baseline_boxes']}==expected
        for box in data['baseline_boxes']:
            entry=audit[box['source_entry']];assert box['original_face_ids']==entry['original_face_ids']
            mesh=np.load(base/f'new-inputs/labeled-mesh-{entry["epoch"]}.npz')
            points=mesh['p'][mesh['faces'][entry['original_face_ids']]].reshape(-1,3)
            if entry['epoch']:
                m=np.array(ctx['methods']['structure']['region_matrices_after_Geo'])[int(box['zone'][1:])-1]@np.array(ctx['base_matrix'])
                m[:3,3]+=ctx['methods']['structure']['errors']['regions'][box['zone']]['translation_bias_u']
                points=points@m[:3,:3].T+m[:3,3]
            np.testing.assert_allclose(box['bounds'],np.stack([points.min(0),points.max(0)]),atol=1e-7)
    result=dict(passed=True,metadata_only=metadata_only,distinct_original_videos=True,distinct_trained_maps=True,gaussians=[s['gaussians'] for s in data['scenes']],frozen_mesh_candidate_groups=len(data['baseline_boxes']),new_simsac_regions=len(pred['boxes']['filtered']),registered_camera_pairs=13,exact_raw_probability_masks_verified=not metadata_only,all_pixel_decompositions_verified=True,person_exclusion_verified=True,self_controls_zero=13,original_mesh_sources_unchanged_verified=not metadata_only)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-map-comparison-20261025'));parser.add_argument('--metadata-only',action='store_true');parser.add_argument('--save',type=Path)
    args=parser.parse_args();result=validate(args.directory,args.metadata_only)
    if args.save:args.save.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
