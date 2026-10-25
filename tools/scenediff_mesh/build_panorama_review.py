"""Full-ERP, pose-aware 360 review of the saved baseline and SimSaC pairs."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
import open3d as o3d
from PIL import Image
from prepare import read, write, sha, view_rotation
from build_focus_comparison import wrapped_box_mask


def erp_rays(width, height):
    y, x = np.mgrid[:height, :width]
    theta = ((x + .5) / width - .5) * 2 * np.pi
    phi = ((y + .5) / height - .5) * np.pi
    return np.stack([np.sin(theta)*np.cos(phi), np.sin(phi), np.cos(theta)*np.cos(phi)], -1)


def camera_common(pose, epoch, zone, ctx):
    matrix = np.eye(4)
    if epoch:
        matrix = np.array(ctx['methods']['structure']['region_matrices_after_Geo'])[int(zone[1:])-1] @ np.array(ctx['base_matrix'])
        matrix[:3, 3] += ctx['methods']['structure']['errors']['regions'][zone]['translation_bias_u']
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    rotation = matrix[:3, :3] @ pose[:3, :3] / scale
    center = matrix[:3, :3] @ pose[:3, 3] + matrix[:3, 3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert abs(np.linalg.det(rotation)-1) < 1e-5
    return dict(center=center.tolist(), rotation=rotation.tolist(), scale=scale)


def pixel_for_direction(direction, width, height):
    direction = np.array(direction, dtype=float, copy=True)
    direction /= np.linalg.norm(direction)
    u = (np.arctan2(direction[0], direction[2])/(2*np.pi)+.5)*width-.5
    v = (np.arcsin(np.clip(direction[1],-1,1))/np.pi+.5)*height-.5
    return int(np.round(u)) % width, int(np.clip(np.round(v),0,height-1))


def build(root, prior):
    root, prior = Path(root), Path(prior)
    out = root/'focus-comparison-20261025'
    target = out/'panoramas'
    target.mkdir(exist_ok=True)
    saved = read(out/'before-after.json')
    manifest = read(root/'model-comparison-20261025/inputs/manifest.json')
    erp_meta = {v['key']:v for v in manifest['erps']}
    ctx = read(prior/'alignment-context.json')
    review = read(prior/'automatic-lifecycle-v2/review.json')
    audit = read(prior/'all-object-review-v1/face-audit.json')
    detections = {(f['epoch'],f['frame']):f for f in read(prior/'image-lifecycle-newmesh-v1/detections.json')['frames']}
    tracks = {t['id']:t for t in review['tracks']}
    scenes, flags, poses, inputs = {}, {}, {}, {}
    for epoch in [0,1]:
        mesh = np.load(prior/f'new-inputs/labeled-mesh-{epoch}.npz')
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(mesh['p'].astype('f4')),o3d.core.Tensor(mesh['faces'].astype('u4')))
        scenes[epoch] = scene
        flags[epoch] = np.zeros(len(mesh['faces']), 'u1')
        poses[epoch] = {p['frame']:np.array(p['camera_to_world']) for p in read(prior/f'new-inputs/poses-{epoch}.json')['poses']}
        folder = root/f'inputs/sam31/dorm{epoch+2}'
        inputs[epoch] = {f['map_frame']:folder/f['file'] for f in read(folder/'input_manifest.json')['frames']}
    for record in review['records']:
        bit = dict(suppressed=1,deferred=2,group_context=4)[record['review']['decision']]
        for side in ['reference','source']:
            key = record.get(side)
            if key in audit:
                a = audit[key]; flags[a['epoch']][a['original_face_ids']] |= bit
    for group in review['event_groups']:
        for key in group['entry_ids']:
            a = audit[key]; flags[a['epoch']][a['original_face_ids']] |= 8
    full_rays, small_rays = erp_rays(1024,512), erp_rays(256,128)
    people, frame_data, raw_depth = {}, {}, {}
    provenance = [out/'before-after.json', root/'model-comparison-20261025/inputs/manifest.json',
                  root/'model-comparison-20261025/results/simsac/model.json',
                  prior/'alignment-context.json',prior/'automatic-lifecycle-v2/review.json',
                  prior/'all-object-review-v1/face-audit.json',prior/'image-lifecycle-newmesh-v1/detections.json']
    provenance += [prior/f'new-inputs/{kind}-{epoch}.{ext}' for epoch in [0,1]
                   for kind,ext in [('labeled-mesh','npz'),('poses','json')]]
    source_hashes = {str(p):sha(p) for p in provenance}
    all_views = saved['views']
    frame_views = {v['erp']:v for v in all_views.values()}
    for erp, view in sorted(frame_views.items()):
        epoch, frame = view['epoch'], view['frame']
        rgb_path = inputs[epoch][frame]
        assert sha(rgb_path) == erp_meta[erp]['source_sha256']
        source_hashes[str(rgb_path)] = sha(rgb_path)
        rgb_name = f'{erp}-rgb.jpg'
        Image.open(rgb_path).convert('RGB').resize((2048,1024),Image.Resampling.LANCZOS).save(target/rgb_name,quality=93)
        pose = poses[epoch][frame]
        def cast(directions):
            rays = np.concatenate([np.broadcast_to(pose[:3,3],directions.shape), directions@pose[:3,:3].T],-1)
            hit = scenes[epoch].cast_rays(o3d.core.Tensor(rays.astype('f4')),nthreads=4)
            depth = hit['t_hit'].numpy(); triangle = hit['primitive_ids'].numpy()
            valid = np.isfinite(depth)&(depth>.04*ctx['height_u'])&(depth<8*ctx['height_u'])&(triangle<len(flags[epoch]))
            return np.where(valid,depth,0).astype('f4'), triangle, valid
        depth, triangle, valid = cast(full_rays)
        mask_flags = np.zeros(valid.shape,'u1');mask_flags[valid]=flags[epoch][triangle[valid]]
        person = np.zeros(valid.shape,bool)
        available = (epoch,frame) in detections
        for obj in detections.get((epoch,frame),{}).get('objects',[]):
            if obj['class_name']!='person':continue
            path=prior.parent/obj['mask_url']
            if not path.exists():available=False;continue
            source_hashes[str(path)] = sha(path)
            mask=np.asarray(Image.open(path).convert('L'))>127
            x0,y0,x1,y1=obj['bbox_unwrapped_px'];mx,my=(x1-x0)*.1,(y1-y0)*.1
            mask |= wrapped_box_mask(mask.shape,[x0-mx,y0-my,x1+mx,y1+my])
            person |= np.asarray(Image.fromarray(mask).resize((1024,512),Image.Resampling.NEAREST))
        people[erp] = person
        rgba = np.zeros((512,1024,4),'u1');rgba[...,0]=mask_flags;rgba[...,1]=person*255;rgba[...,3]=255
        Image.fromarray(rgba).save(target/f'{erp}-baseline.png')
        low_depth,_,_=cast(small_rays)
        low_depth.astype('<f4').tofile(target/f'{erp}-depth.bin')
        raw_depth[erp] = low_depth
        frame_data[erp] = dict(epoch=epoch,frame=frame,rgb=f'panoramas/{rgb_name}',
            baseline=f'panoramas/{erp}-baseline.png',depth=f'panoramas/{erp}-depth.bin',
            depth_width=256,depth_height=128,person_available=available,
            person_pixels=int(person.sum()),candidate_pixels=int(((mask_flags&8)!=0).sum()),
            candidate_pixels_filtered=int((((mask_flags&8)!=0)&~person).sum()),
            source_sha256=erp_meta[erp]['source_sha256'])
        print('360 frame',erp,flush=True)
    # Remove duplicate direction crops of the same two full ERP frames.
    unique={}
    for old_pair in saved['pairs']:
        b,a=[all_views[old_pair[s]] for s in ['before','after']]
        unique.setdefault((old_pair['zone'],b['erp'],a['erp']),(b,a))
    result=dict(schema='registered-360-baseline-simsac-v1',frames=frame_data,cases=[],
                height_reference='Frozen alignment-context.height_u, a scene vertical reference, not camera height',
                height_u=ctx['height_u'],threshold=.5,projection='Complete ERP sphere, not a 224x224 crop',
                displayed_rgb_resolution=[2048,1024],model_resolution=[1024,512],
                model=saved['model'],conditions=saved['conditions'],
                alignment='Saved region Sim3 and translation bias; same 3D point looking directions, not identical camera positions')
    predictions={}
    for (zone,bkey,akey),(b,a) in unique.items():
        cameras={side:camera_common(poses[v['epoch']][v['frame']],v['epoch'],zone,ctx)
                 for side,v in [('before',b),('after',a)]}
        groups=[g for g in saved['groups'] if g['zone']==zone]
        if groups:
            anchor=np.array(tracks[groups[0]['entries'][0]]['center'],float)
            anchor_kind='existing_candidate_location'
        else:
            forward=np.array(cameras['after']['rotation'])@view_rotation(np.deg2rad(a['yaw']))[:,2]
            local=np.array(cameras['after']['rotation']).T@forward
            u,v=pixel_for_direction(local,256,128)
            distance=float(raw_depth[akey][v,u])*cameras['after']['scale']
            if distance<=0:distance=2*ctx['height_u']
            anchor=np.array(cameras['after']['center'])+forward*distance
            anchor_kind='registered_mesh_view_center'
        case=dict(id=f'{zone}--{bkey}--{akey}',zone=zone,before=bkey,after=akey,cameras=cameras,
                  anchor=anchor.tolist(),anchor_kind=anchor_kind,stats={},model_masks={},anchor_person={})
        for side,qkey,rkey in [('before',bkey,akey),('after',akey,bkey)]:
            path=root/'model-comparison-20261025/results/simsac/pairs'/f'erp__{qkey}__{rkey}.npz'
            assert path.exists();predictions[str(path)]=sha(path)
            raw=np.load(path)['prob'].astype('f4')>.5;person=people[qkey];filtered=raw&~person
            assert not(filtered&person).any()
            assert np.array_equal(raw[~person],filtered[~person])
            filename=f'{qkey}--{rkey}-simsac.png'
            texture=np.zeros((512,1024,4),'u1');texture[...,0]=raw*255;texture[...,1]=person*255;texture[...,3]=255
            Image.fromarray(texture).save(target/filename)
            case['model_masks'][side]='panoramas/'+filename
            case['stats'][side]=dict(raw_pixels=int(raw.sum()),filtered_pixels=int(filtered.sum()),
                                     removed_person_pixels=int((raw&person).sum()))
            direction=np.array(cameras[side]['rotation']).T@(anchor-np.array(cameras[side]['center']))
            u,v=pixel_for_direction(direction,1024,512);case['anchor_person'][side]=bool(person[v,u])
            # The same registered point must reconstruct through each camera.
            back=np.array(cameras[side]['rotation'])@direction+np.array(cameras[side]['center'])
            assert np.allclose(back,anchor,atol=1e-6)
        case['camera_gap_u']=float(np.linalg.norm(np.array(cameras['before']['center'])-np.array(cameras['after']['center'])))
        case['camera_gap_h']=case['camera_gap_u']/ctx['height_u']
        case['has_existing_candidate']=bool(groups)
        result['cases'].append(case)
    result['cases'].sort(key=lambda c:c['camera_gap_u'])
    for path,digest in source_hashes.items():assert sha(path)==digest,path
    write(out/'panorama-review.json',result)
    write(out/'panorama-validation.json',dict(frames=len(frame_data),cases=len(result['cases']),
        height_reference=result['height_reference'],
        complete_sphere=True,display_resolution=[2048,1024],model_inference_rerun=False,
        original_mesh_and_decisions_unchanged=True,camera_rotations_orthonormal=True,
        shared_anchor_camera_roundtrip=True,filtered_person_candidate_pixels=0,
        source_hashes={**source_hashes,str(prior/'alignment-context.json'):sha(prior/'alignment-context.json'),
                       str(prior/'automatic-lifecycle-v2/review.json'):sha(prior/'automatic-lifecycle-v2/review.json')},
        prediction_hashes=predictions))
    (out/'PANORAMA_REVIEW.md').write_text(
        '# 원본 360° 전후 비교 · 기존 정합오차 / SimSaC\n\n'
        f'원본 ERP {len(frame_data)}프레임을 전체 2048×1024로 표시한다. '
        f'방향 뷰의 중복을 제외한 구역별 {len(result["cases"])}개 전후 쌍을 사용한다. '
        '기존 224×224 방향 뷰는 cropped-review.html에 별도로 보관했다.\n\n'
        '저장된 구역 Sim3·이동 편향·원본 카메라 포즈로 같은 3D 지점을 바라보도록 회전을 맞춘다. '
        '카메라 위치는 유지한다. 깊이 표면이 없는 방향은 공통 좌표계 방향으로 연결한다. '
        '촬영 위치·정합 잔차·가림에 따른 원근 차이는 남으며 픽셀 단위 영상 정합이나 공통 시점의 새 RGB 생성을 주장하지 않는다.\n\n'
        '전후 카메라 간격이 작은 순으로 표시한다. 간격은 맵 단위와 기준높이 대비 비율이며 객체 이동 거리가 아니다. '
        '360° 전체 펼침, 드래그 시야 연동, 같은 지점 확대, 원본 전체 이미지 열기를 지원한다.\n\n'
        'SimSaC는 전체 ERP(1024×512) 쌍의 기존 양방향 캐시를 사용한다. 새 모델 추론이나 재학습은 없다. '
        '원본 확률과 기존 객체 판정은 변경하지 않으며 사람 분할·여유를 둔 검출 박스의 제외 영역을 두 방법에 동일 적용한다.\n\n'
        'Dorm2·Dorm3는 서로 다른 층의 공간 차이 파일럿이다. 관측량이 같은 정확도 비교나 시간 변화 정답이 아니다.\n',encoding='utf-8')
    # Preserve the earlier crop review as an explicitly named supplementary page.
    shutil.copy2(Path(__file__).with_name('before_after.html'),out/'cropped-review.html')
    for src,dst in [('panorama_review.html','index.html'),('panorama_review.js','panorama_review.js'),
                    ('panorama_canvas.js','panorama_canvas.js')]:
        shutil.copy2(Path(__file__).with_name(src),out/dst)
    print('360 review ready',len(frame_data),'frames',len(result['cases']),'cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024'))
    parser.add_argument('--prior',type=Path,default=Path('outputs/registration-benchmark-20261019/new-meshfirst-20261022'))
    args=parser.parse_args();build(args.root,args.prior)
