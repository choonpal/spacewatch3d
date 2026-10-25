"""Candidate boxes and observed-after RGB reprojection for a single before panorama.

This consumes the exported review, never model weights or source meshes. Boxes
are connected candidate regions, not semantic instances or verified changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy import ndimage


def sphere_rays(width, height):
    y, x = np.mgrid[:height, :width]
    longitude = ((x + .5) / width - .5) * 2 * np.pi
    latitude = ((y + .5) / height - .5) * np.pi
    return np.stack([np.sin(longitude)*np.cos(latitude), np.sin(latitude),
                     np.cos(longitude)*np.cos(latitude)], -1)


def project(source_camera, target_camera, directions, depth):
    """Project native source rays into target ERP; return rotation fallback flags."""
    source_rotation = np.asarray(source_camera['rotation'])
    target_rotation = np.asarray(target_camera['rotation'])
    common_directions = directions @ source_rotation.T
    fallback = ~np.isfinite(depth) | (depth <= 0)
    points = np.asarray(source_camera['center']) + common_directions * (
        np.where(fallback, 1, depth) * source_camera['scale'])[..., None]
    vectors = points - np.asarray(target_camera['center'])
    # With no surface, keep the common direction and explicitly record fallback.
    vectors[fallback] = common_directions[fallback]
    local = vectors @ target_rotation
    local /= np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1e-12)
    u = (np.arctan2(local[..., 0], local[..., 2])/(2*np.pi)+.5) % 1
    v = np.arcsin(np.clip(local[..., 1], -1, 1))/np.pi+.5
    return u, v, fallback


def nearest(array, u, v):
    h, w = array.shape[:2]
    x = np.floor(u*w).astype(int) % w
    y = np.clip(np.floor(v*h).astype(int), 0, h-1)
    return array[y, x]


def bilinear(array, u, v):
    h, w = array.shape[:2]
    x = u*w-.5
    y = np.clip(v*h-.5, 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = (x-x0)[..., None], (y-y0)[..., None]
    a, b = array[y0, x0 % w], array[y0, (x0+1) % w]
    c, d = array[np.minimum(y0+1, h-1), x0 % w], array[np.minimum(y0+1, h-1), (x0+1) % w]
    return np.clip((a*(1-tx)+b*tx)*(1-ty)+(c*(1-tx)+d*tx)*ty, 0, 255).astype('u1')


def seam_labels(mask, padding=3):
    """Group nearby pixels for boxes, wrap ERP seam, retain every original pixel."""
    expanded = ndimage.maximum_filter(mask, size=(2*padding+1, 2*padding+1),
                                     mode=('constant', 'wrap'))
    labels, total = ndimage.label(expanded, structure=np.ones((3, 3)))
    parents = np.arange(total+1)
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    for dy in [-1, 0, 1]:
        first, last = max(0, -dy), min(mask.shape[0], mask.shape[0]-dy)
        for left, right in zip(labels[first:last, 0], labels[first+dy:last+dy, -1]):
            if left and right:
                a, b = find(int(left)), find(int(right))
                parents[b] = a
    mapping = np.array([find(i) for i in range(total+1)])
    return np.where(mask, mapping[labels], 0)


def circular_bounds(xs, width):
    values = np.unique(xs)
    gaps = np.diff(np.r_[values, values[0]+width])
    start = values[(int(np.argmax(gaps))+1) % len(values)]
    unwrapped = (values-start) % width
    lo, hi = start, start+unwrapped.max()+1
    return ((lo+hi)/2/width) % 1, min(.499, max(6, (hi-lo)/2+3)/width)


def build(directory):
    out = Path(directory)
    data = json.loads((out/'panorama-review.json').read_text(encoding='utf-8'))
    target = out/'interactions'
    target.mkdir(exist_ok=True)
    width, height = data['model_resolution']
    rays = sphere_rays(width, height)
    frames, source_hashes = {}, {'panorama-review.json':hashlib.sha256((out/'panorama-review.json').read_bytes()).hexdigest()}
    def load(path):
        p = out/path
        source_hashes[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return np.asarray(Image.open(p))
    for key, frame in data['frames'].items():
        depth_path = out/frame['depth']
        source_hashes[frame['depth']] = hashlib.sha256(depth_path.read_bytes()).hexdigest()
        depth = np.fromfile(depth_path, dtype='<f4').reshape(frame['depth_height'], frame['depth_width'])
        depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.NEAREST))
        frames[key] = dict(rgb=load(frame['rgb'])[..., :3], mask=load(frame['baseline']), depth=depth)
    result = dict(schema='single-before-candidate-ghost-v1', cases={},
        box_meaning='Connected candidate regions grouped within 3 ERP pixels; no instance identity or change confirmation',
        ghost='Observed after RGB resampled onto before first-hit mesh surfaces using saved cameras. Missing before depth stays transparent; occlusions and registration residuals remain.',
        after_only='After candidates projected into before ERP with after native depth; missing depth uses explicitly counted rotation-only fallback',
        person_policy='Source person exclusion plus before target-person exclusion for projected after candidates',
        resolution=[width, height])
    covered = 0
    for case in data['cases']:
        before, after = frames[case['before']], frames[case['after']]
        bc, ac = case['cameras']['before'], case['cameras']['after']
        # Inverse mapping: sample original after photograph at every before surface.
        u, v, missing = project(bc, ac, rays, before['depth'])
        ghost_rgb = bilinear(after['rgb'], u, v)
        ghost_mask = np.zeros((height, width, 4), 'u1')
        ghost_mask[..., 0] = (~missing)*255
        ghost_mask[..., 1] = ((before['mask'][..., 1]>127) | (nearest(after['mask'][..., 1], u, v)>127))*255
        ghost_mask[..., 3] = 255
        name = case['id']
        Image.fromarray(ghost_rgb).save(target/f'{name}-after.jpg', quality=91)
        Image.fromarray(ghost_mask).save(target/f'{name}-valid.png')
        # Forward mapping is essential for regions found in the after image only.
        pu, pv, fallback = project(ac, bc, rays, after['depth'])
        px, py = np.floor(pu*width).astype(int) % width, np.clip(np.floor(pv*height).astype(int), 0, height-1)
        target_person = before['mask'][py, px, 1]>127
        sim = {s:load(case['model_masks'][s]) for s in ['before', 'after']}
        entry = dict(ghost_rgb=f'interactions/{name}-after.jpg',
                     ghost_mask=f'interactions/{name}-valid.png', regions={},
                     camera_gap_u=case['camera_gap_u'], ghost_depth_coverage=float((~missing).mean()))
        for mode in ['raw', 'filtered']:
            entry['regions'][mode] = {}
            for method in ['baseline', 'simsac']:
                masks = sim if method == 'simsac' else dict(before=before['mask'], after=after['mask'])
                selected = {s:((masks[s][..., 0]>127) if method=='simsac' else ((masks[s][..., 0]&8)>0)) for s in ['before', 'after']}
                if mode == 'filtered':
                    for s in selected:
                        selected[s] &= masks[s][..., 1]<128
                    selected['after'] &= ~target_person
                projected = np.zeros((height, width), bool)
                projected[py[selected['after']], px[selected['after']]] = True
                union = selected['before'] | projected
                labels = seam_labels(union)
                regions = []
                for label in np.unique(labels[union]):
                    ys, xs = np.where(labels == label)
                    cu, hu = circular_bounds(xs, width)
                    v0, v1 = max(0, (ys.min()-3)/height), min(1, (ys.max()+4)/height)
                    if (v1-v0)*height < 12:
                        mid = (v0+v1)/2
                        v0, v1 = max(0, mid-6/height), min(1, mid+6/height)
                    count_b = int(((labels==label)&selected['before']).sum())
                    source_a = selected['after'] & (labels[py, px]==label)
                    count_a = int(source_a.sum())
                    # A real source candidate provides the focus point, not padding.
                    if count_b:
                        points = np.argwhere((labels==label)&selected['before'])
                        dx = ((points[:, 1]+.5)/width-cu+.5)%1-.5
                        dy = (points[:, 0]+.5)/height-(v0+v1)/2
                        ay, ax = points[np.argmin(dx*dx+dy*dy)]
                        focus_cam, focus_ray, distance = bc, rays[ay, ax], before['depth'][ay, ax]
                    else:
                        points = np.argwhere(source_a)
                        dx = ((px[source_a]+.5)/width-cu+.5)%1-.5
                        dy = (py[source_a]+.5)/height-(v0+v1)/2
                        ay, ax = points[np.argmin(dx*dx+dy*dy)]
                        focus_cam, focus_ray, distance = ac, rays[ay, ax], after['depth'][ay, ax]
                    depth_valid = bool(np.isfinite(distance) and distance>0)
                    if not depth_valid: distance = 2*data['height_u']/focus_cam['scale']
                    anchor = np.asarray(focus_cam['center']) + np.asarray(focus_cam['rotation']) @ focus_ray * distance*focus_cam['scale']
                    region = dict(method=method, before_pixels=count_b, after_pixels=count_a,
                        bounds=[float(cu), float(hu), float(v0), float(v1)], anchor=anchor.tolist(),
                        focus_depth_valid=depth_valid, after_rotation_fallback_pixels=int((source_a&fallback).sum()),
                        ghost_coverage=float((~missing[labels==label]).mean()),
                        ghost_coverage_filtered=float(((~missing)&(ghost_mask[..., 1]<128))[labels==label].mean()))
                    regions.append(region)
                regions.sort(key=lambda r:r['before_pixels']+r['after_pixels'], reverse=True)
                for i, region in enumerate(regions): region['id']=f'{method}-{i+1}'
                # Every retained source candidate contributes to exactly one box.
                assert sum(r['before_pixels'] for r in regions)==int(selected['before'].sum())
                assert sum(r['after_pixels'] for r in regions)==int(selected['after'].sum())
                entry['regions'][mode][method] = regions
                covered += sum(r['before_pixels']+r['after_pixels'] for r in regions)
        result['cases'][name] = entry
        print('candidate boxes', name, sum(len(x) for x in entry['regions']['filtered'].values()), flush=True)
    (out/'panorama-interactions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validation = dict(cases=len(result['cases']), source_candidate_pixels_accounted_across_modes_and_methods=covered,
        raw_model_predictions_unchanged=True, original_geometry_unchanged=True,
        all_retained_source_candidates_have_boxes=True, ghost_uses_observed_rgb_only=True,
        missing_before_depth_transparent=True, source_hashes=source_hashes)
    for path, digest in source_hashes.items():
        assert hashlib.sha256((out/path).read_bytes()).hexdigest()==digest, path
    (out/'interaction-validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'PANORAMA_REVIEW.md').write_text(
        '# 360° 전 영상 후보 둘러보기\n\n'
        '처음에는 전 360° 영상 하나만 표시한다. 기본 목록은 사람 제외 후 선택한 방식의 후보가 남은 쌍이다. '
        '전체 쌍·사람 제외 비교 모드에서는 후보 0인 쌍도 확인할 수 있으며 전/후 후보 수를 목록에 표시한다. '
        'R01 전 133 ↔ 후 142의 SimSaC는 원본 전 0·후 156픽셀이며 후 156픽셀이 모두 사람 영역으로 제외된다.\n\n'
        '기존 정합오차·관측검증과 SimSaC만 선택한다. 전·후 양쪽 후보를 전 영상 좌표에 표시한다. '
        '후에서만 검출된 후보는 후의 첫 메쉬 깊이와 저장 포즈로 전 시점에 투영한다. 깊이가 없는 후 후보는 '
        '방향만 맞추는 근사 위치이며 개수를 기록한다. 사람 제외 시 후 원본 사람 영역과 투영 위치의 전 사람 영역을 제외한다.\n\n'
        '바운딩박스는 1024×512 ERP의 후보 연결 영역을 3픽셀 여유로 묶는다. 원본 후보를 새로 만들거나 '
        '확률을 바꾸지 않는다. 보존된 각 원본 후보 픽셀이 정확히 한 박스에 포함되는지 검사한다. '
        '박스는 semantic instance나 확정 객체 변화가 아니며 내부의 모든 픽셀이 변화 후보라는 뜻도 아니다.\n\n'
        '박스를 누르면 해당 박스 안에서만 후 RGB를 55% 겹쳐 표시한다. 전만 보기·반투명 겹침·선택 영역 후만 '
        '및 0–100% 슬라이더로 비교한다. 전 메쉬의 첫 표면 깊이와 저장된 구역 Sim3·카메라 포즈로 원본 후 RGB를 '
        '샘플링한다. 전 깊이가 없는 곳은 투명하다. 관측 RGB를 재투영할 뿐 새로운 내용을 생성하지 않는다. '
        '촬영 위치·가림·메쉬·정합 잔차는 남으며 반투명 경계 차이가 객체 이동의 증거는 아니다.\n\n'
        '원본 ERP 20프레임 전체에서 36쌍을 사용한다. 둘러보기 모드가 기본이고 드래그·방향키·확대·전체 화면을 지원한다. '
        'SimSaC는 전체 ERP 쌍의 저장된 양방향 추론이며 새 추론·재학습은 없다. 원본 예측·판정·메쉬를 바꾸지 않는다. '
        '이전 224px 방향 뷰는 cropped-review.html에 보관했다.\n\n'
        'Dorm2·Dorm3는 서로 다른 층의 공간 차이 파일럿이다. 정확도 순위나 시간 변화 정답을 제시하지 않는다. '
        'panorama-interactions.json과 interaction-validation.json에 박스·근사 투영·입력 해시를 기록한다.\n',encoding='utf-8')
    for source, dest in [('panorama_review.html','index.html'), ('panorama_review.js','panorama_review.js'),
                         ('panorama_canvas.js','panorama_canvas.js')]:
        shutil.copy2(Path(__file__).with_name(source), out/dest)
    print('Single-before candidate interactions ready', len(result['cases']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('outputs/scenediff-mesh-rgb-20261024/focus-comparison-20261025'))
    build(parser.parse_args().directory)
