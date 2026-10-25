"""Display the frozen baseline and cached, bidirectional SimSaC image pairs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from prepare import read, sha, write
from report_pair_models import overlay
from run_pair_models import project_probability


def regions(flags, valid, excluded=None):
    candidate = (flags & 8) != 0
    deferred = ((flags & 2) != 0) & ~candidate
    context = ((flags & 4) != 0) & ~candidate & ~deferred
    suppressed = ((flags & 1) != 0) & ~candidate & ~deferred & ~context
    masks = dict(candidate=candidate, deferred=deferred, group_context=context,
                 suppressed=suppressed, unaudited=valid & (flags == 0), no_mesh=~valid)
    if excluded is not None:
        masks = {k: mask & ~excluded for k, mask in masks.items()}
        masks['person_excluded'] = excluded
    assert sum(int(mask.sum()) for mask in masks.values()) == valid.size
    return masks


def measure(mask, areas, person):
    counts = {key: int((mask & area).sum()) for key, area in areas.items()}
    assert sum(counts.values()) == int(mask.sum())
    return dict(pixels=int(mask.sum()), fraction=float(mask.mean()), by_baseline=counts,
                on_person_pixels=int((mask & person).sum()),
                disagreement_with_baseline_pixels=int((mask ^ areas['candidate']).sum()))


def build(root):
    root = Path(root)
    out = root / 'focus-comparison-20261025'
    source = root / 'model-comparison-20261025'
    saved = read(out / 'comparison.json')
    manifest = read(source / 'inputs/manifest.json')
    rgb = np.load(source / 'inputs/rgb.npz')
    projection = np.load(out / 'baseline-projections.npz')
    lookup = {v['key']: v for v in saved['views']}
    views = {}
    for key, view in lookup.items():
        assert hashlib.sha256(rgb[key].tobytes()).hexdigest() == view['rgb_sha256']
        views[key] = {k: view[k] for k in ['key','zone','epoch','frame','yaw','erp','rgb_sha256',
                                         'baseline','baseline_filtered','person','map_error_u',
                                         'records','visible_groups','visible_groups_filtered']}
    # Both directions were already inferred for every overlap-selected pair.
    directed = {(v['key'], ref) for v in manifest['views'] for ref in v['refs']}
    assert all((b, a) in directed for a, b in directed)
    pairs = sorted((a, b) for a, b in directed if lookup[a]['epoch'] == 0)
    assert len(pairs) * 2 == len(directed)
    folder = out / 'pairs'
    folder.mkdir(exist_ok=True)
    evidence = {}
    result = dict(schema='baseline-simsac-before-after-v1', views=views, pairs=[],
                  groups=saved['groups'], baseline_summary=saved['baseline_summary'],
                  person_filter=saved['person_filter'], model=saved['models']['simsac'],
                  conditions=saved['conditions'], threshold=.5,
                  prediction='Cached SimSaC prediction for the selected pair, in each direction; no multi-reference averaging',
                  before_epoch=0, after_epoch=1)
    color = np.array([203, 153, 255])
    for before_key, after_key in pairs:
        before, after = lookup[before_key], lookup[after_key]
        assert before['epoch'] == 0 and after['epoch'] == 1 and before['zone'] == after['zone']
        pair_id = before_key + '--' + after_key
        row = dict(id=pair_id, zone=before['zone'], before=before_key, after=after_key, modes={})
        for mode in ['view', 'erp']:
            row['modes'][mode] = {}
            for side, query, reference in [('before', before, after), ('after', after, before)]:
                qa, rb = (query['key'], reference['key']) if mode == 'view' else (query['erp'], reference['erp'])
                path = source / 'results/simsac/pairs' / f'{mode}__{qa}__{rb}.npz'
                assert path.exists(), path
                evidence[str(path)] = sha(path)
                probability = np.load(path)['prob'].astype('f4')
                if mode == 'erp':
                    probability = project_probability(probability, query['yaw'])
                assert probability.shape == (224, 224) and np.isfinite(probability).all()
                key = query['key']
                raw = probability > .5
                person = projection['person__' + key]
                areas = regions(projection['flags__' + key], projection['valid__' + key])
                item = dict(raw=measure(raw, areas, person), filtered=None,
                            prediction_file=path.relative_to(source).as_posix())
                masks = {'raw': raw}
                if query['person']['available']:
                    filtered = raw & ~person
                    assert not (filtered & person).any()
                    assert np.array_equal(filtered[~person], raw[~person])
                    assert raw.sum() == filtered.sum() + (raw & person).sum()
                    masks['filtered'] = filtered
                    filtered_areas = regions(projection['flags__' + key], projection['valid__' + key], person)
                    item['filtered'] = dict(measure(filtered, filtered_areas, person),
                                            removed_person_pixels=int((raw & person).sum()))
                for state, mask in masks.items():
                    Image.fromarray(overlay(rgb[key], mask, color)).save(folder / f'{pair_id}-{mode}-{side}-{state}.png')
                row['modes'][mode][side] = item
        result['pairs'].append(row)
    write(out / 'before-after.json', result)
    write(out / 'before-after-validation.json', dict(
        pairs=len(pairs), directed_pairs=len(directed),
        model='SimSaC', model_inference_rerun=False, selected_pair_predictions=True,
        before_epoch=0, after_epoch=1, opposite_epochs_verified=True,
        cache_dtype='float16', threshold=.5, filtered_person_overlap_pixels=0,
        outside_person_exclusion_unchanged=True,
        source_hashes={str(source / 'inputs/manifest.json'): sha(source / 'inputs/manifest.json'),
                       str(out / 'comparison.json'): sha(out / 'comparison.json'),
                       str(out / 'baseline-projections.npz'): sha(out / 'baseline-projections.npz')},
        prediction_hashes=evidence))
    (out / 'BEFORE_AFTER.md').write_text(
        '# 기존 정합오차·관측검증 / SimSaC 전후 비교\n\n'
        f'기하 겹침으로 고정했던 {len(pairs)}개 사진 쌍을 전(Dorm2)·후(Dorm3)로 나란히 표시한다. '
        '두 촬영은 서로 다른 층이며 시간 변화 정답이 아니다.\n\n'
        '기존 판정은 전체 촬영에서 저장된 후보를 각 관측 RGB에 투영한 결과다. '
        'SimSaC는 선택한 두 사진의 저장된 양방향 예측을 각각 사용한다. '
        '앞선 다중 상대 평균 결과와 구분하며 새 추론은 실행하지 않았다.\n\n'
        '방향 RGB와 ERP 전체 입력 후 방향 투영을 전환할 수 있다. '
        '캐시 확률(float16)을 읽어 0.5 초과를 표시한다. '
        '사람 제외는 두 화면의 대상 프레임 마스크를 각각 사용하고, 원본 후보와 기존 객체 판정은 유지한다.\n\n'
        '사람 제외로 달라진 장면은 전·후 중 하나라도 SimSaC 후보가 사람 제외 영역과 겹쳤던 쌍이다. '
        '해당 장면을 선택해도 전후 모두와 두 방식의 객체 후보를 표시한다.\n\n'
        '전후 후보 픽셀의 단순 차이를 물리적 추가·소실·이동으로 판정하지 않는다. '
        '원본 RGB·전체 ERP·사람 제외 영역·기존 보류 이유도 전후로 확인할 수 있다.\n\n'
        'before-after.json에 쌍별 수치, before-after-validation.json에 입력·캐시 해시와 연결 검증을 보관했다.\n',
        encoding='utf-8')
    for src, dst in [('before_after.html', 'cropped-review.html'), ('before_after.js', 'before_after.js')]:
        shutil.copy2(Path(__file__).with_name(src), out / dst)
    print(f'Before/after review: {len(pairs)} pairs, cached SimSaC predictions in both directions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/scenediff-mesh-rgb-20261024'))
    build(parser.parse_args().root)
