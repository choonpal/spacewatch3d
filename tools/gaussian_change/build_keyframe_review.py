"""Publish the saved keyframe experiment beside the existing Gaussian viewer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare import read,save,sha


def build(run,destination):
    source=run/'published';destination.mkdir(parents=True,exist_ok=True)
    for p in source.rglob('*'):
        if not p.is_file() or 'models' in p.relative_to(source).parts:continue
        target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    data=read(source/'gaussian-data.json')
    for record in data['keyframe_sources']:
        name='source-requests/'+record['group']+'.json'
        target=destination/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(record['render_requests'],target)
        if sha(target)!=record['render_requests_sha256']:raise ValueError('Source request changed')
        record['source_requests_file']=name
    for phase in ['before','after']:
        for asset in data['models'][phase]:
            asset['file']='../gaussian-map-comparison-20261025/'+asset['file']
            if sha(destination/asset['file'])!=asset['sha256']:
                raise ValueError('Shared original Gaussian asset differs from this execution')
    save(destination/'gaussian-data.json',data)
    shutil.copy2(run/'protocol.json',destination/'protocol.json')
    shutil.copy2(run/'selection-audit.json',destination/'selection-audit.json')
    execution=run/'source/executed'
    if execution.exists():
        shutil.copytree(execution,destination/'execution-source',dirs_exist_ok=True)
        archive=run/'keyframe-results.tar.gz';archive_hash=None
        if archive.exists():
            digest=hashlib.sha256()
            with archive.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            archive_hash=digest.hexdigest()
        receipt={'server_run':read(run/'protocol.json')['remote_root'],
                 'render':read(run/'source/render-status.json'),'inference':read(run/'source/inference-status.json'),
                 'source_sha256':{p.name:sha(p) for p in sorted(execution.glob('*.py'))},
                 'source_protocol_sha256':sha(run/'source/protocol.json'),
                 'archive_sha256':archive_hash,
                 'registration_fitted_again':False,'initial_axis_bridge_attempt_rejected':True}
        save(destination/'execution-receipt.json',receipt)
    tools=Path(__file__).parent
    for name in ['viewer.js','viewer.css','tour.js','tour_navigation.js']:shutil.copy2(tools/name,destination/name)
    shutil.copy2(tools/'keyframes.html',destination/'index.html')
    from build_tour_routes import build as build_routes
    routes=build_routes(run,destination/'tour-routes.json')
    from build_tour_candidates import build as build_candidates
    fixed=build_candidates(run,destination.parent/'gaussian-map-comparison-20261025',destination/'tour-candidates.json')
    (destination/'.gitignore').write_text('predictions/*.npz\nrenders/*/*-geometry.npz\n',encoding='utf-8')
    large={}
    for p in destination.rglob('*.npz'):
        large[p.relative_to(destination).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    save(destination/'large-inputs.json',{'shared_models':'../gaussian-map-comparison-20261025/models',
                                       'server_run':read(run/'protocol.json')['remote_root'],
                                       'omitted_probability_and_depth_files':large})
    from validate_keyframes import validate
    report=validate(destination)
    save(destination/'local-verification.json',report)
    lines=['# 저장된 투어 키프레임 변화탐지 실험','',
           '실행일: 2026-10-26 (Asia/Seoul). 원본 모델·기존 정합·키프레임 계획을 수정하지 않고 별도 출력에 실행했다.','',
           '## 시점과 정합','',
           '- 서버의 기준 Dorm2 13개, 기준 Dorm3 13개, 최신 Dorm3 통계 실험 20개 요청을 모두 사용했다. 기존 변화 후보를 시점 선택에 사용하지 않았다.',
           '- 각 요청의 가상 카메라 위치·방향을 기존 전역·구역별 정합 및 이동 편향으로 공통 프레임에 변환하고, 동일한 포즈에서 두 맵을 렌더링했다. 가까운 반대 회차 키프레임으로 대체하지 않았다.',
           '- 계획의 원본 촬영 카메라 233 / 248개와 현재 Gaussian 모델의 native 카메라 좌표가 모두 일치했다. 이는 좌표계 연결의 검증이며 실제 표면 정합 오차가 0이라는 뜻이 아니다.',
           '- 기존 13개 구역 중 R07·R09·R10·R11은 구조 관측의 정합 지원이 부족하다. Gaussian 간 새 정합이나 픽셀 단위 워핑은 하지 않았다.',
           '- 세 원본 계획은 모두 `planning_status=incomplete`, `publication_ready=false`다. 경로 연결·미관측 객체·커버리지 실패를 그대로 기록했다. 개별 시점에서의 실험을 완성된 가상 투어로 설명하지 않는다.','',
           '## 실행과 결과','',
           '- 46개 시점에서 전후 92장을 새로 렌더링했다. 원본 SH3 Gaussian 전체(386,826 / 369,446)를 유지했다.',
           '- 원본 투어 요청의 2048×1024와 같은 포즈·화면 비율로 1024×512 ERP를 렌더링해 기존 공개 synthetic+ChangeSim SimSaC 가중치에 입력했다. 원본 영상 사진은 모델 입력에 사용하지 않았다.',
           '- 양쪽 alpha ≥ 0.85, 유한한 예상 깊이·거리 범위와 사람 제외를 적용했다. 기존 메쉬 양성 후보나 구역별 ROI로 검출 픽셀을 제한하지 않았다.',
           f'- 동일 이미지 자기 비교 {report["self_controls_zero"]}회에서 후보 0을 확인했다. 입력 RGB·확률·깊이 해시, 요청 포즈 보존, 두 렌더의 카메라 일치, 후보 픽셀 분해와 모든 3D 영역의 픽셀 보존을 검증했다.','',
           '| 계획 | 시점 | 후보 픽셀이 남은 시점 | 평균 양쪽 유효 렌더 비율 |',
           '|---|---:|---:|---:|']
    for group,g in report['groups'].items():lines.append(f'| {group} | {g["views"]} | {g["positive_views"]} | {g["mean_valid_overlap"]:.1%} |')
    lines+=['',f'사람 제외 후 3D voxel 연결 조각은 시점별 합계 {report["filtered_view_regions"]:,}개다. 여러 시점에 중복되고 작은 노이즈 조각을 포함하므로 물리적 변화 객체 수가 아니다. 화면의 기본 박스는 전후 합계 100 px 이상만 표시하며, 전체 조각 선택과 원본 확률은 보존했다.',
            '', '## 사용과 한계','',
            f'투어 박스는 기존 Gaussian RGB SimSaC 결과의 사람 제외 {len(fixed["boxes"]["filtered"])}개 3D 영역을 재사용한다. 원본 id·bounds·center·픽셀 수를 바꾸지 않으며, 이 실험의 시점별 새 후보 {report["filtered_view_regions"]}개로 대체하지 않는다. `tour-candidates.json`에 원본 SHA-256과 후보를 기록한다.',
            '키프레임·미니맵 지점을 누르면 서버의 저장 polyline을 따라 이동한다. 키프레임의 전후 예상 깊이·alpha와 고정 후보 영역의 겹침을 검사한 뒤, 현재 카메라에 투영해 표시한다. 이동 중에는 중간 깊이 검증이 없어 박스를 숨긴다. 같은 영역은 여러 키프레임에서 같은 id와 3D 범위를 유지한다. 후보 박스를 누르면 그 3D 영역의 후 맵을 55%로 겹치며 비율 조절·닫기를 지원한다.',
            f'실제 저장 링크 {routes["total_links"]}개로만 46개 시점을 연결했다. P2 / P3 / S3는 각각 4 / 5 / 6개의 연결 구간이다. 누락 링크를 만들거나 구간 사이를 자동 이동하지 않는다. 구간 선택은 명시적인 위치 전환이며 자동 둘러보기는 구간 끝에서 멈춘다. 후보 없는 키프레임도 경로에서 생략하지 않는다.',
            '둘러보기에서 시선을 움직여도 추론을 실시간 재실행하지 않는다. 새 키프레임에는 해당 포즈의 두 렌더와 SimSaC 추론을 추가해야 한다. 기본 계획은 P3이며 변화 후보와 무관하게 가장 긴 저장 연결 구간에서 시작한다.',
            '두 맵은 서로 다른 층의 Dorm2·Dorm3이며 시간 변화 정답이 없다. 재구성 흐림·가림·정합 잔차와 ERP 모델 적용의 영향을 포함한다. 시점 연결과 실행을 확인한 결과이며 변화탐지 정확도 검증은 아니다.',
            '현재 뷰어의 입력 RGB 펼치기에서 실제 두 입력을 확인할 수 있다. 원본 기존 메쉬 후보는 이전 구역 대표 비교 화면에 그대로 보존했다.',
            '', '전체 Gaussian 모델은 이전 화면의 `models/`를 공유한다. 확률·깊이 NPZ는 Git에서 제외하고 `large-inputs.json`에 해시를 기록한다. 포즈 출처·검증·실행 소스 해시는 각 JSON을 참고한다.','']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('outputs/gaussian-keyframes-20261026-v2'))
    p.add_argument('--destination',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-keyframe-comparison-20261026'))
    a=p.parse_args();build(a.run,a.destination)
