"""Publish the selected 21-camera tour using unchanged Gaussian-RGB detections."""
from pathlib import Path
import argparse
import shutil
from prepare import read,save,sha
from build_tour_routes import build as build_routes
from build_tour_candidates import build as build_candidates
from validate_uncapped_tour import validate


def build(run,destination,source):
    destination.mkdir(parents=True,exist_ok=True);published=run/'published'
    for p in published.rglob('*'):
        if not p.is_file() or 'models' in p.relative_to(published).parts:continue
        target=destination/p.relative_to(published);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    original=destination.parent/'gaussian-map-comparison-20261025'
    data=read(published/'gaussian-data.json')
    for phase in ['before','after']:
        for a in data['models'][phase]:
            a['file']='../gaussian-map-comparison-20261025/'+a['file']
            if sha(destination/a['file'])!=a['sha256']:raise ValueError('Shared Gaussian export mismatch')
    for record in data['keyframe_sources']:
        record['source_requests_file']='source-requests/U3.json'
        target=destination/record['source_requests_file'];target.parent.mkdir(exist_ok=True)
        shutil.copy2(source/'plan/render_requests.json',target)
        if sha(target)!=record['render_requests_sha256']:raise ValueError('Source requests changed')
    save(destination/'gaussian-data.json',data)
    for name in ['simsac-candidates.json','simsac-model.json']:shutil.copy2(original/name,destination/name)
    for name in ['protocol.json','selection-audit.json']:shutil.copy2(run/name,destination/name)
    shutil.copy2(source/'dorm3-uncapped-overview.png',destination/'route-overview.png')
    shutil.copy2(source/'map_report.json',destination/'route-map-source.json')
    shutil.copy2(source/'source-receipt.json',destination/'route-source-receipt.json')
    build_routes(run,destination/'tour-routes.json');build_candidates(run,original,destination/'tour-candidates.json')
    from link_segmented_objects import build as link_objects
    object_result=link_objects(destination,Path('outputs/registration-benchmark-20261019/new-meshfirst-20261022'))
    from build_change_regions import build as build_change_regions
    build_change_regions(destination)
    tools=Path(__file__).parent
    for name in ['viewer.js','viewer.css','tour.js','tour_navigation.js','object_evidence.js','change_regions.js']:shutil.copy2(tools/name,destination/name)
    shutil.copy2(tools/'uncapped.html',destination/'index.html')
    execution=destination/'execution-source';execution.mkdir(exist_ok=True)
    for name in ['render_maps.py','prepare.py']:shutil.copy2(run/'source'/name,execution/name)
    save(destination/'execution-receipt.json',{'server_run':read(run/'protocol.json')['remote_root'],
        'render':read(run/'source/render-status.json'),'launch':read(run/'source/launch.json'),
        'source_sha256':{p.name:sha(p) for p in execution.glob('*.py')},
        'protocol_sha256':sha(run/'protocol.json'),'archive':read(run/'package.json'),
        'new_simsac_inference':False,'automatic_object_matching':True,'manual_overrides_used':False})
    report=validate(destination);save(destination/'tour-validation.json',report)
    save(destination/'object-review-validation.json',report['object_evidence'])
    large={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in destination.rglob('*geometry.npz')}
    save(destination/'large-inputs.json',{'shared_models':'../gaussian-map-comparison-20261025/models','omitted_depth_files':large})
    (destination/'.gitignore').write_text('renders/*/*-geometry.npz\n',encoding='utf-8')
    lines=['# Dorm3 K1–K21 · 기존 변화영역의 키프레임 투어 연결','',
        '- 사용자가 지정한 평지·개수 상한 해제 계획: `/workspace/task5b_dorm3_fast_uncapped_20261026_IQK7um/plan/tour_plan.json`.',
        '- 첨부 이미지와 `route-overview.png`의 SHA-256은 동일하다. 21개 관측점, 저장 링크 20개, 연결 영역 1개이며 사진의 K번호를 보존한다.',
        '- 기존 Gaussian RGB→SimSaC 결과를 그대로 사용한다. 사람 제외 5개 영역의 id·bounds·center·전후 픽셀 수와 6개 필터 전 영역은 원본과 정확히 같다. 새 SimSaC 추론을 실행하지 않았다.',
        f'- 기존 분할 메쉬와 영상 인스턴스 트랙으로 {len(object_result["objects"])}개 메쉬 조각을 자동 연결했다. 병합된 의자 줄은 다중 시점 마스크의 3D 표본으로 자동 분리한다. 같은 자리에 남은 객체 대응은 보호하고 잔여 객체를 yaw+translation 강체 형상 및 양쪽 위치의 표면 개선, 도착 위치의 독립 빈 공간 관측으로 1:1 대응한다. 자동 추론 결과와 모든 탈락 근거는 automatic-matching.json에 기록한다. 수동 유형 보정 입력은 사용하지 않는다.',
        '- 키프레임의 원본 Z-up 포즈에 기존 축 변환·전역/구역 정합을 적용한다. 원래 촬영 카메라 248개 연결, 요청 포즈와 20개 경로 끝점 일치를 확인했다.',
        '- 각 키프레임에서 같은 카메라로 두 Gaussian 맵을 렌더링했다. 42개의 RGB와 예상 깊이는 현재 위치의 화면·가림 검사에 사용하며 새 변화 검출에는 사용하지 않는다.',
        '- 키프레임·지도·바닥 이동 지점을 누르면 실제 저장 polyline을 따라 이동한다. 도착하면 고정 3D 후보를 투영한다. 이동 박스를 누르면 전 Gaussian 배경 위에 청록 출발 구역과 주황 도착 객체의 분할 메쉬 색 그림자·방향 화살표를 표시한다. 후 Gaussian 원래 색상은 겹치지 않으며 키프레임 카메라 위치를 유지하고 시야만 돌려 양쪽을 본다. R12는 병합된 758면 의자 줄에서 영상 트랙에 해당하는 원본 135면을 자동 분리해 137면 도착 의자와 연결한다. 색 그림자는 의자 줄 전체가 아닌 자동 분리 객체를 표시한다. 그 외 유형은 기존 후 Gaussian 겹침을 유지한다.',
        '- 후보 없는 키프레임도 경로에서 생략하지 않는다. 자동 둘러보기는 구간 끝에서 멈추며, 박스 선택 시 자동 진행을 중지한다. 중간 이동 위치에는 깊이 검증이 없어 도착 전 박스를 숨긴다.',
        '- 변화는 구역별 전역 레지스트리에 한 번만 등록한다. 같은 원본 ID 또는 같은 자동 객체 대응을 공유하는 후보는 같은 이벤트이며, 키프레임 관측 링크로 새 이벤트를 만들지 않는다. 현재 필터 후 3구역·고유 후보 5개·관측 링크 13개다. K1/K2/K3/K5/K6는 구역1의 같은 후보 2개를 관측한다. 원본 필터 전 6개 중 동일 객체 대응 2개도 하나로 묶어 고유 이벤트는 5개다. 원본 박스는 변경하지 않는다.',
        '- 가림 검사는 각 고정 박스 내부 9³ 표본과 두 맵 중 하나의 alpha≥0.85·예상 깊이 일치를 요구한다. 거리 허용치는 0.08h + 거리의 2%, 서로 다른 최소 3개 ERP ray다. 측정된 실제 표면이나 물리적 가시성 정답이 아니다.',
        '- 두 맵은 다른 층 Dorm2·Dorm3이며 시간 변화 정답이 없다. 기존 정합의 R07·R09·R10·R11 지원 부족과 재구성·가림 잔차는 유지된다. 원본 경로의 객체 관측 커버리지·계산 모델 검증 및 실제 보행 안전은 미완료 상태다.',
        '', '## 검증','',
        '```powershell','python tools/gaussian_change/prepare_uncapped_tour.py',
        '# 서버: render_maps.py --root /workspace/gaussian_uncapped_tour_20261026 (추론 스크립트 실행 없음)',
        'python tools/gaussian_change/build_uncapped_tour.py',
        'python tools/gaussian_change/validate_uncapped_tour.py --directory <뷰어 폴더> --metadata-only','```','',
        '`tour-validation.json`은 원본 후보 보존, 카메라·링크 끝점, 42 RGB 해시, 가림 검사를 검증한다. 대형 모델·깊이 NPZ는 Git에서 제외하며 `large-inputs.json`에 기록한다.','']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    alias=destination.parent/'gaussian-keyframe-comparison-20261026'
    if (alias/'index.html').exists():
        archive=alias/'experiment-46.html'
        if not archive.exists():shutil.copy2(alias/'index.html',archive)
        href='../'+destination.name+'/index.html'
        (alias/'index.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={href}"><title>K1–K21 투어로 이동</title><a href="{href}">선택한 Dorm3 21지점 투어 열기</a></html>\n',encoding='utf-8')
    print(report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('outputs/gaussian-uncapped-tour-20261026'))
    p.add_argument('--destination',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-uncapped-tour-20261026'))
    p.add_argument('--source',type=Path,default=Path('outputs/route-versions-20261026/dorm3-fast-uncapped'))
    a=p.parse_args();build(a.run,a.destination,a.source)
