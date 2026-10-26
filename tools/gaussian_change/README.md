# 가우시안 맵 입력 · 메쉬 기준선 / SimSaC 비교

`dorm1_3dgs_integration`은 공통 Nerfstudio 코드 패키지다. 실제 비교 대상은 그 코드로 각각 학습한 Dorm2·Dorm3 두 맵이며 원본 영상 SHA-256, 데이터 receipt, checkpoint SHA-256이 모두 다르다. 각각 60,000 step, 386,826 / 369,446 Gaussians다. 같은 맵의 체크포인트 두 개를 전후로 사용하지 않는다.

- 원본 모델: `/workspace/v1_splatfacto_comparison_20261025/models/{dorm2,dorm3}/splatfacto/linked_recipe/nerfstudio_models/step-000059999.ckpt`
- 원본 코드 패키지: `/workspace/v1_splatfacto_comparison_20261025/vendor/dorm1_3dgs_integration`
- 현재 실행 산출물: `/workspace/gaussian_change_20261025/published`

`prepare.py`는 기존 메쉬 정합·오차·관측검증 판정을 그대로 읽는다. 기존 `native_mesh_to_canonical`과 `mesh_to_glb.npy`를 합성하고 233 / 248개 원본 카메라에서 위치·방향을 확인한다. 새 정합을 피팅하지 않는다. 기준선 박스는 `event_groups.entry_ids` 트랙의 원본 면만 사용하며 그룹 문맥 전체를 양성으로 만들지 않는다.

`render_maps.py`는 원래 학습된 모든 Gaussian을 SH3 PLY로 보존한다. 후 맵의 고정 구역 소속에 따라 13개 파일로 나누고 각 구역에 기존 정합·이동 편향을 적용한다. 평균·공분산의 회전·축척을 함께 변환하며 SH는 각 Gaussian의 원래 좌표계에서 평가한다. 기존 구역 경계 단차는 남는다. 두 맵을 각 구역의 **같은 카메라**에서 렌더링한다. 사진 선명화나 원본 RGB 덧씌우기는 없다.

`infer_simsac.py`는 기존 공개 synthetic+ChangeSim 가중치를 strict 로드해 새 Gaussian RGB 렌더에 양방향 추론한다. 전체 ERP를 기존 520/256 branch 전처리로 사용한다. 0.5 임계값은 유지한다. 13개 동일 입력 대조에서 후보 0을 확인했다. 6개 cube 렌더의 YOLO11l-seg 사람 마스크와 각 변 10% 여유 박스를 양쪽 합집합으로 제외한다. 양쪽 alpha ≥ 0.85, 유한한 예상 깊이, 고정 공간 구역만 표시 근거로 사용하며, 제외 픽셀을 미관측·구역 밖·사람으로 따로 기록한다. 기존 양성 메쉬로 SimSaC 후보를 제한하지 않는다. 사람 여유 박스는 주변 표면도 제외할 수 있고, 흐린 Gaussian 렌더에서 사람 검출 오류가 날 수 있다.

3D 박스는 Gaussian 예상 깊이로 연결한 후보점의 0.06h voxel 연결 성분이며 0.02h 여유를 둔다. 모든 보존 후보 픽셀이 한 박스에 포함되는지 검사한다. 박스를 실제 객체 ID·이동·변화 정답으로 해석하지 않는다. 현재 기존 후보 2묶음, 사람 제외 후 SimSaC 5영역이다.

브라우저는 [Spark 2.4.0](https://sparkjs.dev/docs/)와 Three.js 0.180.0으로 **실제 Gaussian PLY**를 표시한다. 원본 Gaussian 386,826 / 369,446개를 모두 유지한다. 처음은 전 맵 하나이고, 박스를 선택하면 해당 3D volume에서 전 가우시안을 흐리게 하고 후 가우시안을 겹친다. 전/후 맵 전환, 시선·위치 이동, 전체 화면, 영역 내 0–100% 겹침을 지원한다. 원본 사진 파노라마를 Gaussian 맵처럼 표시하지 않는다.

이 실험은 서로 다른 층의 공간 차이 파일럿이다. 이전 RGB 실험 20프레임·36쌍을 그대로 재사용한 통제 평가가 아니다. 13구역별 기존 카메라 중 기하 중심/고정 메쉬 후보에 가까운 위치를 미리 선택했다. 후보 개수를 정확도나 성능 향상으로 해석하지 않는다. 재구성 차이·정합 잔차·가림, Gaussian 예상 깊이의 편향이 남는다.

## 확인

```powershell
python tools/gaussian_change/validate.py
python -m unittest discover -s tests -p test_gaussian_map_bridge.py -v
```

저장소 스냅샷에서 대형 원본 없이 메타데이터·입력 이미지·후보 마스크만 확인할 때는 `validate.py --directory <뷰어 폴더> --metadata-only`를 쓴다. 전체 Gaussian PLY와 float32 확률 NPZ는 Git에서 제외하고 SHA-256과 경로를 기록한다. 전체 검증은 이 파일들과 보존된 메쉬 입력을 요구한다.

실행 환경은 기존 서버의 `v1_splatfacto_comparison_20261025/env.sh`(gsplat 1.4.0)와 `sam31_panorama_test_20261023/env/bin/python`(SimSaC)을 분리했다. 사람 검출 코드는 기존 `image-lifecycle-v4/deps`를 읽는다. 원본 모델과 메쉬 파일은 수정하지 않으며 실행 결과는 새 디렉터리에 쓴다.

## 저장된 Task 5b 키프레임 시점

`prepare_keyframes.py`는 서버에서 가져온 기준 Dorm2 13개·Dorm3 13개와 Dorm3 통계 실험 20개의 `render_requests.json`을 그대로 사용한다. 이 경로는 기존 변화 후보를 시점 선택에 사용하지 않는다. 세 원본 계획은 모두 경로 연결·커버리지 검증 미완료이며, 개별 포즈 실험을 완성된 투어 경로로 해석하지 않는다.

전후의 원본 키프레임 위치를 같게 바꾸는 방식이 아니다. 선택한 한쪽 키프레임의 포즈를 공통 프레임으로 옮긴 후 그 포즈에서 **두 맵을 모두 렌더링**한다. 반대 회차에서 가장 가까운 키프레임을 골라 대체하지 않는다. 어느 쪽 키프레임을 기준으로 하든 같은 절차를 사용한다.

계획의 `dorm*_mesh`는 Z-up, Gaussian 학습 프레임은 Y-up이다. `mesh_to_glb.npy`를 먼저 적용하고 기존 `native_to_mesh`와 전역·구역 정합 및 이동 편향을 합성한다. 위치만 바꾸거나 카메라 회전을 축척하면 안 된다. 원본 촬영 카메라 233/248개와의 좌표 연결, 요청 포즈 보존·roll·광선 변환을 검증한다.

```powershell
python tools/gaussian_change/prepare_keyframes.py
# protocol.json / partition.npz와 실행 코드를 별도 서버 run 폴더로 전달한다.
# 서버: render_maps.py --root /workspace/gaussian_keyframes_20261026_v2
# 렌더 완료 후: infer_keyframes.py --root /workspace/gaussian_keyframes_20261026_v2
# 결과를 로컬 run/published에 내려받은 후:
python tools/gaussian_change/build_keyframe_review.py
python tools/gaussian_change/validate_keyframes.py --directory outputs/scenediff-mesh-rgb-20261024/gaussian-keyframe-comparison-20261026
```

이 경로는 전체 360°의 양쪽 alpha·유효 깊이와 사람 제외를 적용하고 기존 양성 메쉬나 구역 ROI로 검출을 제한하지 않는다. 작은 voxel 조각이 생길 수 있어 화면 기본 박스는 전후 합계 100 px 이상만 표시한다. 이는 표시 필터이며 원본 확률·마스크와 전체 조각은 보존한다. 시야를 움직이는 것은 표시만 바꾸며 추론을 재실행하지 않는다.

R07·R09·R10·R11은 기존 구조 정합의 관측 지원이 부족하다. 가상 시점이 학습 관측에서 멀거나 반대 맵에 해당 영역이 없으면 변화 없음으로 판단할 수 없다. 새 Gaussian 정합·픽셀 워핑이나 정확도 검증까지 완료한 것으로 설명하지 않는다. 전체 결과를 공개하기 전에 `validate_keyframes.py`의 원본 확률·깊이·입력 해시·요청 포즈 검사를 통과해야 한다.


## 선택한 21지점 경로에 기존 검출 연결

현재 투어는 사용자가 지정한 `task5b_dorm3_fast_uncapped_20261026_IQK7um` 계획이다. 사진의 K1–K21, 20개 저장 이동선, 단일 연결 영역을 보존한다. `prepare_uncapped_tour.py`는 원본 계획 해시·K번호·248개 촬영 카메라 좌표·모든 링크 끝점을 확인한다.

**변화탐지는 기존 Gaussian RGB SimSaC 결과 그대로다.** `build_tour_candidates.py`가 원본 사람 제외 5영역(필터 전 6영역)의 id·3D bounds·center·검출 픽셀 수를 보존한다. 시점별 새 추론으로 교체하지 않는다. 각 키프레임에서 새로 만든 전후 RGB·예상 깊이 42개는 현재 위치의 렌더 표시와 가림 검사에 사용한다. SimSaC/Yolo 추론 스크립트는 실행하지 않는다.

키프레임·미니맵·바닥 이동 지점을 누르면 저장된 polyline을 따라 선택 위치까지 이동한다. 자동 둘러보기는 원래 방문 순서의 모든 키프레임을 거친다. 도착하면 현재 위치에서 관측 가능한 고정 3D 박스가 투영되고, 선택하면 해당 volume의 후 Gaussian이 기본 55%로 겹친다. 박스 선택 시 자동 진행은 멈춘다. 중간 이동 위치는 깊이 검증이 없어 도착 전까지 박스를 숨긴다.

```powershell
python tools/gaussian_change/prepare_uncapped_tour.py
# 서버에서 render_maps.py --root /workspace/gaussian_uncapped_tour_20261026 만 실행
# 렌더 결과를 로컬 outputs/gaussian-uncapped-tour-20261026/published 로 수신
python tools/gaussian_change/build_uncapped_tour.py
python tools/gaussian_change/validate_uncapped_tour.py --directory outputs/scenediff-mesh-rgb-20261024/gaussian-uncapped-tour-20261026
```

`gaussian-uncapped-tour-20261026/index.html`이 현재 연결 화면이다. 이전 `gaussian-keyframe-comparison-20261026/index.html` 주소도 이 화면으로 이동한다. 기존 46시점 추론 실험은 원래 run과 `experiment-46.html`에 보존한다. 전체 Gaussian PLY와 깊이 NPZ는 Git에 포함하지 않는다. 화면 연결·입력 무결성 검증을 실제 변화 정확도·객체 커버리지·보행 안전 검증으로 해석하지 않는다.


## 인스턴스 자동 분리와 변화 유형 자동 분류

`link_segmented_objects.py`와 `automatic_object_matching.py`는 기존 219개 semantic mesh component, 116개 영상 객체 트랙 및 다중 시점 마스크의 3D 표본을 사용한다. 변화 박스 안의 객체만 보지 않고 같은 구역의 박스 밖 객체도 자동으로 연결한다. 여러 의자가 합쳐진 메쉬는 각 영상 인스턴스의 3D 표본에 대한 면별 거리로 자동 분리한다. 원본 삼각형·정점 좌표와 정합을 변경하지 않는다.

같은 위치에 남아 있는 영상 인스턴스 대응은 먼저 보호한다. 나머지 같은 종류 객체는 높이를 기준으로 정한 공통 임계값으로 yaw+translation 강체 형상 비교를 수행한다. 객체별 축척은 맞추지 않는다. 양방향 F1≥0.65, 각 방향 커버리지≥0.5, 양쪽 위치에서 그대로 남아 있는 전체 메쉬 표면보다 정합 후 커버리지 개선≥0.10, 정합 오차 3배 및 높이 0.15배 이상의 이동, 도착 위치의 독립 빈 공간 관측≥2회를 요구한다. 경쟁 대응과의 F1 차이가 0.05 미만이면 유형 불확실로 남긴다. null 대응을 허용하는 1:1 할당으로 한 객체를 여러 이동에 재사용하지 않는다.

- R12: 병합된 S0-010의 원본 758면 중 영상 트랙 I0-005에 연결된 135면을 자동 분리한다. 후 의자 S1-024 / I1-067과 자동 대응한다. 사용자가 지정한 유형·출발 ID·구역별 정답 보정은 입력하지 않는다.
- R08: 영상 트랙과 분할 메쉬의 같은 자동 대응 조건으로 이동 추정을 계산한다.
- R13: 전 문 메쉬만 연결되어 소실 추정으로 표시한다. 부재 관측이 부족하므로 신뢰도는 segmentation_only다.
- R01: 전후 창문 메쉬가 모두 존재해 유형 불확실이다. 전 방향 변화 픽셀만으로 소실을 만들지 않는다.
- 작은 R08 조각: 직접 연결되는 객체 근거가 없어 유형 불확실이다.

이동 박스는 전 Gaussian 배경 위에 청록 출발 객체와 주황 도착 객체의 색 그림자 및 전→후 화살표를 표시한다. R12는 의자 줄 전체 대신 자동 분리된 출발 조각만 표시한다. 카메라는 키프레임 위치를 유지한다. 기존 SimSaC의 5개 필터 후 / 6개 필터 전 영역의 id·bounds·픽셀 수, 21개 키프레임·20개 경로는 변경하지 않는다.

`automatic-matching.json`은 공통 임계값·자동 분리 매핑·수락/탈락한 모든 비교를 기록한다. `object-evidence.json`은 자동 분류와 입력 SHA를, `object-meshes/`는 원본 면의 변환 좌표와 ID를 기록한다. `manual_overrides_used`는 false이며 수동 분류 보정 경로는 제거했다. 유형과 동일 객체 대응은 자동 추정이며 물리적 정답 보장은 아니다.

```powershell
python tools/gaussian_change/link_segmented_objects.py
python tools/gaussian_change/build_uncapped_tour.py
```

## 구역별 고유 변화와 관측 시점

`build_change_regions.py`는 같은 원본 후보 ID 또는 같은 자동 객체 대응을 공유하는 검출을 하나의 이벤트로 등록한다. 각 키프레임에는 이벤트의 ID와 관측 근거만 연결한다. 가까우면서 공통 관측 시점이 있는 이벤트를 공간 구역으로 묶되, 다른 객체의 이벤트를 하나로 합치지는 않는다. 원본 박스와 정합은 변경하지 않는다.

현재 사람 제외 결과는 **3구역 / 고유 후보 5개 / 관측 링크 13개**다. 구역1의 R12·R13 후보 2개를 K1·K2·K3·K5·K6에서 관측하며 8개의 새 후보로 세지 않는다. 필터 전 원본 검출 6개 중 같은 자동 이동 대응 2개도 이벤트 하나로 연결해 고유 후보는 5개다. `change-regions.json`에 이벤트·원본 검출 alias·구역·키프레임 관측을 분리해 저장한다.

화면의 구역별 고유 목록은 시점 이동 중에도 유지한다. 키프레임에 따라 현재 관측 표시와 투영 박스만 갱신한다. 현재 시점에서 보이지 않는 기존 후보를 누르면 그 후보를 관측할 수 있는 저장 키프레임으로 이동한다. 21개 원본 경로와 20개 링크를 보존하며 전체 관측 시점 목록은 접어서 표시한다.

## 자동화 검증 범위와 재현 감사

현재 `build_uncapped_tour.py`는 저장된 정합·분할 트랙·Gaussian 렌더·SimSaC 검출 파일을 읽어 객체 자동 분리·대응·유형 분류·중복 제거·투어 연결을 계산한다. 신규 맵 입력부터 정합, 분할, GPU 렌더, SimSaC 추론까지 모두 한 번에 실행하는 연결은 아직 완성·검증하지 않았다. 투어에서 시점 이동은 저장된 검출의 관측 연결이며 새 SimSaC 추론이 아니다.

현재 R12 유형에 수동 보정이나 객체 ID를 고정하는 분기는 없다. 다만 R12에 대한 피드백을 본 뒤 공통 매칭 규칙과 임계값을 개발·조정한 결과이므로, 독립 데이터에 대한 일반화 검증을 마친 것으로 표현하지 않는다. 형상 F1은 점군 형상 적합도이며 이동 분류 정확도나 동일 물리 객체의 정답 확률이 아니다.

`audit_automatic_replay.py`는 새 폴더에 `protocol.json`과 원본 `simsac-candidates.json`만 복사하고 기존 분할·트랙·메쉬 입력에서 전체 객체 분리/대응/분류를 다시 계산한다. 기존 유형 판정 파일은 복사하지 않는다. 현재 재실행은 기존 `automatic-matching.json`과 `object-evidence.json`의 SHA-256을 그대로 재현했다. 대응 결과를 제거하면 두 이동 사례 모두 유형 불확실로 바뀐다. 이 감사는 재현성을 확인하며, 상류 모델 재추론이나 변화탐지 정확도를 검증하지 않는다.

```powershell
python tools/gaussian_change/audit_automatic_replay.py
```

결과는 `automatic-replay-audit.json`에 기록한다.
