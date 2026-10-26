# Task 5a · 모델 비교 진행 중

2026-10-25 기준, **기존 정합오차·관측검증 방식과 SimSaC·ViewDelta의 변화 후보를 비교하고 있다.** 현재까지의 코드, 측정 결과 JSON, 비교 화면, 가중치 로딩 검사와 캡처를 이 브랜치에 보관한다. 공통 Task 5a backend의 통합 완료를 뜻하지 않는다.

## 현재 화면: 실제 가우시안 맵의 메쉬 기준선 / 새 SimSaC 후보

공통 `dorm1_3dgs_integration` 코드를 사용해 각각 학습한 **Dorm2·Dorm3 두 맵**을 비교한다. 원본 영상·데이터 receipt·체크포인트 SHA-256이 다르다. 각각 60,000 step, 386,826 / 369,446 Gaussians이며 전체를 유지한다. 큰 PLY와 원본 float32 확률 NPZ는 Git에 포함하지 않고 `large-inputs.json`에 경로·해시를 기록했다.

기존 정합·오차·관측검증은 메쉬에 그대로 유지한다. 기존 후보 2묶음의 양성 트랙·원본 면 번호와 경계를 보존했다. 기존 메쉬 정규화와 GS 좌표 변환을 합성해 233 / 248개 원본 카메라에서 위치·회전을 확인했고 새 정합은 피팅하지 않았다. 후 맵은 고정 구역 소속별로 13개 PLY로 나누어 기존 구역 정합·이동 편향을 적용한다. 전체 Gaussian 개수는 변하지 않는다.

SimSaC는 두 GS 맵을 13구역의 동일 정합 카메라에서 새로 렌더링해 양방향 추론했다. 원래 SH3·opacity·scale·회전을 사용하고 원본 사진 덧씌우기나 선명화는 하지 않았다. 기존 RGB의 확률은 재사용하지 않는다. 공개 synthetic+ChangeSim 가중치 strict 로드, 기존 520/256 전처리·0.5 임계값을 유지한다. 13개 동일 입력 대조에서 후보 0이다.

새 렌더의 YOLO11l-seg 사람 분할과 각 변 10% 여유 박스를 양쪽 합집합으로 제외한다. 양쪽 alpha ≥ 0.85, 유한한 예상 깊이와 고정 공간 구역에서 표시한다. 원본 후보는 구역 밖·미관측·사람·남은 픽셀로 분해하며 기존 양성 메쉬로 SimSaC를 제한하지 않는다. 현재 사람 제외 후 SimSaC 5영역: R01 1, R08 2, R12 1, R13 1이다. R12에서 필터를 끄면 2영역으로 복원된다.

처음에는 전 GS 맵 하나와 후보 박스만 보인다. 박스를 누르면 해당 3D volume 안에서만 후 가우시안이 겹친다. 전/후 맵 전환, 시선·위치 이동, 전체 화면과 0–100% 겹침을 지원한다. 브라우저는 Spark 2.4.0 / Three.js 0.180.0을 로컬로 로드한다. 이전 RGB 단일 파노라마는 `focus-comparison-20261025/candidate-review-rgb.html`에 보관했고 예전 URL은 새 GS 화면으로 이동한다.

이 실험은 서로 다른 층의 공간 차이 파일럿이다. 이전 RGB 20프레임·36쌍과 관측량·카메라 선택을 맞춘 통제 평가는 아니다. 13구역별 기존 reference 카메라 중 기하 중심/기존 후보에 가까운 위치를 미리 골랐다. Gaussian 예상 깊이·재구성 차이·정합 잔차·가림 때문에 후보가 생길 수 있으며 2묶음과 5영역을 객체 수나 정확도 순위로 해석하지 않는다.

## 입력 방식 정리 · 2026-10-26

현재 실행 결과는 기존 메쉬 정합·관측검증과 Gaussian 렌더 RGB 입력 SimSaC의 비교다.

| 경로 | 변화 판단 | 가우시안 맵의 역할 |
|---|---|---|
| 기존 방식 | 메쉬 정합오차와 RGB 객체 트랙, 여러 시점의 빈 공간·가림·관측 근거로 판정 | 저장된 메쉬 후보 박스를 표시하는 3D 배경. 기존 판정은 재계산하지 않음 |
| SimSaC | 두 Gaussian 맵을 같은 정합 카메라에서 렌더링한 RGB의 변화 픽셀 추론 | RGB 입력 생성, 예상 깊이를 통한 3D 후보 위치 연결, 결과 표시 |

SimSaC에 원본 영상 프레임을 넣은 이전 실험은 보관했고, 현재 Gaussian 실험의 확률은 새 렌더로 다시 계산했다. 가우시안 위치·공분산·SH 속성을 직접 입력받는 변화탐지 모델은 현재 연결하지 않았다. 사람 제외·유효 깊이 필터는 학습 모델의 원본 확률과 별도이며, 기존 후보 2묶음과 SimSaC 5영역은 정확도 비교가 아니다.

Gaussian 렌더 영상을 변화 분석에 사용하는 연구로 [3DGS-CD](https://arxiv.org/abs/2411.03706), [O-SCD](https://chumsy0725.github.io/O-SCD/)가 있다. 이 연구들의 기준 맵 렌더와 새 촬영 영상 비교를, 현재 두 맵의 렌더를 SimSaC로 비교한 실험과 동일한 모델·프로토콜로 설명하지 않는다.

[직접 입력 변화탐지 조사](gaussian-direct-change-detection-research-20261025.md)에서는 GS-Diff·GS-Pool의 목적과 공식 실행 코드 미공개 상태를 확인했다. GS-Diff 저자는 2026-10-22에 잠정 코드 공개 시점을 2027년 초로 안내했다. ChronoFuseGS는 다중 시점 맵 융합·재학습 및 변화 시각화 방법이므로 요청한 변화탐지 모델의 적용 후보에서 제외했다. 이 갱신은 설명·조사 기록이며 새 모델 실행이나 기존 검출 결과 변경을 포함하지 않는다.

## 투어 키프레임 시점 추가 실험 · 2026-10-26

서버의 기준 Dorm2 13개·기준 Dorm3 13개와 최신 Dorm3 통계 실험 20개의 저장된 렌더 요청을 모두 사용했다. 기존 변화 후보를 시점 선택에 사용하지 않았다. 선택한 어느 쪽 키프레임이든 그 위치·방향을 공통 프레임으로 옮긴 다음, 같은 포즈에서 두 Gaussian 맵을 렌더링한다. 서로 다른 원본 키프레임 사진을 그대로 겹치거나 가장 가까운 반대 회차 키프레임으로 대체하지 않는다.

계획의 Z-up mesh 좌표에서 Y-up Gaussian 좌표로 `mesh_to_glb.npy`를 적용한 후 기존 전역·구역 정합과 이동 편향을 합성한다. 233 / 248개 원본 촬영 카메라와 축 변환을 확인했고, 저장된 46개 요청의 위치·방향 보존, 두 RGB의 카메라 일치, 입력·확률·깊이 해시와 모든 후보 픽셀 분해를 검증했다. 초기 축 변환 누락 실행은 폐기하고 `/workspace/gaussian_keyframes_20261026_v2`의 재실행만 사용했다. 실제 실행 소스와 해시는 결과의 `execution-source/`, `execution-receipt.json`에 보관했다.

| 계획 | 시점 | 후보 픽셀이 남은 시점 | 평균 양쪽 유효 렌더 영역 |
|---|---:|---:|---:|
| 기준 Dorm2 | 13 | 5 | 99.9% |
| 기준 Dorm3 | 13 | 5 | 100.0% |
| 최신 Dorm3 통계 실험 | 20 | 8 | 98.0% |

전체 유효 렌더 평균은 99.1%이며 정합 정확도가 아니다. 최신 실험에는 최소 66.6%인 시점도 있어 미관측 부분을 변화 없음으로 판정하지 않는다. 동일 이미지 자기 비교 92회는 후보 0이다. 사람 제외 후 시점별 voxel 영역 합계 26개는 여러 시점에 중복되므로 물리적 변화 객체 수가 아니다. 기본 화면은 100 px 이상 박스만 표시하고 전체 조각·원본 확률은 보존한다.

기존 정합을 적용한 뒤 실행했으며 새 Gaussian 정합이나 픽셀 단위 워핑은 하지 않았다. R07·R09·R10·R11은 기존 구조 정합의 관측 지원이 부족하다. 세 원본 투어 계획도 `planning_status=incomplete`, `publication_ready=false`다. 이는 키프레임 시점 연결·추론 실험이며 완성된 투어 또는 변화탐지 정확도 검증을 뜻하지 않는다.

[키프레임 비교 화면](../src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-keyframe-comparison-20261026/index.html)에서 계획·키프레임을 선택하고 실제 입력 RGB를 펼칠 수 있다. 같은 키프레임 위치에서 전/후 전체 맵 전환·후 후보 영역 겹침도 확인했다. 이전 구역 대표 시점 화면과 기존 메쉬 판정은 보존했다. 확률·깊이 NPZ는 대형 외부 입력으로 Git에서 제외하고 해시를 기록했다. Gaussian 모델과 브라우저 런타임은 이전 Gaussian 화면의 파일을 공유한다.

```bash
python tools/gaussian_change/validate_keyframes.py --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-keyframe-comparison-20261026 --metadata-only
```

신규 카메라 변환 테스트 4개를 포함한 모델 비교 Python 테스트 27개와 기존 후보 표시 Node 테스트 7개를 통과했다. 신규 테스트는 축척과 roll을 포함한 광선 일치, Z-up→Y-up 높이 연결, 기존 변화 후보 위치를 바꿔도 같은 가상 포즈를 쓰는지, 다른 촬영 좌표의 계획을 거부하는지를 확인한다. 브라우저에서 원본 Gaussian 전체 개수, 고정 키프레임 선택, 실제 RGB 입력 링크, 영역 겹침, 작은 조각 표시와 관측 지원 부족 경고를 확인했다.

## 비교 화면 실행

저장소 루트에서 실행한다. 브라우저 표시에는 WebGL2가 필요하다. 대형 Gaussian PLY는 별도로 서버에서 가져온다. SSH 비밀번호는 로컬 프롬프트에서 입력하며 저장하지 않는다.

```bash
scp -P 4444 -r root@155.230.28.172:/workspace/gaussian_change_20261025/published/models src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-map-comparison-20261025/
```

이후 로컬 HTTP 서버를 실행한다. 이전 RGB 결과와 정적 캡처는 대형 입력 없이도 확인할 수 있다.

```bash
python -m http.server 4190 --bind 127.0.0.1 --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025
```

브라우저에서 [http://127.0.0.1:4190/gaussian-map-comparison-20261025/index.html](http://127.0.0.1:4190/gaussian-map-comparison-20261025/index.html)을 연다. GitHub에서 HTML 링크를 누르면 소스 보기로 열리므로 로컬 HTTP 서버를 사용한다.

| 화면 | 내용 |
|---|---|
| `gaussian-map-comparison-20261025/index.html` | 실제 두 GS 맵의 기존 메쉬 / 새 SimSaC 후보와 3D 영역 겹침 |
| `focus-comparison-20261025/candidate-review-rgb.html` | 이전 RGB 입력의 단일 전 파노라마·후 겹침 |
| `model-comparison-20261025/index.html` | 앞서 실행한 SceneDiff 두 입력 경로까지 포함한 비교 |
| `paired/index.html` | 동일 RGB에서 SceneDiff 원본 Pi3/SAM 입력과 외부 메쉬·인스턴스 입력의 후보 차이 |

각 화면의 `REPORT.md`, `comparison.json`, `validation.json`에 수치·조건·검증을 보관했다. [스냅샷 파일 해시](../src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/snapshot-manifest.json)로 복사된 결과 파일을 확인할 수 있다.

## 입력과 이전 다중 상대 평균 실험 (보관)

- Dorm2 / Dorm3의 서로 다른 층을 비교한 공간 차이 파일럿이다. 시간 변화의 물리적 정답은 없다.
- R01·R07·R08의 30개 방향 RGB를 사용한다. 대응하는 전체 ERP는 20프레임이다. 방향 RGB는 기존 SceneDiff 입력과 바이트·해시가 같다.
- 두 모델은 **동일 방향 RGB**와 **전체 ERP → 동일 방향 결과 투영**을 각각 실행했다. 원본 모델의 전처리를 유지했으며 구면 전용 모델로 바꾼 것은 아니다.
- 두 모델의 상대는 기존 메쉬 SceneDiff의 view-overlap으로 고정했다. 모든 비영 상대의 예측확률을 평균하고 **0.5 초과**를 후보로 표시했다. 기존 후보 마스크로 출력을 제한하지 않았다.
- 기존 방식은 저장된 `occupancy-evidence-v2` 자동 판정을 사용한다. 이 판정은 전체 촬영을 사용했으므로 두 모델과 관측량을 맞춘 성능 평가가 아니다.
- 기존 양성은 후보를 만든 트랙의 원본 면만 first-hit 투영한다. 같은 종류의 주변 메쉬, 보류·문맥·억제·판정 연결 없음·mesh hit 없음은 따로 기록한다. 보류를 변화 없음으로 취급하지 않는다.

기존 전체 13구역에는 의자 관련 후보 묶음 2개가 있고 현재 모델 범위에는 R08의 E001 / I1-096이 포함된다. 후보 묶음을 물리적 객체 변화 건수로 해석하지 않는다.

## 이전 다중 상대 평균의 사람 제외 비교 (보관)

기존 정적 메쉬 객체 목록은 이미 `person`을 제외했다. 기존 관측검증도 `person_box_iou < 0.15`인 관측만 비점유 근거로 사용한다. RGB에는 촬영자가 남아 있으므로 SimSaC 출력에는 별도의 사람 제외가 필요했다.

현재 화면은 **사람 제외를 기본으로 적용**한다. 같은 대상 프레임의 기존 YOLO11l 사람 분할과 검출 박스의 합집합을 사용하며, 박스의 각 변에 폭·높이의 10% 여유를 둔다. ERP seam을 감싸고 극점 밖은 자른다. 사람 외곽·옷을 보수적으로 제외하므로 박스 안의 주변 표면도 비교에서 빠질 수 있다. SimSaC·ViewDelta와 기존 메쉬 투영에 같은 제외 영역을 적용한다.

메쉬에 사람이 없으면 RGB 사람 뒤의 표면이 first-hit될 수 있다. 해당 투영은 현재 관측 가능한 표면으로 세지 않는다. 기존 객체 판정·원본 모델 확률·임계값은 유지한다. ‘사람 제외’를 끄면 필터 전 원본을 볼 수 있다. 대상 RGB 좌표에서의 후처리이며, 반대 촬영 사람의 영향까지 모두 제거됐다는 뜻은 아니다.

| 입력 | 모델 | 필터 전 후보 | 제외한 사람 검출 영역 | 남은 후보 |
|---|---|---:|---:|---:|
| 동일 방향 RGB | SimSaC | 93,826px | 60,718px | 33,108px |
| 동일 방향 RGB | ViewDelta | 61,154px | 28,546px | 32,608px |
| 전체 ERP → 공통 방향 | SimSaC | 63,866px | 40,069px | 23,797px |
| 전체 ERP → 공통 방향 | ViewDelta | 6px | 0px | 6px |

R08의 기존 후보 트랙 표면은 원본 3개 관측 투영에서 4,681px이었지만 사람 가림을 제외하면 **1,989px**이 보인다. 같은 표면과의 겹침은 다음과 같다.

| 모델 입력 | SimSaC | ViewDelta |
|---|---:|---:|
| 동일 방향 RGB | 930px | 0px |
| 전체 ERP 입력 후 투영 | 1,725px | 0px |

이는 기존 후보 표면과의 동의 정도이며 정확도·재현율이 아니다. 필터 후 120개 방향 모델 마스크에서 사람 제외 영역과의 겹침이 0이며, 제외 영역 밖의 원본 후보가 그대로 유지됨을 확인했다.

## 공개 모델과 실행 검사

| 모델 | 고정 소스 커밋 | 실행한 체크포인트 SHA-256 |
|---|---|---|
| [ViewDelta](https://github.com/drags99/viewdelta-scd) | `e86596ab505856ac2a454f38c519f52864b1c375` | `708b19ee253686fe998b9ac9fe41b81d2b4d1e936800b17244bc5bd8a27da28f` |
| [SimSaC](https://github.com/SAMMiCA/SimSaC) | `43abe282231dc7d8c8cb829a92f9ebc014fa6cb6` | `55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a` |
| [SceneDiff](https://github.com/yuqunw/scene_diff) | `62dcbd285fcfe984bb3db55dc01bdcd1a0055714` | 구성 모델별 검사는 `paired/native-model-audit.json` 참고 |

ViewDelta·SimSaC의 학습 가중치를 `strict=True`로 전량 로드했다. 임의 초기화한 변화 head로 실행하지 않았다. 모델별 158개 방향 있는 쌍을 추론했고, 동일 입력 자기 비교에서는 두 입력 모드 모두 후보 0픽셀이었다. 실제 로딩·전처리·호환 처리 내역은 `model-comparison-20261025/results/{viewdelta,simsac}/model.json`에 있다.

ViewDelta는 공개 [hoskerelab/ViewDelta](https://huggingface.co/hoskerelab/ViewDelta), SimSaC는 공식 공개 synthetic+ChangeSim 체크포인트를 사용했다. 외부 코드와 가중치는 각 배포처의 사용 조건을 따른다. 모델 가중치·원본 전체 영상·전체 메쉬·대용량 쌍별 확률 배열은 이 문서용 스냅샷에 포함하지 않는다.

## 코드와 검증

실험 코드는 [`tools/scenediff_mesh/`](../tools/scenediff_mesh/)에 있다.

| 코드 | 역할 |
|---|---|
| `prepare.py` | 기존 메쉬·카메라·인스턴스를 RGB 방향 뷰에 투영 |
| `compare_inputs.py` | SceneDiff 원본 / 외부 입력 후 후속 후보 탐지 비교 |
| `prepare_model_comparison.py`, `run_pair_models.py` | 공통 상대 고정, ViewDelta / SimSaC 실제 추론 |
| `report_pair_models.py`, `report_inputs.py` | 이전 비교 화면 생성 |
| `build_focus_comparison.py` | 저장된 기존 판정과 투영을 준비하고 마지막에 전후 뷰어 생성 |
| `build_before_after.py`, `before_after.html`, `before_after.js` | 선택한 전후 쌍과 기존 방식 / SimSaC의 양방향 후보 표시 |
| `build_panorama_review.py` | 원본 360° 텍스처·전체 메쉬 투영·저장 포즈·깊이 준비 |
| `build_panorama_interactions.py`, `panorama_canvas.js`, `panorama_review.*` | 후보 박스·관측 RGB 재투영·전 영상 둘러보기·선택 영역 반투명 비교 |
| `frozen_object_event_policy.py` | 기존 판정 정책의 소스 스냅샷. 이 화면에서 판정을 다시 계산하지 않음 |

실제 추론·화면 재생성에는 별도로 보존된 입력과 확률 캐시가 필요하다. 일부 서버용 스크립트는 실행 당시 `/workspace/...` 배치를 고정하므로 새 환경에서는 외부 소스·가중치·입력 경로를 맞춰야 한다. 이 커밋만으로 원본 전체 촬영을 재추론할 수 있다고 설명하지 않는다. `run.py`의 이전 기하/RGB 실험은 현재 요청한 기존 정합오차·관측검증 기준선이 아니다.

현재 로컬 입력이 같은 배치로 준비된 환경에서는 다음 명령으로 집중 비교를 다시 만든다.

```bash
python tools/scenediff_mesh/build_focus_comparison.py --root outputs/scenediff-mesh-rgb-20261024 --prior outputs/registration-benchmark-20261019/new-meshfirst-20261022
```

두 모델 어댑터 검증은 NumPy·SciPy·Pillow·PyTorch, 외부 메쉬 검증은 추가로 Open3D·PyYAML·timm 환경을 사용한다. 실제 모델 추론 환경은 각 공식 소스의 의존성도 필요하다. 무거운 의존성이 있는 실험 검증은 공통 CPU 규약 검사와 별도로 실행한다.

```bash
python -m unittest discover -s tests/task_5a_change_detection/model_comparison -v
node --test tests/task_5a_change_detection/model_comparison/test_panorama_candidate_filter.mjs
```

검증 범위는 correlation의 offset·padding·채널 평균, ERP 투영의 방향·주기성, 카메라 round-trip, 첫 표면 가림·미관측 유지, 비연속/빈 인스턴스 ID, 관측 없는 voxel 평균이다. 실제 30뷰 메쉬 투영은 기존 깊이·유효 hit와 정확히 일치했고 후보 픽셀의 배타적 분해를 확인했다. 이는 모델 변화 탐지 정확도의 검증과 구분한다.

이번 업로드에서 공통 fixture 7종, 공통 규약 테스트 19개, 모델·메쉬·사람 제외·360°/Gaussian 기하 어댑터 테스트 23개와 후보 필터·화면 기본값 Node 테스트 7개를 통과했다. 스냅샷 해시, 화면의 로컬 링크, 30뷰 집계, 모델 커밋·체크포인트 해시와 HTTP 미리보기도 확인했다. 스냅샷은 `.gitattributes`로 줄바꿈 자동 변환을 끄고 원래 바이트를 유지한다.

## 진행 중인 작업

- 사람·시점·가림·관측 범위에 따른 추가 후보를 확인한다.
- 사람 제외 후 추가 후보에 기존 관측검증을 적용하는 연결 방식을 검토한다.
- 공통 Task 5a 입출력과 backend에 통합한다.
- 동일 관측량과 변화 정답이 있는 자료에서 별도로 평가한다.

조사 기록: [파노라마 모델](panoramic-change-models-20261024.md), [메쉬·RGB 모델](mesh-rgb-instance-change-models-20261023.md), [CYWS와 두 촬영 영상 비교](360-video-change-comparison-20261025.md).

이전 2×2 전후 뷰어 갱신 당시 44쌍의 반대 촬영 연결과 176개 방향·입력별 예측 맵, 사람 제외 영역 밖 후보 보존을 확인했다. 쌍별 사진·마스크 파일과 UI의 전후 동시 표시도 확인했다.

360° 갱신에서는 전체 구면 ray·카메라 Sim3·입력 방향 벡터 보존 테스트를 추가했다. 원본 20프레임 해시, 카메라 회전의 정규성, 공통 지점 round-trip, 구역별 36쌍의 70개 ERP 방향 예측 캐시와 360° 전체/확대/시야 연동을 검증했다. 이는 광학적 픽셀 정합이나 탐지 정확도 검증이 아니다.

단일 전 영상 갱신에서 실제 R01 133/142 필터 전후, 후보가 남은 쌍만 선택, 후에서만 검출된 영역 유지, 한 개의 초기 캔버스·숨겨진 겹침 패널, 동일 카메라 RGB 보존, Sim3 이동·스케일, ERP seam 박스의 연속성과 상하 비주기성을 검사했다. 브라우저에서 박스 클릭·전만 보기·박스 위 드래그를 확인하고 `viewer-candidate-before.png`, `viewer-candidate-ghost.png`에 캡처했다.

저장된 스냅샷만으로 박스·겹침을 다시 만들 때는 NumPy·SciPy·Pillow 환경에서 아래 명령을 사용한다. GPU·가중치·원본 전체 메쉬는 필요하지 않다.

```bash
python tools/scenediff_mesh/build_panorama_interactions.py --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/focus-comparison-20261025
```

Gaussian 갱신에서는 원본 두 영상과 두 checkpoint의 서로 다른 SHA-256, 481개 카메라 좌표 변환, 모든 Gaussian 보존, 26개 실제 RGB 입력·float32 확률·사람/관측 마스크의 해시와 픽셀 분해, 기존 양성 원본 면·박스 보존을 확인했다. Gaussian 공분산 회전·축척, 원래 SH 방향, 기존 구역 변환·bias, level ERP 카메라, voxel 후보의 전후 픽셀 보존 테스트 5개를 추가했다. 브라우저에서 두 전체 맵 개수, 기존/SimSaC 박스 선택, 영역 후만·반투명, 사람 제외, 전체 화면과 실제 위치 이동을 확인했다.

```bash
python tools/gaussian_change/validate.py --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-map-comparison-20261025 --metadata-only
```

전체 검증은 `models/`, `predictions/*.npz`와 보존된 원본 메쉬를 갖춘 원래 작업 환경에서 `python tools/gaussian_change/validate.py`로 실행한다. 원본 GS·SimSaC 학습 가중치는 수정하지 않았고 실행한 GPU 작업은 완료됐다. 자세한 단계·서버 경로·한계는 `tools/gaussian_change/README.md`, 실제 실행 소스 해시는 `execution-receipt.json`, 원본 포함 로컬 검증은 `local-verification.json`에 있다.


## 2026-10-26 · K1–K21 경로와 기존 검출 연결

선택한 원본은 `task5b_dorm3_fast_uncapped_20261026_IQK7um`의 Dorm3 평지·개수 상한 해제 계획이다. 첨부 지도 이미지의 SHA-256이 서버 원본과 정확히 같으며, K번호 21개와 저장 링크 20개가 하나의 연결 영역을 이룬다. 기존 13·13·20시점 실험과는 별도 계획이다.

현재 투어는 **기존 Gaussian RGB→SimSaC 변화탐지를 재사용**한다. 사람 제외 5개 3D 영역과 필터 전 6개 영역의 id·bounds·center·전후 픽셀 수를 변경하지 않았다. 새 키프레임별 SimSaC 후보를 표시하는 방식이 아니다. 기존 전역/구역 정합의 공통 좌표에 후보를 유지한 채 현재 카메라로 투영한다.

카메라·가림 검사용으로 21개 포즈에서 양쪽 Gaussian 맵을 동일하게 렌더링했다(42 RGB·예상 깊이). 이 실행에서 SimSaC·사람 검출 추론은 하지 않았다. 현재 위치에서 관측 가능한 기존 영역만 표시하며 중간 이동 위치에는 깊이가 없어 도착 전까지 박스를 숨긴다. 직접 선택은 저장 polyline을 따라 선택 키프레임까지 이동하고, 자동 이동은 저장 방문 순서를 모두 따른다. 박스를 누르면 같은 3D volume의 후 맵이 기본 55%로 겹치며 자동 진행을 중지한다.

실행: `prepare_uncapped_tour.py` → 서버 `render_maps.py` → `build_uncapped_tour.py`. 감사: `validate_uncapped_tour.py`(메타데이터만 검증 시 `--metadata-only`). 원본 계획·요청 포즈·K번호·20개 링크 끝점, 42 이미지 해시, 두 원본 Gaussian의 export·변환, 원본 후보 보존, 예상 깊이 기반 가림 검사를 확인한다. 실제 변화 정확도·객체 관측 커버리지·보행 안전은 미검증이며, 두 맵은 서로 다른 층 Dorm2·Dorm3다.

현재 화면은 `gaussian-uncapped-tour-20261026/index.html`이며 이전 키프레임 URL도 이 화면으로 연결한다. 46시점 추론 실험은 원래 run과 `experiment-46.html`에 보존한다. Git 스냅샷에는 뷰어·RGB·포즈·후보·감사 기록을 포함하고 원본 Gaussian PLY·깊이 NPZ는 제외한다.


## 2026-10-26 · 분할 객체와 유형 판단 근거

## 인스턴스 자동 분리 및 이동 자동 분류 · 2026-10-26

기존 Gaussian RGB SimSaC의 5개 영역을 유지하면서, 기존 영상 인스턴스 트랙과 의미 분할 메쉬로 추가·이동·소실 유형을 자동 추론한다. 수동 유형 보정 입력을 제거했다. 같은 자리에 남아 있는 객체는 기존 자동 대응으로 보호하며, 병합 메쉬의 면을 다중 시점 3D 마스크 표본으로 자동 분리한다. 잔여 객체는 구역·종류가 같은 대상끼리 객체별 축척 없이 yaw+translation 강체 형상, 양쪽 기존 위치의 표면 잔존율, 도착 위치의 독립 빈 공간 관측 및 경쟁 대응 점수 차이로 1:1 할당한다. 애매한 대응은 유형 불확실로 남긴다.

R12는 S0-010(758면)에서 I0-005에 해당하는 원본 135면을 자동 분리하고 S1-024(137면), 영상 트랙 I1-067과 대응한다. 다른 네 의자의 같은 위치 대응은 재사용하지 않는다. 자동 대응은 `automatic-matching.json`에 점수·탈락 이유와 함께 저장한다. 사람의 오른쪽→왼쪽 설명이나 구역별 정답 ID를 분류 입력으로 사용하지 않는다.

이동 박스를 누르면 전 Gaussian 위에 청록 출발 객체와 주황 도착 객체의 색 그림자·전→후 화살표가 표시된다. R12의 출발 색은 의자 줄 전체가 아닌 자동 분리 조각에 입힌다. 화살표는 위치 대응이며 실제 이동 궤적은 아니다. 21개 키프레임·20개 링크·기존 정합·원래 SimSaC 박스 및 확률은 그대로 둔다. 객체 물리 ID와 유형은 자동 추정이며 시간 변화 정답으로 검증된 결과는 아니다.

### 구역별 중복 제거 · 같은 변화의 관측 링크

변화 ID와 관측 키프레임을 분리했다. 동일 원본 후보 ID 또는 동일 자동 객체 대응에 따른 검출 alias는 하나의 이벤트이며, 키프레임이 바뀌어도 이벤트를 추가하지 않는다. 구역별 목록과 합계는 이벤트 ID의 고유 개수다. 현재 필터 후 3구역·고유 후보 5개·관측 링크 13개이며, K1/K2/K3/K5/K6의 구역1은 동일 후보 2개를 관측한다. 필터 전 6개 원본 검출도 같은 자동 이동 대응을 공유하는 2개를 하나로 연결해 고유 이벤트는 5개다.

`change-regions.json`은 원본 후보→이벤트, 이벤트→공간 구역, 키프레임→관측 ID를 기록한다. 가까우면서 같은 시점에서 관측되는 다른 객체는 공간 구역만 공유하고 이벤트는 별도로 유지한다. 기존 검출의 id·bounds·픽셀 수, 등록 변환 및 21개 키프레임·20개 이동선은 그대로 둔다. 화면은 구역별 목록을 한 번만 만들고 이동 시 현재 관측 여부와 박스를 갱신한다.
