# Task 5a · 모델 비교 진행 중

2026-10-25 기준, **기존 정합오차·관측검증 방식과 SimSaC·ViewDelta의 변화 후보를 비교하고 있다.** 현재까지의 코드, 측정 결과 JSON, 비교 화면, 가중치 로딩 검사와 캡처를 이 브랜치에 보관한다. 공통 Task 5a backend의 통합 완료를 뜻하지 않는다.

## 비교 화면 실행

저장소 루트에서 다음 명령을 실행한다. 저장된 결과를 보는 데 GPU나 모델 설치는 필요하지 않다.

```bash
python -m http.server 4190 --bind 127.0.0.1 --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025
```

브라우저에서 [http://127.0.0.1:4190/focus-comparison-20261025/index.html](http://127.0.0.1:4190/focus-comparison-20261025/index.html)을 연다. GitHub에서 HTML 링크를 누르면 소스 보기로 열리므로 로컬 HTTP 서버를 사용한다.

| 화면 | 내용 |
|---|---|
| `focus-comparison-20261025/index.html` | 기존 정합오차·관측검증 / SimSaC / ViewDelta 중심 비교. 기존 후보·보류·사람 겹침 필터, 상대 RGB, 입력 방식 전환 |
| `model-comparison-20261025/index.html` | 앞서 실행한 SceneDiff 두 입력 경로까지 포함한 비교 |
| `paired/index.html` | 동일 RGB에서 SceneDiff 원본 Pi3/SAM 입력과 외부 메쉬·인스턴스 입력의 후보 차이 |

각 화면의 `REPORT.md`, `comparison.json`, `validation.json`에 수치·조건·검증을 보관했다. [스냅샷 파일 해시](../src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/snapshot-manifest.json)로 복사된 결과 파일을 확인할 수 있다.

## 현재 입력과 비교 방법

- Dorm2 / Dorm3의 서로 다른 층을 비교한 공간 차이 파일럿이다. 시간 변화의 물리적 정답은 없다.
- R01·R07·R08의 30개 방향 RGB를 사용한다. 대응하는 전체 ERP는 20프레임이다. 방향 RGB는 기존 SceneDiff 입력과 바이트·해시가 같다.
- 두 모델은 **동일 방향 RGB**와 **전체 ERP → 동일 방향 결과 투영**을 각각 실행했다. 원본 모델의 전처리를 유지했으며 구면 전용 모델로 바꾼 것은 아니다.
- 두 모델의 상대는 기존 메쉬 SceneDiff의 view-overlap으로 고정했다. 모든 비영 상대의 예측확률을 평균하고 **0.5 초과**를 후보로 표시했다. 기존 후보 마스크로 출력을 제한하지 않았다.
- 기존 방식은 저장된 `occupancy-evidence-v2` 자동 판정을 사용한다. 이 판정은 전체 촬영을 사용했으므로 두 모델과 관측량을 맞춘 성능 평가가 아니다.
- 기존 양성은 후보를 만든 트랙의 원본 면만 first-hit 투영한다. 같은 종류의 주변 메쉬, 보류·문맥·억제·판정 연결 없음·mesh hit 없음은 따로 기록한다. 보류를 변화 없음으로 취급하지 않는다.

기존 전체 13구역에는 의자 관련 후보 묶음 2개가 있고 현재 모델 범위에는 R08의 E001 / I1-096이 포함된다. 후보 묶음을 물리적 객체 변화 건수로 해석하지 않는다.

## 이번 결과에서 관찰한 차이

R08의 기존 후보 트랙 표면은 3개 관측 화면에서 중복 관측을 포함하여 4,681픽셀 보인다.

| 모델 입력 | SimSaC와 기존 후보 표면의 겹침 | ViewDelta와 기존 후보 표면의 겹침 |
|---|---:|---:|
| 동일 방향 RGB | 1,127px | 0px |
| 전체 ERP 입력 후 투영 | 3,376px | 0px |

이는 기존 후보 표면과의 동의 정도이며 정확도·재현율이 아니다. SimSaC는 기존에 보류된 표면도 선택했다. R07에서는 두 모델이 촬영자 머리를 선택하는 사례가 있으며 ViewDelta가 주변 문·벽을 선택한 화면도 있다. 새 모델 후보를 바로 시설물 변화로 확정하지 않았다. 같은 프레임의 기존 사람 분할과 겹침은 확인 단서로만 표시하고 원본 모델 후보에서 제거하지 않았다.

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
| `build_focus_comparison.py`, `focus.html`, `focus.js` | 저장된 기존 판정과 두 모델의 집중 비교 화면 생성 |
| `frozen_object_event_policy.py` | 기존 판정 정책의 소스 스냅샷. 이 화면에서 판정을 다시 계산하지 않음 |

실제 추론·화면 재생성에는 별도로 보존된 입력과 확률 캐시가 필요하다. 일부 서버용 스크립트는 실행 당시 `/workspace/...` 배치를 고정하므로 새 환경에서는 외부 소스·가중치·입력 경로를 맞춰야 한다. 이 커밋만으로 원본 전체 촬영을 재추론할 수 있다고 설명하지 않는다. `run.py`의 이전 기하/RGB 실험은 현재 요청한 기존 정합오차·관측검증 기준선이 아니다.

현재 로컬 입력이 같은 배치로 준비된 환경에서는 다음 명령으로 집중 비교를 다시 만든다.

```bash
python tools/scenediff_mesh/build_focus_comparison.py --root outputs/scenediff-mesh-rgb-20261024 --prior outputs/registration-benchmark-20261019/new-meshfirst-20261022
```

두 모델 어댑터 검증은 NumPy·SciPy·Pillow·PyTorch, 외부 메쉬 검증은 추가로 Open3D·PyYAML·timm 환경을 사용한다. 실제 모델 추론 환경은 각 공식 소스의 의존성도 필요하다. 무거운 의존성이 있는 실험 검증은 공통 CPU 규약 검사와 별도로 실행한다.

```bash
python -m unittest discover -s tests/task_5a_change_detection/model_comparison -v
```

검증 범위는 correlation의 offset·padding·채널 평균, ERP 투영의 방향·주기성, 카메라 round-trip, 첫 표면 가림·미관측 유지, 비연속/빈 인스턴스 ID, 관측 없는 voxel 평균이다. 실제 30뷰 투영은 기존 깊이·유효 hit와 정확히 일치했고 후보 픽셀의 배타적 분해를 확인했다. 이는 모델 변화 탐지 정확도의 검증과 구분한다.

이번 업로드에서 공통 fixture 7종, 공통 규약 테스트 19개, 모델·메쉬 어댑터 테스트 8개를 통과했다. 스냅샷 해시, 화면의 로컬 링크, 30뷰 집계, 모델 커밋·체크포인트 해시와 HTTP 미리보기도 확인했다. 스냅샷은 `.gitattributes`로 줄바꿈 자동 변환을 끄고 원래 바이트를 유지한다.

## 진행 중인 작업

- 사람·시점·가림·관측 범위에 따른 추가 후보를 확인한다.
- 모델의 추가 후보에 기존 관측검증을 적용하는 연결 방식을 검토한다.
- 공통 Task 5a 입출력과 backend에 통합한다.
- 동일 관측량과 변화 정답이 있는 자료에서 별도로 평가한다.

조사 기록: [파노라마 모델](panoramic-change-models-20261024.md), [메쉬·RGB 모델](mesh-rgb-instance-change-models-20261023.md), [CYWS와 두 촬영 영상 비교](360-video-change-comparison-20261025.md).
