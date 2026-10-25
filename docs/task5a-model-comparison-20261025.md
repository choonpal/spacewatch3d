# Task 5a · 모델 비교 진행 중

2026-10-25 기준, **기존 정합오차·관측검증 방식과 SimSaC·ViewDelta의 변화 후보를 비교하고 있다.** 현재까지의 코드, 측정 결과 JSON, 비교 화면, 가중치 로딩 검사와 캡처를 이 브랜치에 보관한다. 공통 Task 5a backend의 통합 완료를 뜻하지 않는다.

## 현재 화면: 전 360° 후보 박스와 선택 영역 반투명 비교

처음에는 **전(Dorm2) 360° 영상 하나**를 둘러보기 모드로 표시한다. 기존 정합오차·관측검증과 SimSaC의 후보 박스를 미리 그린다. 박스를 누르면 해당 영역 안에만 후(Dorm3) RGB를 55% 겹친다. 전만 보기·반투명·선택 영역 후만 및 슬라이더로 비교한다. 드래그·방향키·확대·전체 화면을 지원하고, 박스 위에서도 드래그할 수 있다.

목록 기본값은 사람 제외 후 선택한 방식의 후보가 남은 쌍이다. 전/후 후보 수를 표시하며 전체 쌍·사람 제외 비교에서는 0인 쌍도 볼 수 있다. R01 전 133 ↔ 후 142의 SimSaC는 원본 전 0 / 후 156픽셀이고, 후의 156픽셀이 모두 사람 제외 영역에 들어가 표시 후보 0이다. 필터를 끄면 복원된다. 현재 기본값은 36쌍 중 28쌍이다.

전후 양쪽의 후보를 전 영상에 표시한다. 후에서만 검출된 영역은 후 메쉬 깊이와 저장된 카메라 정합으로 전으로 투영한다. 후 깊이가 없으면 방향만 맞춘 근사 투영임을 기록한다. 사람 제외 시 원본 후의 사람 영역뿐 아니라 투영 위치의 전 사람 영역도 제외한다. 이 때문에 전 시점 박스에 포함된 후 후보 수가 원본 필터 후 수보다 작을 수 있다. 박스는 1024×512 ERP의 연결 후보 영역을 3픽셀 여유로 묶으며 객체 단위의 확정 판정이 아니다. 모든 보존 후보 픽셀이 정확히 한 박스에 포함되는지 검사한다.

반투명 후 영상은 전 메쉬의 첫 표면 깊이와 저장된 구역 Sim3·이동 편향·카메라 포즈로 원래 후 RGB를 샘플링한다. 전 깊이가 없는 곳은 겹침이 투명하다. 새 관측 내용을 생성하지 않는다. 촬영 위치·메쉬·가림·정합 잔차는 남고, 겹침 경계 차이를 객체 이동의 증거로 해석하지 않는다.

전체 ERP 20프레임·구역별 36쌍을 사용한다. SimSaC는 기존 전체 ERP(1024×512) 양방향 예측이며 새 추론·재학습은 없다. `panorama-review.json`은 카메라·원본 수치, `panorama-interactions.json`은 후보 박스와 겹침, `interaction-validation.json`은 픽셀 분해·입력 해시를 담는다. 기존 224px 방향 비교는 `cropped-review.html`, 이전 다중 상대 평균은 `comparison.json`과 `REPORT.md`로 보관한다.

## 비교 화면 실행

저장소 루트에서 다음 명령을 실행한다. 저장된 결과를 보는 데 GPU나 모델 설치는 필요하지 않다.

```bash
python -m http.server 4190 --bind 127.0.0.1 --directory src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025
```

브라우저에서 [http://127.0.0.1:4190/focus-comparison-20261025/index.html](http://127.0.0.1:4190/focus-comparison-20261025/index.html)을 연다. GitHub에서 HTML 링크를 누르면 소스 보기로 열리므로 로컬 HTTP 서버를 사용한다.

| 화면 | 내용 |
|---|---|
| `focus-comparison-20261025/index.html` | 전 360° 영상의 기존 / SimSaC 후보 박스. 선택 영역 후 반투명 비교·사람 제외 |
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

이번 업로드에서 공통 fixture 7종, 공통 규약 테스트 19개, 모델·메쉬·사람 제외·360° 기하/재투영 어댑터 테스트 18개와 후보 필터·화면 기본값 Node 테스트 7개를 통과했다. 스냅샷 해시, 화면의 로컬 링크, 30뷰 집계, 모델 커밋·체크포인트 해시와 HTTP 미리보기도 확인했다. 스냅샷은 `.gitattributes`로 줄바꿈 자동 변환을 끄고 원래 바이트를 유지한다.

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
