# point cloud 생성기 v1

사용자 지정 이름: **point cloud 생성기 v1** · 버전 `1.0.0` · 지정일 2026-10-17. 검증된 코드 식별 정보와 결과 위치는 [RELEASE.json](RELEASE.json)에 기록했습니다.

동일한 입력과 명시적인 설정으로 마스크 보정 → 추론 → 영상 매칭 → 추가 매칭 → 프레임별 자세 보정 → 경계 연결 → 전체 PLY → 뷰어를 실행합니다. 실행 도중 Codex가 로그를 보고 다음 알고리즘이나 파라미터를 선택할 필요가 없습니다.

이 패키지의 입력은 **동결한 RGB 248장과 보정 전 마스크, 원본 영상, 모델 파일**입니다. 원본 MP4 한 개만으로 새로운 영상의 촬영자 마스크까지 자동 생성하는 범용 프로그램은 아닙니다. 기존 SAM·몸 영역·머리 보정 결과를 보정 전 마스크 입력으로 보관하고, 이번에 요청한 시각 검토 후 보정 규칙(팽창·다각형)을 설정으로 분리했습니다. 새 영상에서는 해당 입력과 설정을 새로 준비해야 합니다.

## 바로 실행

서버에 설치된 위치는 `/workspace/panovggt_pipeline`입니다. 비밀번호는 코드·설정·명령에 넣지 않습니다.

```bash
ssh -p 4444 root@155.230.28.172
```

서버에서 다음 명령 하나를 실행합니다. 이 모드는 저장된 원시 예측과 매칭을 해시 검증 후 재사용하고, **관측 구성·최적화·지도 출력은 처음부터 다시 계산**합니다. GPU를 사용하지 않으며 캐시가 없으면 실패합니다.

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline/run.py \
  --config /workspace/panovggt_pipeline/configs/dorm3.json \
  --output /workspace/dorm3_repro_my_run \
  --replay-cache /workspace/panovggt_repro_inputs/dorm3/replay_cache.json
```

PanoVGGT와 LoFTR도 새로 실행하려면 `--replay-cache`를 빼고 새 출력 폴더를 지정합니다. GPU 1 한 개만 사용합니다. 다른 작업이 해당 GPU를 사용 중이면 중단하며 다른 GPU나 Colab으로 전환하지 않습니다. 서버 Docker 컨테이너가 바뀌면 `execution.server_container_hostname`을 실제 접속한 서버의 `hostname` 값으로 갱신해야 합니다.

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline/run.py \
  --config /workspace/panovggt_pipeline/configs/dorm3.json \
  --output /workspace/dorm3_repro_fresh
```

같은 명령을 다시 실행하면 완료된 단계의 파일 해시를 검증하고 재개합니다. 설정·코드·입력·환경이 바뀌면 같은 출력 폴더에 이어 쓰지 않고 오류를 냅니다. 이때 새 출력 폴더를 지정합니다. 중단된 매칭/출력 단계는 이전 부분 파일을 `interrupted/`에 보존하고 다시 실행합니다. 추론 구간은 구간별 NPZ와 메타데이터 해시가 맞을 때만 이어서 사용합니다.

`--validate-only`를 추가하면 입력과 캐시만 검증하고 실행하지 않습니다. 모든 경로는 서버 경로입니다. 실행 중 `status.json` 및 `logs/`를 보면 현재 단계와 오류를 알 수 있습니다. 예약 작업은 생성하지 않습니다.

## 설정 파일

`configs/dorm3.json`이 처리 정책의 기준입니다. 코드 안의 `248` 또는 `경계 4개`를 수정할 필요가 없습니다.

| 설정 | 의미 |
|---|---|
| `input` | 원본 SHA256, RGB 크기 1036×518, 248프레임, 명목 간격 0.5초, 입력 묶음 경로 |
| `mask.periodic_dilation_kernel` | 가로·세로 101×31px 타원 팽창. ERP 좌우 연결을 고려 |
| `mask.polygons` | 0부터 세는 프레임 245~247의 추가 제외 다각형. 이전 시각 판단을 그대로 명시 |
| `mask.coordinate_size` | 다각형 좌표 기준 크기. 다른 해상도에 몰래 적용하지 않음 |
| `mask.max_fraction` | 과도한 마스킹을 막는 면적 상한 |
| `model` | PanoVGGT 체크포인트/소스 버전, FP32, 10프레임 묶음·중복5, 깊이/상단 필터 |
| `matching` | LoFTR 모델·원근 뷰 크기·신뢰도·시간 간격, SIFT와 원본 해상도 보강 정책 |
| `registration` | 학습/평가 분리, RANSAC, 깊이 유효성, 상대 스케일 및 연결 채택 기준 |
| `optimization.attempts` | 300→600→1200회 재시도, 각 시도의 LSMR 허용오차와 반복 상한 |
| `boundary.policy` | `fail`: 연결 실패 시 중단. `original_relative_pose`: 명시적으로 허용한 경계만 원래 자세로 연결 |
| `boundary.allowed_pairs` | 사용자 승인 경계 210→211, 221→222, 235→236, 245→246. 추가 매칭 후에도 끊긴 경계에만 적용 |
| `boundary.max_unverified_bridges` | 미검증 연결 최대 개수. 미등록 경계와 연속적이지 않은 성분은 중단 |
| `output` | `pixel_step`, voxel 크기, 미리보기 점 수, 파일 이름 |

마스크 다각형은 사전에 사람이 정한 실험 설정입니다. 실행 중 자동으로 새 다각형을 그리거나 검증 실패를 통과시키도록 기준을 낮추지 않습니다. 보수적 마스크는 몸 주변 배경 일부도 제외할 수 있습니다.

## 단계 순서와 이전 오류 방지

1. 원본·RGB·보정 전 마스크의 SHA256을 검사하고 설정에 따라 최종 마스크를 생성합니다.
2. PanoVGGT 추론 또는 명시적인 원시 예측 캐시 재생을 수행합니다.
3. 공유 프레임의 동일 픽셀 깊이 비율로 묶음 스케일을 맞춥니다.
4. 기본 매칭 완료 → 관측 구성 및 연결 검사.
5. 연결이 끊겼으면 방향 간 추가 매칭 **완료** → 관측 재구성.
6. 여전히 끊겼으면 SIFT 보강 **완료** → 관측 재구성.
7. 여전히 끊겼으면 현재 연결 상태에서 원본 해상도 매칭 쌍을 자동 선정 → 추가 매칭 **완료** → 관측 재구성.
8. 각 연결 성분의 프레임 자세를 최적화합니다. 실패한 시도의 출력은 같은 실행의 다음 시도 시작값으로만 사용합니다.
9. 모든 성분이 수렴하면 경계 정책을 적용하고 점군을 출력합니다. 허용하지 않은 경계나 미수렴 결과에서는 출력을 중단합니다.

각 단계는 동기적인 하위 프로세스입니다. 종료 코드와 산출물 해시가 기록되기 전에는 다음 단계로 진행하지 않습니다. 추가 매칭 파일을 생성하는 중에 이전 관측을 최적화하던 기존 오류를 방지합니다.

`observation_phases/{base,cross,sift,hires}/observation_inputs.json`에 **실제로 읽은 파일 경로와 SHA256**이 남습니다. `edge_audit.json`의 `match_source`는 파일을 읽는 순간 기록합니다. 파일이 존재한다는 이유로 나중에 출처를 덧붙이지 않습니다. 중간 관측은 단계별로 보존되므로 재시작해도 어느 단계의 관측인지 섞이지 않습니다.

## 프레임별 재보정의 의미

`workers/refine.py`의 `Problem`이 각 프레임의 위치 3개와 회전 3개, 총 6개 변수를 최적화합니다. 각 연결 성분의 첫 프레임은 좌표계 기준으로 고정합니다. 동일한 장면 지점에 해당하는 두 프레임의 3D 대응점을 각각의 카메라 자세로 변환했을 때 서로 가까워지도록 robust least squares를 풉니다.

`workers/optimize.py`가 초기 자세 구성과 재시도를 자동 실행합니다. `workers/connect.py`가 허용된 미검증 경계에서 모델의 원래 상대 자세를 계산하고 후속 성분에 동일한 변환을 적용합니다. 사람이 점이나 벽을 움직여 맞추는 작업은 없습니다.

깊이맵의 픽셀별 형상은 고정하고, 묶음 간 깊이 스케일과 프레임별 카메라 자세를 보정합니다. **전체 bundle adjustment나 장거리 루프 폐쇄를 구현한 것은 아닙니다.** voxel=0.005는 첫 프레임의 깊이 중앙값을 1로 둔 정규화 단위이며 미터가 아닙니다.

## 입력/캐시 묶음 생성

이 단계는 서버에 이미 한 번 수행했습니다. `/workspace/panovggt_repro_inputs/dorm3/`에 해시 목록과 보정 전 마스크가 있습니다. RGB·원본 영상·원시 추론/매칭 데이터는 기존 서버 저장 위치를 참조하므로 해당 위치도 보존해야 합니다.

다른 저장 위치로 구성할 때는 다음 명령으로 동결합니다. 기존 최종 지도나 잘못된 관측 파일은 가져오지 않습니다. 보정 전 마스크에서 설정을 적용한 결과가 기존 최종 마스크와 정확히 일치하는지도 검사합니다.

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline/freeze_inputs.py \
  --config /workspace/panovggt_pipeline/configs/dorm3.json \
  --legacy-run /workspace/dorm3_frame_refine_05s_20261016 \
  --base-masks /workspace/dorm3_pre_review_masks \
  --output /workspace/panovggt_repro_inputs/dorm3
```

이 명령은 빈 대상 폴더만 허용합니다. 경로를 변경했으면 `input.bundle`도 변경합니다. 모델이나 마스크·매칭 설정이 다른 캐시는 재사용할 수 없습니다.

## 출력과 재현 범위

- `result/map.ply`: 전체 지도. `result/validation.json`: 점 수·SHA256·단위·미검증 연결.
- `result/preview.ply`, `result/overview.png`, `viewer/`: 표본 미리보기.
- `registration_report.json`: 평가 지표와 경계 연결 정책.
- `components/`: 재시도별 자세와 종료 이유.
- `run_identity.json`: 설정·입력·코드·환경 식별 정보. `stages/`: 단계별 완료 기록.
- `run_report.json`: 전체 단계·시간·재사용 여부. 캐시 재생 시간은 새 GPU 추론 시간과 다릅니다.

뷰어는 서버에서만 띄워 두고 SSH 포워딩으로 접근할 수 있습니다.

```bash
# 서버
/workspace/panovggt_env/bin/python -m http.server 8797 --bind 127.0.0.1 \
  --directory /workspace/dorm3_repro_my_run
# 로컬 별도 터미널
ssh -p 4444 -L 8797:127.0.0.1:8797 root@155.230.28.172
# 로컬 브라우저: http://127.0.0.1:8797/viewer/
```

이전 결과에서 빠졌던 고해상도 매칭을 반영하므로 **기존 65,547,557점 지도와 동일한 결과를 만드는 것이 목표가 아닙니다.** 올바르게 고정한 새 처리 규칙을 반복 실행하는 것이 목표입니다. 신선한 GPU 연산의 비트 단위 동일성은 보장하지 않습니다. 하드웨어·라이브러리 차이는 수치나 임계값 부근의 채택 결과에 영향을 줄 수 있습니다.

## 검증

```bash
/workspace/panovggt_env/bin/python -m unittest discover \
  -s /workspace/panovggt_pipeline/tests -v
```

회귀 테스트는 마스크 좌우 연결, 프레임별 다각형, 미등록 경계 거부, 재시도 시작값 전달, 재시도 실패 시 중단, 추가 매칭 우선순위, 단계 완료 대기, 변경된 파일·심볼릭 링크 거부, 단계별 관측 복원, 알려진 합성 카메라 자세 복원을 확인합니다. 실제 서버 실행 검증 결과는 `VALIDATION.json`에 별도로 기록합니다.

### 이번 실제 검증 결과

- 로컬 전용 CPU 환경과 서버에서 회귀 테스트 12개 통과.
- 서로 다른 빈 폴더에서 동일한 실행 명령으로 2회 완료. 마스크·대응점·카메라 자세가 동일하고, 전체 PLY도 SHA256까지 동일함.
- 각 결과 65,602,283점. PLY SHA256: `3dafa5bd86e167826856e5e4f69b924ad6168fea61f4bbfe7863a69da0be5d90`.
- 105·110.5·117.5초 경계는 추가 매칭으로 채택되고, 약 122.5초의 한 경계만 설정된 원래 자세로 연결됨.
- 완료된 실행에 같은 명령을 다시 호출하여 16개 단계 모두 파일 검증 후 스킵되는 것을 확인.
- 검증은 저장된 예측·매칭의 CPU 재현 모드 기준. 신규 GPU 추론의 반복 동일성을 측정한 것은 아님.
- `run_report.json`의 `invocation_seconds`는 단계 실행 구간의 시간이며 시작 시 입력/캐시 해시 검사 시간은 제외됨.
- 상세 근거: `VALIDATION.json`, `validation/reproducibility.json`.

두 독립 실행의 결과는 다음 명령으로 다시 비교할 수 있습니다.

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline/tests/check_replay.py \
  /workspace/dorm3_repro_validation_20261016 \
  /workspace/dorm3_repro_repeat_20261016 \
  --output /workspace/panovggt_pipeline/reproducibility.json
```
