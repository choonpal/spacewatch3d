# point cloud 생성기 v2

v2(2.0.0)는 v1의 PanoVGGT 추론·정밀 마스크·프레임별 재보정 뒤에 **전체 지도 연결 성분 중 10,000점 이하 삭제**를 추가합니다. v1 코드는 별도 폴더에 보존합니다. v1 알고리즘 설명은 `V1_REFERENCE.md`, 원래 코드 해시는 `V1_RELEASE.json`, v2 해시는 `RELEASE.json`에 있습니다.

## 제거 규칙

- 최종 voxel=0.005 병합 점군 전체에 적용합니다. 창별·프레임별 제거가 아닙니다.
- `floor(XYZ / 0.02)`로 복셀을 만들고, 면·모서리·꼭짓점이 닿는 26방향 점유 복셀을 하나의 성분으로 연결합니다.
- 좌표계는 원래 복원의 world 축, 원점은 `[0,0,0]`입니다. 눈으로 결과를 보고 회전하거나 규칙을 바꾸지 않습니다.
- 크기는 복셀 수가 아니라 **그 성분에 포함된 병합 후 실제 점 수**입니다. 9,999점과 10,000점은 삭제하고 10,001점부터 보존합니다.
- 남긴 점의 XYZ·RGB와 순서를 그대로 복사합니다. 새 점을 생성하거나 위치를 이동하지 않습니다.
- 이 버전은 반경 기반 이상치 제거(ROR)나 통계적 이상치 제거(SOR)를 추가하지 않습니다.
- `0.02`와 `0.005`는 첫 입력 프레임의 예측 깊이 중앙값을 1로 둔 정규화 단위입니다. 미터·센티미터가 아니며, 높이 2.7m 가정도 적용하지 않습니다.

실제 물체라도 분리된 성분의 점 수가 10,000점 이하면 삭제됩니다. 큰 고립 노이즈나 유지되는 성분과 연결된 노이즈는 남을 수 있습니다. 복셀 연결은 정확한 유클리드 거리 반경 연결과 다르고, 축 방향과 격자 원점에 영향을 받습니다.

## 입력과 처리 조건

`configs/dorm2.json`, `configs/dorm3.json`은 원본 MP4의 SHA256에 연결된 RGB·보정 전 마스크 묶음을 사용합니다. dorm2는 233프레임/46창, dorm3는 248프레임/49창입니다. 두 영상 모두 명목 0.5초 간격, 1036×518, FP32, 10프레임 창/중복5, 정밀 마스크, pixel_step=1, 출력 병합 voxel=0.005입니다.

새 실행에서는 PanoVGGT와 LoFTR 추론, 추가 매칭, 프레임별 정합을 다시 수행합니다. RGB와 기존 정밀 마스크는 검증 후 재사용합니다. 원본 MP4만으로 임의의 새 영상의 촬영자 마스크까지 생성하는 범용 파이프라인은 아닙니다.

dorm3의 사전 승인 경계 네 쌍은 모든 매칭 복구 단계 뒤에도 끊겼을 때만 원래 모델의 상대 자세로 연결하며 결과에 미검증으로 표시합니다. dorm2는 미검증 연결을 허용하지 않습니다. 실행 중 임의로 매칭 기준을 낮추거나 마스크를 편집하지 않습니다.

## 서버에서 실행

모든 모델 추론은 사용자 GPU 서버에서만 실행하며 GPU 1 한 장을 사용합니다. GPU가 바쁘면 오류로 중단합니다. 출력 폴더를 새로 지정하면 새 추론을 수행합니다.

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline_v2/run.py \
  --config /workspace/panovggt_pipeline_v2/configs/dorm2.json \
  --output /workspace/dorm2_pointcloud_v2_new
```

두 영상을 같은 GPU에서 순차 처리하는 실행기:

```bash
/workspace/panovggt_env/bin/python /workspace/panovggt_pipeline_v2/batch.py \
  --jobs /workspace/panovggt_pipeline_v2/configs/batch_dorm2_dorm3.json \
  --report-directory /workspace/pointcloud_v2_batch_20261017
```

새 실험에서는 작업 목록의 출력 경로도 새 경로로 바꿉니다. 같은 `run.py` 명령은 해시 검증 후 완료된 단계를 재사용해 재개합니다. 설정·코드·입력이 달라지면 기존 폴더 재개를 거부합니다. `--replay-cache`는 별도의 명시적 CPU 재생 모드이며 이번 두 영상의 실행에서는 사용하지 않습니다.

## 출력과 시간 기록

| 파일 | 내용 |
|---|---|
| `result/map.ply` | 10,000점 이하 성분을 제거한 최종 전체 점군 |
| `unfiltered_result/map.ply` | 이번 실행에서 새로 생성한 제거 전 전체 점군 |
| `component_filter/removed.ply` | 삭제된 모든 점 |
| `component_filter/component_ids.npy` | 제거 전 점 순서에 대응하는 성분 ID |
| `component_filter/component_point_counts.npy` | 성분별 실제 점 수 |
| `component_filter/keep_mask.npy` | 제거 전 점 순서에 대응하는 보존 여부 |
| `component_filter/components.csv` | 성분별 크기와 보존 여부 |
| `component_filter/report.json` | 제거 규칙, 점 수, 전체 검증, 필터 시간 |
| `result/validation.json` | 최종 점 수·SHA256·미검증 연결·한계 |
| `viewer/` | 최종 점군에서 균일 추출한 최대 750,000점 미리보기 |
| `run_report.json` | 단계별 시간, 실행 모드, 전체 실행 시간 |
| `inference_timing.json` | PanoVGGT 모델 연산 시간 |
| 배치 폴더의 `batch_report.json` | 입력 검증을 포함한 각 프로세스 벽시계 시간 및 순차 배치 총시간 |

성분 계산은 메모리 사용을 제한하는 슬랩 단위 CPU 연산이지만 슬랩 경계의 모든 26방향 연결을 병합하므로 전체 지도와 동일한 성분을 구합니다. 원본 지도와 보존/삭제 파일을 다시 읽어 모든 좌표·RGB 바이트와 점 수를 검증합니다. 실제 지도 정답과 비교한 기하 정확도 검증은 아닙니다.

## 테스트

서버 Python 환경에서 15개 테스트를 통과했습니다. 슬랩과 전체 밀집 격자 연결 성분의 일치, 음수 좌표·중복 점·대각선·빈 슬랩, 9,999/10,000/10,001점 경계, 전체 출력 XYZ/RGB 보존을 포함합니다. v1의 마스크·최적화·경계 정책·재개 검증 테스트도 유지합니다.

```bash
cd /workspace/panovggt_pipeline_v2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /workspace/panovggt_env/bin/python -m unittest discover -s tests -v
```
