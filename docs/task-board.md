# Task 작업표

사람별 배정 없이 바로 작업을 나눌 수 있는 기준표입니다. 각 담당자가 자신의 task README와 issue에 진행 상황을 기록합니다.
아래 구현 상태는 이 scaffold가 만들어진 시점 기준입니다. 모델 구현이 main에 통합되면 해당 행을 갱신합니다.

| Task | 첫 구현 목표 | 제공된 독립 개발 입력 | 완료 시 확인할 것 | 구현 상태 |
|---|---|---|---|---|
| 1a | RGB 영상 프레임 추출 | `before/video.json` 및 작은 MP4 | 원본 index·시각·해상도 일치 | 미구현 |
| 1b | 360° → perspective 변환 | `panorama/video.json` 및 작은 MP4 | 뷰 방향·가상 intrinsics·출처 보존 | 미구현 |
| 2 | LingBot-Map 출력 adapter | `before/frames.json` 및 프레임 | 점군·카메라·좌표 단위·깊이 연결 | 미구현 |
| 3 | MV3DIS 출력 adapter | `before/reconstruction.json` 및 합성 점군 | 원본 점 ID 복원·객체 ID·미할당 처리 | 미구현 |
| 4 | 객체별 GLB/OBJ export | `before/instances.json` 및 라벨 CSV | 객체 로컬/월드 변환·재로딩 | 미구현 |
| 5a | 정합 후 객체 변화 비교 | `before/objects.json`, `after/objects.json` | 회차별 ID 대응·정합 실패·가림 구분 | 미구현 |
| 5b | 객체 대표 프레임 순위화 | `before/objects.json` 및 상위 프레임 정보 | 객체→frame_id→원본 시각 추적 | 미구현 |

경로의 기준은 `examples/fixtures/`입니다. 합성 점군은 8개 점에 불과하므로 실제 모델 추론용 데이터셋이 아닙니다.
Task 2·3 등의 모델 성능 검증에는 실제 촬영 데이터와 모델 요구 조건을 충족하는 별도 입력이 필요합니다.

## 통합 순서

1. 각 task는 자체 구현·단위 실험을 병렬로 수행합니다.
2. 먼저 `1a → 2`로 한 회차의 실제 프레임·점군·카메라 대응을 검증합니다.
3. `2 → 3 → 4`로 같은 객체 ID가 점군과 mesh까지 연결되는지 확인합니다.
4. Task 5a는 두 회차 데이터로, Task 5b는 한 회차 데이터로 각각 통합합니다.
5. Task 1b는 같은 frames 규약으로 Task 2에 연결합니다.

이 순서는 개발 담당자의 착수 순서를 제한하지 않습니다. 공통 규약을 바꾸는 작업은 소비 task와 함께 합의합니다.
