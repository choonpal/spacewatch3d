# Task 5b · 중앙 경로 생성 및 360° 투어 키프레임 위치 선정

3D 공간과 객체별 메시·메타데이터에서 중앙 경로를 생성하고, 공간 커버리지와 객체 중요도 및 중복을 고려해 가상 촬영 위치를 선정한다. 무거운 3D 맵은 서버에서 처리하며, 사용자에게 보여줄 미디어는 선정 위치에서 재생성한 **360° 파노라마 전체**이다.

본 task의 핵심 출력은 원본 프레임 목록이 아니라 **중앙 경로, 선정 위치와 카메라 자세, 투어 연결, 이미지 생성 요청, 커버리지 보고서**이다. 실제 가상 시점 이미지 생성은 외부 렌더러와 연결한다. 전체 중앙 경로 영상을 먼저 생성할 필요는 없다.

## 상태와 적용 범위

- **구현 상태:** 단일 회차용 CPU 프로토타입을 구현했다. 바닥/장애물 격자, medial-axis 경로 그래프, 360° 후보, 삼각형 가림 검사, 의미 중요도, 정보 증가량 선정과 역제거, 이동용 위치, 생성 실패 후보 제외, 전용 출력 검사를 포함한다. 실제 팀 데이터 성능이나 이미지 품질을 입증한 상태는 아니다.
- **현재 코드:** 새 기능은 `planner.create_plan()` / `planner.generate()`와 task 전용 CLI로 실행한다. 기존 `pipeline.run()`은 여전히 `BackendNotImplemented`를 반환하여 원본 프레임 계약과 섞이지 않게 유지했다.
- **기존 실행 계약:** 현재 registry와 공통 스키마의 Task5b는 여전히 `objects -> keyframes`, 즉 원본 프레임 선정을 정의한다. 새 출력은 아직 공통 CLI에 등록되지 않았다.
- **변경 경계:** 구현은 Task5b 디렉터리와 전용 테스트에 한정한다. 다른 task의 코드, 공통 schema·registry·validator·fixture는 변경하지 않는다. 공통 통합은 [협업 규칙](../../../../CONTRIBUTING.md)에 따라 별도로 합의한다.
- **1차 대상:** 단일 층의 비교적 평평한 실내 공간이다. 계단·다층 공간은 미지원으로 명시한다.

이 문서의 `필수`는 구현 시 충족해야 하는 조건이다. `제안`은 아직 생산·소비 task 간 확정되지 않은 인터페이스 또는 실험용 초기값이다. 이어서 작업할 때는 짧은 [개선 작업 메모](DEVELOPMENT_NOTES.md)의 결정사항과 완료 상태를 먼저 확인한다.

## 프로토타입 실행

저장소 루트에서 task 전용 가상환경을 사용한다. 검증 환경은 Python 3.13, CPU이며 GPU·모델 가중치를 사용하지 않는다. 실제 검증 버전은 [requirements-tested.txt](requirements-tested.txt)에 기록했다. 범위 의존성은 [requirements.txt](requirements.txt)에 있으며 다른 조합은 다시 검증해야 한다.

```bash
python -m venv .venv-task5b
source .venv-task5b/bin/activate
python -m pip install -e .
python -m pip install -r src/spacewatch3d/tasks/task_5b_keyframes/requirements-tested.txt
python -m spacewatch3d.tasks.task_5b_keyframes plan \
  --scene src/spacewatch3d/tasks/task_5b_keyframes/examples/room.scene.json \
  --output outputs/task5b-demo-001
python -m spacewatch3d.tasks.task_5b_keyframes validate \
  outputs/task5b-demo-001/tour_plan.json outputs/task5b-demo-001/render_requests.json
python -m unittest discover -s tests -v
```

`--output`은 존재하지 않는 새 디렉터리여야 한다. 기존 결과와 상위 입력을 덮어쓰지 않는다. 종료 코드는 기하학적 계획 목표 달성 `0`, 입력/실행 오류 `2`, 유효 입력이지만 목표 미달 `3`이다. `3`일 때도 실패 사유를 포함한 계획을 저장한다. 오류 `2`는 표준 오류에 `planning_status=invalid_input`을 기록한다.

선택 인자: `--config planning.config.example.json`, `--importance-cache 중요도.json`, `--feedback 피드백.json`. 모두 task 전용 CLI의 인자이며 공통 `spacewatch3d run`의 인자가 아니다.

합성 예시는 문으로 연결된 두 공간·벽·객체 3개를 사용한다. `box`는 직접 정의한 합성 메시이며 실제 재구성 성능의 증거가 아니다. 기본값은 격자 0.15m·초기 후보 간격 0.6m·보강 간격 0.15m이며, 적응형 후보와 이동용 장수를 고려한 선정을 기본 사용한다. 나머지는 `models.Config`를 따른다. 아래 설계의 초기 제안값과 구별한다.

`adaptive_candidates=false`, `route_aware_selection=false`로 기존 고정 후보·정보 증가량 방식을 비교할 수 있다. 두 옵션은 독립적이다. 작은 예시 맵의 밀도/선정 비교는 `examples/compare_planning.py --scene 장면.json --output 새디렉터리`로 실행한다. 같은 표면 표본과 전체 경로 노드의 공통 관측 가능 분모를 사용하며, 같은 이미지 장수 비교나 실제 이미지 품질 평가를 대신하지 않는다.

### 실제 입력 형식

입력은 [examples/room.scene.json](examples/room.scene.json)의 `spacewatch3d.planning_scene / 0.1.0` 형식이다.

- `coordinates`: 계획 좌표계의 `frame_id`, `units="m"`, `up_axis="+Z"`, `gravity_aligned=true`. 알고리즘이 스케일이나 중력 방향을 추측하지 않는다.
- `floor`: `confirmed=true`, 하나의 수평 높이 `height_m`, XY 외곽 `polygon`과 선택적 `holes`. 구멍은 미확인/통행 금지 영역이다. 불규칙 바닥과 다층은 아직 지원하지 않는다.
- `camera.height_above_floor_m`: 선택적 실제 촬영/관람 높이. 우선순위는 명시적 `config.camera_height_m` → 이 필드 → 미검증 기본 1.5m와 경고다. 설정의 null은 장면 값을 사용한다는 뜻이다. 해석된 높이와 출처는 `provenance.camera_height`에 기록하고, 후보 Z는 바닥 높이와 합산한다. 몸 높이 검사 범위 밖이면 입력 오류이며 `body_max_m` 등을 실제 조건에 맞춰 함께 설정해야 한다.
- `geometry_complete=true`: 바닥·벽·가림/장애물 구조를 전달했음을 입력 생산자가 명시적으로 확인한다. 코드가 실제 공간의 누락 여부를 자동으로 보증하지는 않는다.
- `objects`: `instance_id`, `class_name` 또는 null, `mesh_uri` 또는 `triangles` 또는 `box`, 선택적 `world_from_object`, `importance`, `must_see`, `description`.
- `structures`: 같은 기하 형식. `solid=false`인 장면 구조 메시에는 몸 높이 구간의 삼각형만 투영한다. 기둥 등 **부피가 찬 개별 장애물**은 `solid=true`로 표시한다. `box`는 기본적으로 solid이다. 방 전체 메시를 solid로 표시하면 내부까지 차단할 수 있으므로 금지한다.
- `regions`: 겹치지 않는 `region_id`·XY `polygon`. 없거나 미지정인 표면은 고정 타일로 계산한다.
- `start_xy_m`: 선택적 시작 방향 선호 위치. 선정 위치 중 가까운 곳부터 방문하며 해당 좌표에 강제로 새 촬영 위치를 만들지 않는다.

메시 파일은 로컬 OBJ/GLB를 지원한다. 경로는 **그 필드를 가진 manifest의 디렉터리 기준**이다. GLB 내부 노드 변환을 먼저 bake하고 `world_from_object`를 한 번 적용한다. 텍스처는 선정 계산에 사용하지 않는다.

기존 Task4 결과를 쓸 때는 `objects` 대신 `objects_manifest`를 지정하면 Task3의 객체명을 ID로 결합하고 Task2까지 scene/capture/synthetic 연속성을 검사한다. 다른 task의 내부 코드는 import하지 않는다. 상위 세계 좌표계가 이미 계획 좌표계와 같은 미터·+Z 좌표이면 그대로 사용한다. 다르면 실측으로 결정한 4x4 `planning_from_world`를 명시한다. 이 행렬은 객체 변환보다 나중에 적용되며, 반사·비균일 스케일은 거부한다. 알려진 `scale_to_meters`와 행렬 스케일이 일치해야 한다. 스케일이 미확인이면 `source_scale_confirmed=true`와 **실제로 보정한** 행렬이 필요하며 임의로 true를 지정하면 안 된다. `floor`·`structures`·`regions`는 이미 계획 좌표계여야 한다.

벽·천장은 공간 커버리지로만 평가한다. `objects`로 들어와도 이름이 `wall`, `walls`, `ceiling`, `ceilings`, `벽`, `천장`인 항목은 구조물로 처리하고 의미 점수와 LLM 요청에서 제외한다. 해당 ID는 `provenance.coverage_only_objects`에 남긴다. 다른 이름을 자동 추론하지 않으므로 구조물은 명시적으로 `structures`에 전달하는 것이 권장된다. 구조물의 `must_see`나 중요도 override를 조용히 무시하지 않고 오류로 보고한다.

`object_overrides`는 벽·천장을 제외한 의미 객체의 알려진 `instance_id`를 키로 `importance`, `must_see`, `description`만 덮어쓸 수 있다. 상위 manifest를 편집하지 않고 필수 객체나 수동 중요도를 설정하는 용도다.

### LLM 중요도 연결

기본 실행은 외부 LLM을 호출하지 않는다. 우선순위는 **수동 값 → 캐시 → 주입한 제공자 → 기본 0.5**이다. 유효하지 않은 수동 값은 오류, LLM/캐시의 잘못된 점수나 호출 실패는 경고와 기본값으로 처리한다. 모든 객체의 최종 값과 근거를 `importance_scores.json`에 저장한다.

```python
from spacewatch3d.tasks.task_5b_keyframes.importance import LLMImportanceProvider
from spacewatch3d.tasks.task_5b_keyframes.planner import generate

# model_json_call(prompt, objects) -> {instance_id: {"score": 0..1, "reason": str}}
# 모델 SDK, 인증, 타임아웃, 재시도는 이 콜백 구현자가 관리한다.
provider = LLMImportanceProvider(model_json_call)
generate("scene.json", "outputs/task5b-run-001", importance_provider=provider)
```

`importance.py`의 고정 프롬프트는 공간 대표성·관리/안전 중요성·상태 확인 필요성과 일시성을 평가하고, 실제 상태/변화를 객체명에서 지어내지 않도록 한다. CLI 캐시는 `{"scene_id":"...","capture_id":"...","scores":{"객체ID":{"score":0.8,"reason":"..."}}}`이며 생성된 `importance_scores.json`을 재사용할 수 있다. 회차를 바꾼 캐시를 조용히 적용하지 않는다.

### 렌더링 실패 피드백

`--feedback`의 입력은 `spacewatch3d.virtual_tour_render_feedback / 0.1.0`, `scene_id`, `capture_id`, `reference_frame_id`, `geometry_sha256`, `results`를 가진다. `geometry_sha256`은 생성 요청의 값을 보존한다. 각 결과는 `viewpoint_id`, 요청과 같은 `reference_from_camera`, `render_status="failed"|"succeeded"`, 선택적 `quality_accepted`, `failure_reasons`를 가진다. 성공은 실제 존재하는 `image_ref`가 필수다.

실패 또는 품질 거부 위치는 후보에서 제외하고 재선정한다. 제외 전후 후보 수·관측 가능 면적을 보고한다. 이동용 위치도 **점수를 계산한 후보 집합** 안에서 고른다. 경로 위에 렌더링 가능한 중간 후보가 없어 최대 간격을 지킬 수 없으면 `transition_gap`으로 미완료 처리하며 벽을 가로지르지 않는다. 원본 후보 간격을 더 촘촘하게 설정하는 것이 해결 방법 중 하나다.

외부 성공 이미지가 있으면 `quality_status=partial`로 기록할 수 있지만, 파일 존재와 외부 응답은 독립적인 이미지 품질 검증이 아니다. 프로토타입은 항상 `publication_ready=false`이며 게시용 `tour_manifest.json`은 아직 출력하지 않는다. 이미 있는 이미지의 요청은 `render_required=false`로 표시한다.

### 아직 구현하지 않은 부분

- 미보정 점군에서 스케일·중력·바닥·구조 메시를 자동 복원하는 기능.
- 반복 회차의 기준 경로 재사용, 경로 버전 갱신과 Task5a 정합 자동 연결. 현재 입력에 `reference_route`/`registration`을 주면 명시적으로 거부한다.
- 원본 카메라 관측 범위에 의한 합성 가능성 제한, 실제 360° 이미지 합성, 상태 가독성/합성 오류 검사와 게시용 exporter.
- 실제 대형 메시의 속도·메모리 평가, 실제 공간의 비교 실험과 최종 가중치 튜닝.

자원 상한을 넘으면 데이터를 조용히 버리지 않고 오류를 낸다. 광선 검사는 CPU BVH와 양면 삼각형 교차를 사용한다. 공간 지표는 면적 가중 표면 샘플의 근사치이므로 작은 물체·얇은 틈은 해상도와 표본 수에 민감하다. 가시성 변화에 따른 후보 보강은 한 번 수행하며, 양 끝 후보에서 모두 안 보이는 좁은 틈의 정보를 반드시 발견한다고 보장하지 않는다. 고정 시드로 같은 입력의 재현성을 확보하지만 서로 다른 삼각분할에서 같은 샘플을 보장하지는 않는다. 경로도 격자 해상도에 따른 근사이며 실제 보행 안전을 보증하는 시스템이 아니다. 정보 보존을 검사하는 가지 정리와 경로 평활화는 아직 구현하지 않았다.

참고 구현 문서: [scikit-image medial_axis](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis), [SciPy distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html), [Trimesh Scene.to_mesh](https://trimesh.org/trimesh.scene.scene.html#trimesh.scene.scene.Scene.to_mesh).

## 담당 범위

### Task5b에서 수행

1. 상위 manifest를 읽고 객체명·메시·카메라·좌표 정보를 결합한다.
2. 확인된 바닥과 전체 장애물로 이동 가능 영역을 계산한다.
3. 그 영역의 중심선으로 중앙 경로 **그래프**를 생성한다.
4. 경로 위에 후보 위치를 배치하고 공간 및 객체 가시성을 계산한다.
5. LLM 또는 캐시·수동값으로 객체 중요도를 평가한다.
6. 정보 증가량에 따라 키프레임 위치를 선정하고 불필요한 위치를 제거한다.
7. 벽을 통과하지 않는 연결 경로와 투어 방문 순서를 생성한다.
8. 선정 위치의 이미지 생성 요청을 출력하고, 외부 품질 피드백으로 필요 시 재선정한다.

### 외부 결과를 사용

- 3D 재구성, 객체 분할·의미 분류, 객체 메시 export는 Task1b·2·3·4의 결과를 사용한다.
- 반복 촬영 간 공통 좌표계 정합과 객체 대응은 Task5a 또는 합의된 정합 모듈의 결과를 사용한다.
- 가상 위치의 사진 품질 360° 이미지 생성은 외부 렌더러가 수행한다.
- 실제 이미지의 서비스 URL 발급, 저장·배포와 뷰어는 서비스 모듈과 연결한다.

## 다른 task에서 받을 수 있는 입력 확인

팀에서는 **포인트클라우드를 객체별 메시로 변환해 공간 맵을 구성하는 것까지 확인했다**. 실험은 계속 진행 중이며, 이 기능 확인과 공통 manifest를 통한 Task5b 연결 검증은 구분한다. 아래의 미연결 표시는 메시·맵 생성 자체를 하지 못했다는 뜻이 아니다.

저장소 대조 기준은 2026-10-10의 `main` 커밋 `b90b0a5`이다. **공통 규약에 필드가 있음**, **저장소에 실험 기록이 있음**, **같은 회차의 실제 파일을 Task5b에서 읽을 수 있음**을 구분한다. 마지막 항목은 실제 입력 묶음을 전달받아 확인한다.

| 필요한 정보 | 생산 위치와 근거 | 확인 결과와 Task5b 처리 |
| --- | --- | --- |
| 객체별 GLB/OBJ와 위치 변환 | [Task4](../task_4_object_export/README.md), `objects[].mesh_uri`, `world_from_object` | 팀에서 객체별 메시와 맵 구성을 확인함. 규약과 ScanNet 객체 3개의 실험 기록이 있으며, Task5b가 받을 실제 회차의 manifest·좌표 연결은 별도 확인 |
| 객체명과 분류 신뢰도 | [Task3](../task_3_instance_segmentation/README.md), `instances[].class_name`, `class_confidence` | 규약에 있음. 객체명은 null일 수 있으므로 중요도 기본값 필요. `objects`의 `instance_id`로 `instances`와 결합 |
| 전체 XYZRGB 점군 | [Task2](../task_2_reconstruction/README.md), `reconstruction.pointcloud_uri` | 규약에 있음. 점군을 완성된 장면 메시로 간주해서는 안 됨 |
| 360° 영상의 점군·카메라 pose | [Task1b 실험](../task_1b_360_input/README.md#panovggt-복원-결과-2026-10-07), [검증 기록](../task_1b_360_input/assets/panovggt_r0010003_20261007/verification.json) | PanoVGGT의 점군 2,376,696개·pose 69개 기록 있음. 원본 파일은 별도 실험 환경에 있으며 공통 manifest 어댑터는 미연결 |
| 원본 프레임·시각·360° 영상 | `objects -> instances -> reconstruction -> frames -> video` | 규약상 추적 가능. perspective view와 파노라마 전체를 구분하고 실제 원본 접근 경로도 전달받아야 함 |
| 바닥·벽·미분류 구조물과 이동 가능 영역의 근거 | 전체 재구성 및 구조 정보 | **현재 객체 export 규약만으로 완전하게 제공된다고 보장할 수 없음. 추가 전달 계약 필요** |
| 실제 스케일과 신뢰할 수 있는 위쪽 방향 | `reconstruction.coordinates` 및 바닥 정의 | 필드는 있으나 360° 실험은 절대 스케일·중력 방향이 미보정. 미터 단위 카메라 높이와 여유 거리를 바로 적용할 수 없음 |
| 회차 간 정합 | [Task5a](../task_5a_change_detection/README.md), `registration.baseline_from_current` | 규약에 있음. 성공 상태와 변환 방향을 확인한 후 기준 경로 재사용. 실패·공통 관측 부족은 별도 처리 |
| 가상 시점 이미지와 품질 피드백 | 외부 렌더러 | **현재 공통 task 규약에서 제공하는 결과가 아님. 요청·응답 인터페이스 합의 필요** |

특히 현재 360° 실험은 촬영자 제거를 위해 하단 35%를 출력·정합에서 제외하여 바닥 일부가 빠져 있다. 표시용 `(X, Z, -Y)` 축 변환도 중력 정렬을 뜻하지 않는다. 객체 3개의 ScanNet 실험과 실제 360° 영상 복원 실험은 다른 데이터이므로 같은 장면의 완성된 입력으로 합쳐서는 안 된다.

Task4의 GLB 실험에는 노드 회전이 있고 OBJ는 원래 세계 좌표를 유지한다. 로딩 시 내부 노드 변환과 `world_from_object`를 조합한 최종 좌표를 확인하고, 표시용 회전을 중복 적용하지 않는다.

### 상위 task와 합의할 추가 입력

1. **공간 기하:** 바닥·벽·기둥·미분류 구조물을 포함한 가림/장애물 기하와 확인된 바닥 영역을 전달한다. 객체 메시들에 실제로 포함되면 재사용할 수 있으나, 없는 구조를 있다고 가정하지 않는다. 점군에서 구조 메시를 만드는 추가 처리가 필요하면 담당과 형식을 별도로 정한다.
2. **길이와 방향:** 실측 기준으로 보정된 스케일과 바닥/위쪽 방향을 전달한다. 미확인 값을 임의로 미터·수평으로 확정하지 않는다.
3. **같은 촬영 회차의 연결:** 점군, 객체, 라벨, 카메라, 원본 영상의 `scene_id`·`capture_id`·좌표계를 일치시킨 manifest 묶음과 실제 파일 접근 방법을 전달한다.
4. **반복 촬영:** 현재 좌표에서 기준 좌표로의 성공한 정합 결과를 전달한다. 최초 한 회차 계획은 회차 간 정합 없이 시작할 수 있다.
5. **렌더러:** 아래 가상 뷰포인트 요청과 실제 이미지·실패/품질 피드백 형식을 연결한다. 객체 메시의 텍스처만으로 미관측 영역까지 사진 품질로 재생성할 수 있다고 가정하지 않는다.

확인된 객체별 메시 맵을 본 task의 입력으로 사용한다. 실제 입력을 받을 때 위 조건을 점검하고, 부족한 필드나 구조 정보만 별도로 요청한다. 전체 메시 생성 실험의 종료를 기다려야만 개발을 시작할 수 있다는 뜻은 아니다. 합성 실내 테스트도 병행한다.

## 입력 계약

### 현재 CLI에서 지원하는 입력

| 이름 | artifact kind | 형식 예제 |
| --- | --- | --- |
| `objects` | `objects` | [before/objects.json](../../../../examples/fixtures/before/objects.json) |

상위 참조는 해당 manifest 디렉터리 기준 상대 경로로 해석한다. 객체명은 `instances.json`에서 결합하고 좌표·카메라는 `reconstruction.json`에서 읽는다. 다른 task의 내부 Python 모듈을 직접 import하지 않는다.

### 새 계획 코어가 소비할 입력 제안

| 입력 | 필수 조건과 의미 |
| --- | --- |
| 공간·회차 ID와 기준 좌표계 | 입력들의 연결과 pose의 소속 좌표계를 명시 |
| 객체 메시와 의미 정보 | ID, 메시, 로컬 변환, 객체명 또는 null, 선택적 설명·수동 중요도·필수 객체 표시 |
| 전체 가림/장애물 기하 | 객체 외 구조를 포함하고 중복 면 참조를 처리 |
| 바닥 정의 | 확인된 바닥 메시 또는 명시적인 평면과 유효 영역 |
| 스케일과 축 | 미터화 가능한 스케일 및 신뢰할 수 있는 위쪽 방향 |
| 출발점과 방 구분 | 선택 입력. 없으면 결정론적 출발점과 공간 타일을 사용 |
| 원본 관측 근거 | 선택 입력. 없으면 기하학적 잠정 선정만 가능 |
| 기준 경로와 정합 | 반복 촬영 시 필요. 최초 계획에는 선택 입력 |
| 중요도 캐시와 렌더러 피드백 | 선택 입력. 외부 호출 없이 단위 테스트 가능하게 구성 |

이 표의 추가 입력 이름과 파일 형식은 아직 공통 CLI의 인자가 아니다. `--input scene=...` 같은 인자를 이미 지원하는 것으로 문서화하거나 구현해서는 안 된다.

### 좌표와 오류 처리

- 행렬은 행 우선 JSON 배열, 계산은 열벡터에 왼쪽 곱을 사용한다.
- 객체 좌표에 `world_from_object`를 먼저 적용하고 정합이 필요한 경우 현재→기준 변환을 적용한다. Sim3 내부 스케일을 중복 적용하지 않는다.
- 카메라 로컬 축은 x 오른쪽, y 아래쪽, z 전방이다. 출력 pose는 카메라→명시된 기준 좌표계 변환이다.
- `class_name=null`은 유효하며, 없는 이름·상태·지속 객체 ID를 만들지 않는다.
- `instance_id`는 회차 내 ID이다. 회차 간 동일 객체는 외부 대응 결과로 확인한다.
- 바닥 누락, 스케일 미확인, 유효하지 않은 변환, 미지원 다층 입력은 `invalid_input`으로 보고한다.
- 유효 후보 없음, 관측 불가능한 필수 객체, 연결/커버리지 미달은 `incomplete`로 보고한다.

## 처리 알고리즘

### 중앙 경로와 후보

1. 바닥을 격자로 만들고 설정된 높이 범위의 구조·객체를 장애물로 투영한다. 수직 벽의 선분 투영도 누락하지 않는다.
2. 장애물 주변 여유 거리를 적용하고 **확인된 바닥 중 장애물·미확인 영역이 아닌 곳만** 이동 가능으로 둔다.
3. 거리 변환과 medial axis로 중심선을 추출하고 분기·끝점·연결 폴리라인을 가진 그래프로 만든다.
4. 벽을 관통하는 직선 연결과 모서리 대각선 통과를 금지한다. 서로 떨어진 공간은 별도 성분으로 남긴다.
5. 초기 일정 간격 후보에 분기·끝점과 좁은 구간의 후보를 추가한다. 인접 후보의 면적 가중 가시 표면 차이 또는 객체 품질 차이가 설정 임계값 이상인 경로 구간을 한 번 더 촘촘하게 보강한다. 거리 비교는 벽을 가로지르는 직선 거리가 아니라 그래프 거리다.
6. 전체 360°이므로 같은 위치의 yaw만 다른 후보를 중복 생성하지 않는다. 초기 표시 방향만 별도로 정한다.

보강 간격은 `max(grid_m, candidate_min_spacing_m)`이며 분기·끝점은 보존한다. `candidate_narrow_clearance_m`는 여유 거리 적용 후 격자의 거리 변환 값 기준이고, `candidate_visibility_change_threshold`는 가시 표면의 면적 가중 대칭 차이/합집합 또는 객체 품질 차이의 임계값이다. 모든 추가 후보의 점수를 계산한 뒤 관측 가능 분모를 고정한다. 원인과 단계별 개수는 `candidate_report`에 기록한다.

물체 중심점들의 평균은 중앙 경로가 아니다. 경로 평활화 후에도 충돌을 재검사한다. 후보 간격은 탐색 해상도이며 선정 위치 사이의 강제 최소 거리와 구분한다.

첫 기준 경로의 ID·버전을 보관하고 이후 회차는 정합 후 재사용한다. 현재 장애물과 충돌하는 위치는 제외하거나 변경 구간을 보고한다. 우회로를 만들면 경로 버전을 갱신한다.

### 가시성과 객체 중요도

- 면적 기준으로 공간 표면 패치를 샘플링하고, 후보→패치 광선의 첫 충돌로 가림을 검사한다. 메시의 삼각형 개수 자체를 정보량으로 쓰지 않는다.
- 모든 유효 후보 중 하나 이상에서 보이는 패치를 관측 가능 대상으로 고정한다. 전체 입력 면적과 관측 불가능 면적도 별도로 보고한다.
- 방 정보가 있으면 방별, 없으면 고정 공간 타일별 커버리지를 계산한다. 타일을 실제 방으로 표현하지 않는다.
- 객체 관측 품질 `q(i,o)`는 가시 표면 비율과 투영 크기/입체각을 함께 사용한다. 모든 후보가 나쁜 객체를 최선 후보라는 이유만으로 1로 정규화하지 않는다.
- LLM은 공간 대표성·관리 중요성·상태 확인 필요성·일시성과 근거를 평가한다. 정적=중요, 동적=불필요라고 단정하지 않는다.
- 객체명만으로 실제 상태·변화 여부·생성 이미지 품질을 판정하지 않는다. 응답 범위를 검증하고 캐시·수동값·실패 시 기본값을 지원한다.

### 정보 증가량 기반 선정

선정 집합 S의 효용을 다음과 같이 정의한다. 한 번의 선정 실행 동안 중요도와 후보 점수는 고정한다.

```text
C_space(S) = 공간 구역별 면적 가중 커버리지의 균형 평균
C_object(S) = sum_o[w(o) * max_i_in_S q(i,o)] / sum_o w(o)
F(S) = alpha * C_space(S) + beta * C_object(S)
gain(candidate | S) = F(S + candidate) - F(S)
```

빈 선정 집합의 커버리지와 객체 품질은 0이다. 의미 객체 중요도 합이 0이면 객체 항을 적용하지 않고 공간 항으로 재정규화한다.

1. 기본 모드에서는 후보를 추가했을 때의 실제 그래프 연결과 이동용 위치까지 계산한다. 추가 정보량을 추가 고유 이미지 수로 나눈 값이 큰 후보를 선택한다. 추가 장수의 분모는 최소 1이며, 양의 증가량 후보 중 연결 가능한 대안을 우선한다. 기존 모드는 정보 증가량만 사용한다.
2. 이미 본 표면과 이미 잘 보이는 객체를 반영해 나머지 증가량을 갱신한다.
3. 공간 전체·구역별 목표, 충분히 관측된 객체 중요도 비율, 필수 객체 조건까지 확인한다. 기본 모드에서는 실제 제공할 이동용 이미지의 정보도 포함하며 `content_only` 결과는 별도로 기록한다.
4. 모든 적용 목표를 만족하면 멈춘다. 후보 소진 또는 수치적으로 증가량이 없어도 멈추되 목표 미달이면 `incomplete`이다.
5. 목표를 보존하면서 불필요한 위치를 역순으로 제거한다. 기본 모드는 실제 이미지 장수가 늘지 않아야 제거하고, 최종 장수와 경로 길이를 줄이는 단일 위치 교환도 최대 한 번 적용한다. 목표나 연결 조건을 완화하지 않는다. 동일 점수는 고정 ID 순서로 처리한다.

`T(S)`를 선정 위치 S와 실제 연결에 필요한 이동용 위치의 합집합이라 하면 기본 순위는 `(F(T(S+i)) - F(T(S))) / max(1, |T(S+i)| - |T(S)|)`다. 이것은 전역 최적 경로/최소 장수 알고리즘이 아닌 결정적 휴리스틱이다.

중복은 추가 정보가 거의 없는 것으로 처리한다. 독립 점수 상위 N개 또는 객체별 top-k만으로 전체 투어 선정을 대체하지 않는다. 관측 가능한 객체 기준과 전체 객체 기준 달성률을 모두 기록하며 관측 불가능한 필수 객체는 실패로 남긴다.

고정 바이트 예산은 없다. 목표를 만족하는 적은 이미지 수를 지향하지만 전역 최소 장수를 보장하지 않는다. 초기 실험값은 공간/객체 비중 0.6/0.4, 균형 공간 목표 0.9, 구역별 최소 커버리지 0.6, 관측 가능 객체 중요도 목표 0.9로 제안한다. 격자 0.1m, 여유 거리 0.3m, 카메라 높이 1.5m, 후보 간격 0.5m도 미터 스케일이 확보된 경우의 **미검증 시작값**이다.

### 연결과 이미지 검증

- 선정 위치는 중앙 경로 그래프의 경로 거리로 연결한다. 분기에서는 재방문을 허용하지만 동일 이미지를 중복 저장하지 않는다.
- 설정된 최대 이동 간격을 맞추기 위한 위치는 `transition` 역할로 추가한다. 정보 때문에 고른 `content`와 구분하며 둘 다 최종 이미지 수에 포함한다.
- 선정된 위치만 렌더러에 요청한다. 생성 실패 후보를 제외하고 점수·대상 집합·연결을 다시 계산한다. 제외 전후 분모와 사유를 남긴다.
- 실제 이미지의 빈 영역·합성 오류·주요 객체 확인 가능성·필요 시 상태 가독성을 검사한다. 기하학적 가시성만으로 이미지 품질을 검증했다고 표시하지 않는다.

## 출력 계약 제안

다음은 **새 계획 코어의 출력 계약**이다. `tour_plan.json`과 `render_requests.json`은 프로토타입에서 실제 생성하며 `importance_scores.json`도 재사용용 캐시로 저장한다. `render_feedback.json`은 외부 입력, `tour_manifest.json`은 아직 미구현인 후속 목표다. 이 형식은 공통 `artifact.schema.json`에 등록된 kind가 아니다. 공통 통합 합의 전까지 task 전용 CLI와 검증기를 사용한다.

| 파일 | 생산과 소비 | 내용 |
| --- | --- | --- |
| `tour_plan.json` | Task5b → 서버·검증·서비스 | 필수 주 산출물. 중앙 경로, 선정 위치와 pose, 방문 순서·연결, 객체 품질, 커버리지·실패·출처 |
| `render_requests.json` | Task5b → 외부 렌더러 | 필수 생성 요청. 같은 뷰포인트 ID·pose·회차·해상도·투영 규약 사용 |
| `render_feedback.json` | 외부 렌더러/검사 → Task5b | 외부 입력. 실제 이미지 참조, 생성 성공/실패, 검사 근거. Task5b가 생성 성공을 지어내지 않음 |
| `tour_manifest.json` | Task5b의 선택적 후속 exporter → 서비스·웹 | 검증 후 생성하는 가벼운 게시용 인덱스. 위치, 초기 방향, 이웃과 실제 이미지 참조만 전달 |

360° 이미지 파일은 외부 렌더러가 생성하는 별도 자산이다. Task5b의 위치 선정 결과에 사진 파일이 이미 존재한다고 가정하지 않는다. 웹 인덱스에는 전체 메시·점군·후보별 가시성 행렬을 싣지 않는다. 서비스 URL 매핑은 서비스 모듈에서 수행한다.

### 주 산출물의 필드

| 필드 | 자료형과 의미 |
| --- | --- |
| `schema_name`, `schema_version` | string. 제안 식별자 `spacewatch3d.virtual_tour_plan`, 버전 `0.1.0` |
| `scene_id`, `capture_id` | string. 공간·촬영 회차 |
| `synthetic` | boolean. 합성/시험 입력 여부이며 가상 시점 생성 여부와 구분 |
| `reference_frame_id`, `units`, `up_axis` | string. pose·경로·거리의 기준 좌표계와 단위 |
| `route_id`, `route_version` | string, integer. 기준 경로 식별과 수정 버전 |
| `planning_status` | `complete`, `incomplete`, `invalid_input` |
| `quality_status` | `unverified`, `partial`, `validated`, `failed` |
| `publication_ready` | boolean. 기하학적 계획과 실제 미디어의 게시 준비 조건 |
| `route_graph` | object. 노드와 성분 ID, 연결과 실제 경로 폴리라인 |
| `candidate_report` | object. 초기/기하/가시성 보강 후보 개수, 후보 ID와 경로 노드, 보강 이유 |
| `viewpoints` | array. 선정 위치, pose, 역할, 선정 이력, 객체 품질, 이미지 상태 |
| `tour_visits`, `tour_links` | array. 재방문 가능한 방문 순서와 그래프 경로 연결 참조 |
| `coverage_report` | object. 구역 균형 점수, 면적 기준 비율, 전체/관측 가능 객체 달성률, 목표 미달 항목 |
| `unobservable_items` | array. 관측 불가능 구역·객체와 사유 |
| `failures`, `warnings` | array. 입력·계획·생성·검사 단계의 실패와 제한 |
| `provenance` | object. 입력 manifest 참조, 실행 설정·시드·코드와 중요도 캐시 버전 |

각 `viewpoints` 항목의 필수 필드는 다음과 같다.

| 필드 | 의미 |
| --- | --- |
| `viewpoint_id`, `candidate_id`, `component_id` | 가상 위치, 원래 후보, 경로 성분 식별자 |
| `reference_from_camera` | 카메라→기준 좌표계의 4x4 행렬 |
| `roles` | `content` 또는 `transition`을 포함하는 배열 |
| `score_basis` | 기하학 또는 확인된 외부 품질을 사용했는지에 대한 기록 |
| `selection_step`, `marginal_gain_at_selection`, `reason` | 탐욕 선정 당시의 순서·효용 증가량·이유. 단일 교환으로 들어온 위치는 순서와 증가량이 null이며 이유로 구별한다 |
| `selection_method`, `marginal_image_count_at_selection` | 순수 정보량/추가 장수 고려/단일 교환 방식과 당시 추가 고유 이미지 수 |
| `object_observations` | 회차 내 객체 ID, 중요도와 출처, 기하학적 품질, 실제 검사 결과 또는 null |
| `render_status`, `image_ref` | `pending`/`succeeded`/`failed`, 실제 상대 경로 또는 null |

선정 후 이미지 생성 전에는 `quality_status=unverified`, `publication_ready=false`, `image_ref=null`이다. `planning_status=complete`는 기하학적 목표 달성일 뿐 게시 완료가 아니다.

`publication_ready=true`는 계획 목표와 경로 조건을 충족하고, 모든 최종 이미지가 실제로 존재하며, 필수 객체를 포함한 명시적 이미지 검사 기준을 통과했을 때만 허용한다. `validated`도 기록된 검사 기준 통과를 뜻하며 완벽한 재생성의 보증은 아니다.

### 렌더러 요청과 응답

요청은 `schema_name=spacewatch3d.virtual_tour_render_requests`, `schema_version=0.1.0`과 `requests` 배열을 가진다. 각 요청은 다음 필드를 포함한다.

- `viewpoint_id`, `capture_id`, `reference_frame_id`
- `reference_from_camera`
- `projection=equirectangular`, `width`, `height`
- `media_origin=novel_view`
- `required_object_ids`

`width:height=2:1`을 사용한다. 4096×2048은 시작 해상도 제안이며 상태 가독성이나 용량의 확정값이 아니다. 원본 근거 프레임 ID는 별도 출처 필드로만 사용한다.

피드백은 `viewpoint_id`, `render_status`, `image_ref`, `failure_reasons`, `quality_checks`를 반환한다. 이미지 경로가 실제로 존재하는지 확인하고 검사 방법·버전을 기록한다. 전역 품질 점수만으로 개별 객체의 상태 가독성이 검증됐다고 간주하지 않는다.

### 선정 위치와 생성 요청 예시

아래는 좌표 규약을 보여주는 **합성 형식 예시**이며 전체 manifest나 실제 실행 결과가 아니다. 기준은 미터·Z-up, 카메라는 +Y 방향을 보고 y축 아래쪽이 -Z이다. 실제 생성 파일에 넣기 전 전용 스키마 검증이 필요하다.

```json
{
  "viewpoint_id": "vp_demo_0001",
  "candidate_id": "candidate_demo_0001",
  "component_id": "component_demo_01",
  "reference_from_camera": [
    [1, 0, 0, 1],
    [0, 0, 1, 2],
    [0, -1, 0, 1.5],
    [0, 0, 0, 1]
  ],
  "roles": ["content"],
  "score_basis": "geometry",
  "selection_step": 1,
  "marginal_gain_at_selection": 0.6,
  "reason": "새 공간 표면과 주요 객체의 관측 품질을 확보한 형식 예시",
  "object_observations": [],
  "render_status": "pending",
  "image_ref": null
}
```

```json
{
  "schema_name": "spacewatch3d.virtual_tour_render_requests",
  "schema_version": "0.1.0",
  "requests": [
    {
      "viewpoint_id": "vp_demo_0001",
      "capture_id": "capture_demo_001",
      "reference_frame_id": "world:demo_reference",
      "reference_from_camera": [
        [1, 0, 0, 1],
        [0, 0, 1, 2],
        [0, -1, 0, 1.5],
        [0, 0, 0, 1]
      ],
      "projection": "equirectangular",
      "width": 4096,
      "height": 2048,
      "media_origin": "novel_view",
      "required_object_ids": []
    }
  ]
}
```

### 가벼운 웹 출력의 경계

`tour_manifest.json`은 검증된 최종 위치별 ID·위치·초기 표시 방향·이웃 링크·실제 이미지 참조를 포함한다. 경로 폴리라인 등 이동에 필요한 최소 정보만 추가한다. 서버용 평가 정보와 전체 3D 맵은 제외한다. 실제 파일 참조는 manifest 기준 상대 경로이며 서비스가 이를 배포 URL로 변환한다.

새 출력에 가짜 `frame_id`, `source_frame_index`, `timestamp_s`를 만들지 않는다. 원본 영상 시각과 중앙 경로의 가상 위치는 일대일 관계가 아니다.

## 기존 계약과의 호환

현재 [공통 규약](../../../../docs/contracts.md#대표-키프레임-선별)의 `keyframes`는 객체 ID와 원본 프레임 ID·시각·이미지 참조를 요구한다. [validator](../../contracts.py)는 실제 원본과의 일치까지 검사하므로 가상 이미지를 기존 `image_uri`에 넣을 수 없다.

- [registry.py](../../registry.py)의 Task5b 출력은 아직 `keyframes`이다.
- [config.example.json](config.example.json)의 `top_k_per_object`는 기존 원본 프레임용 예시이며 새 선정 설정이 아니다.
- [before/keyframes.json](../../../../examples/fixtures/before/keyframes.json)은 기존 **합성 형식 예제**이며 중앙 경로나 자동 선정 결과가 아니다.
- 기존 `check-fixtures`와 `validate` 통과는 새 경로·가시성·렌더링의 검증이 아니다.
- 새 출력과 추가 입력은 schema·registry·validator·소비 모듈·fixtures·테스트를 함께 합의한 후 공통 CLI에 연결한다. 이 문서 변경만으로 기존 `run`의 반환 계약을 바꾸지 않는다.

## 검증과 구현 완료 기준

### 프로토타입 확인 결과 (2026-10-10)

- 개선 후 Task5b 전용 합성 테스트 53개와 기존 공통 테스트 19개, 총 72개 통과. task 의존성이 있는 전용 환경에서는 건너뛴 테스트가 없다. 공통 환경만 있을 때 Task5b 기하 테스트는 명시적으로 skip한다.
- 기존 task 7개의 형식 예제 검사 통과. 이것은 새 경로 또는 실제 모델 성능 검증과 별개다.
- 이전 고정 후보/순수 정보량 방식의 합성 두 공간 예시: 후보 35곳 → 정보용 3곳 + 이동용 6곳 = 고유 촬영 위치 9곳. 최종 표면 커버리지는 관측 가능한 표면 기준 99.42%, 입력된 전체 표면 기준 60.88%였다. 개선 결과와 목표의 차이는 [개선 작업 메모](DEVELOPMENT_NOTES.md)에 기록한다. 실제 이미지는 생성하지 않았다.
- 벽/객체의 뒤쪽 등 관측 불가능한 면을 제외한 수치를 전체 공간 커버리지라고 표시하지 않는다. 설정 목표를 만족하는 장수를 줄이면 목표 이상으로 과도하게 확보했던 커버리지는 낮아질 수 있다.
- 파일·ID·pose/생성 요청 일치, 벽 가림, 문 연결, ㄱ자 공간/바닥 구멍, 분리 공간, 중복 효용, 중요도/필수 객체, GLB 변환, 입력 보존, 실패 재선정과 게시 미검증 상태를 시험했다.
- 실제 팀 입력 연결, 외부 LLM 응답 품질, 합성 이미지 품질, 대형 장면 성능과 회차 간 경로 재사용은 미검증/미구현이다. 아래 완료 목록은 전체 기능 목표이므로 모두 완료 처리하지 않는다.

### 현재 실행 가능한 기존 계약 검사

저장소 루트에서 Python 3.10+ 환경에 공통 패키지를 설치한 후 실행한다.

```bash
python -m pip install -e .
python -m spacewatch3d check-fixtures --task 5b
python -m unittest discover -s tests -v
```

아래는 기존 scaffold 진입점이다. 현재는 미구현 오류와 종료 코드 2가 정상이다. 새 `tour_plan.json` 실행 명령으로 사용하지 않는다.

```bash
python -m spacewatch3d run --task 5b --input objects=examples/fixtures/before/objects.json --output outputs/task-5b-trial --config src/spacewatch3d/tasks/task_5b_keyframes/config.example.json
```

### 새 기능의 완료 기준

- [ ] 같은 회차의 실제 입력과 파일 접근 방법, 스케일·축·바닥·구조 기하를 검증한다.
- [ ] 바닥 누락·스케일 미확인·다층 입력을 명시적으로 거부한다.
- [ ] 방, 장애물 방, ㄱ자 복도, 문 연결, 분리 공간 fixture로 경로와 연결 폴리라인의 충돌을 검사한다.
- [ ] 벽 뒤 객체의 가림, 같은 후보의 중복 감소, 중요한 객체의 효용 증가를 시험한다.
- [ ] 미분류·LLM 실패·객체 없음·후보 없음·필수 객체 관측 실패를 시험한다.
- [ ] 면적 가중과 고정 시드·중요도 캐시로 재현성과 삼각형 세분화에 대한 안정성을 검사한다.
- [ ] 공간 구역별/전체 및 전체 객체/관측 가능 객체 지표와 미달성 이유를 출력한다.
- [ ] `tour_plan.json`과 `render_requests.json`의 전용 검증기 및 ID·pose 일치 테스트를 만든다.
- [ ] 이동용 위치의 이미지 수 포함, 재방문 자산 재사용, 실제 경로 연결과 최대 간격을 시험한다.
- [ ] 생성 전 품질 미검증 상태와 생성 실패 후 재선정·미완료 처리를 시험한다.
- [ ] 반복 촬영에서 정합 방향·경로 버전·차단 구간·회차 내 객체 ID를 확인한다.
- [ ] 상위 파일을 수정하지 않고 기존 공통 테스트를 유지한다.
- [ ] 실제 데이터에서 균일 간격/공간만/공간+객체 중요도를 같은 이미지 수 또는 목표로 비교한다.
- [ ] 실제 이미지 수·바이트·커버리지·객체 확인률·생성 실패율·시간·메모리와 한계를 기록한다.

기존 합성 입력은 객체 하나와 8개 점을 가진다. 바닥·벽이 있는 공간으로 해석하지 말고 별도의 실내 fixture를 추가한다. SLAM 내부 키프레임이나 고정 fps 추출도 게시용 투어 선정 완료와 구분한다.

## 작업 위치와 의존성

- 이 디렉터리의 backend·설정·의존성 설명과 `tests/task_5b_keyframes/`에서 구현한다.
- 경로 생성, 가시성, 중요도 제공자, 선정기, 연결과 출력 어댑터를 분리하고 외부 LLM·렌더러 없이 단위 테스트 가능하게 한다.
- 공통 환경은 Python 3.10+와 jsonschema이다. 무거운 3D 의존성은 담당 모듈에서만 로딩하고 검증한 버전만 `requirements.txt`에 기록한다.
- 실제 영상·메시·점군·가중치는 Git에 올리지 않고 접근 방법·출처·revision·필요 시 체크섬을 기록한다.
- 공통 계약 변경은 다른 task 담당자와 합의한 별도 PR로 진행한다. 이 task의 문서만으로 상위 task에 새로운 결과가 이미 존재한다고 가정하지 않는다.
