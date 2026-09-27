# 공통 입출력 규약 0.1.0

이 문서는 **팀 개발 시작용 규약**입니다. 실제 backend 연동에서 필요한 항목을 확인하고 버전과 함께 확장합니다.
기계 검증의 기준은 [`artifact.schema.json`](../src/spacewatch3d/schemas/artifact.schema.json)입니다.
Python 3.10+ 공통 CLI의 검증은 JSON 구조·연결 ID·파일 존재·라벨 CSV 순서를 대상으로 합니다. 모델 정확도·카메라 보정 정확도·메시 품질을 평가하지는 않습니다.

## 1. 모든 task에 공통인 원칙

| 필드/규칙 | 의미 |
|---|---|
| `schema_version` | 현재 `0.1.0`. 필수 필드·의미 변경 시 버전과 모든 관련 fixture를 함께 변경 |
| `kind` | `video`, `frames`, `reconstruction`, `instances`, `objects`, `changes`, `keyframes` |
| `scene_id` | 같은 실제 공간에 공통으로 부여 |
| `capture_id` | 촬영 회차마다 다른 ID. 변화 결과는 현재 회차 ID를 사용 |
| `synthetic` | 합성/시험 데이터면 `true`. 합성 입력에서 만들어진 결과에도 유지 |
| `producer` | backend·버전 식별 이름. 실제 실행은 정확한 revision도 `parameters`에 기록 |
| `parameters` | 사용 설정·원본 모델 revision·전처리·실행 환경 등 JSON 객체 |
| 경로 | **해당 manifest 파일의 디렉터리 기준** 상대 경로. URI/절대 경로는 사용하지 않음 |
| 상위 결과 참조 | `*_manifest`로 원본 manifest를 연결. 상위 결과를 덮어쓰거나 필드 추가하지 않음 |

예: `outputs/capture_001/03_instances/instances.json`의
`reconstruction_manifest: "../02_reconstruction/reconstruction.json"`.
작업 디렉터리가 달라도 같은 파일을 가리켜야 합니다. 결과를 복사할 때 참조한 상위 결과와 미디어도 함께 보존합니다.

공용 manifest 하나를 여러 task가 수정하지 않습니다. task마다 자기 출력 manifest만 씁니다.
한 task가 여러 프로세스로 실행되면 각기 다른 output directory를 사용합니다.
개발 편의를 위한 fixture는 입력과 기대 출력의 **형식 예제**이며 자동 알고리즘 결과가 아닙니다.

## 2. Task 함수 및 실행 계약

```python
from pathlib import Path
from spacewatch3d.task_api import TaskRequest

def run(request: TaskRequest) -> Path:
    # request.inputs: dict[str, Path], registry에서 정의한 이름별 manifest
    # request.output_dir: 이 실행만의 출력 디렉터리
    # request.config: task 설정 객체
    # 입력은 읽기 전용. 출력 디렉터리는 backend가 생성한다.
    # 실제 처리 후 출력 manifest의 경로를 반환한다.
    ...
```

task별 입력 이름·출력 kind·fixture 위치는 `src/spacewatch3d/registry.py`가 정의합니다.
모델 의존성은 task 내부에서만 import합니다. 공통 규약 검증 때문에 다른 task의 GPU 모델까지 설치할 필요가 없어야 합니다.

`spacewatch3d run`은 입력과 파일을 검증하고 backend를 호출한 후 출력도 검증합니다.
미구현 backend는 `BackendNotImplemented`와 종료 코드 2를 반환합니다. 빈 결과나 fixture를 실제 결과인 것처럼 저장하지 않습니다.

## 3. 좌표·크기·ID

- 모든 행렬은 JSON의 행 우선 배열이며, 계산은 **열벡터**에 왼쪽에서 곱합니다.
- `world_from_camera`: `p_world = T_world_camera @ [x_camera, y_camera, z_camera, 1]`.
- 카메라 축은 x 오른쪽·y 아래쪽·z 전방입니다. 월드 축은 오른손 좌표계이며 `up_axis`를 기록합니다. 임의로 바꾸지 않습니다.
- 카메라 투영: `p_camera = inverse(world_from_camera) @ p_world`, `u=fx*x/z+cx`, `v=fy*y/z+cy`.
- reconstruction의 카메라별 `image_uri`, 해상도, intrinsics가 투영의 기준입니다. resize/undistort 후 카메라 정보와 보정 전 이미지 정보를 섞지 않습니다.
- `units=m`이면 실제 좌표를 미터로 변환한 상태이고 `scale_to_meters=1`입니다. 스케일이 미확인되면 `arbitrary`, `scale_to_meters=null`입니다. 알려진 변환 배율만 기록한 상태는 `arbitrary`와 양의 배율을 사용합니다.
- `world_from_object`는 mesh 로컬 → 같은 회차의 reconstruction 좌표입니다. 객체 centroid와 bbox는 reconstruction 좌표로 기록합니다.
- `point_index`는 **해당 재구성 PLY의 vertex 순서**, 0부터 시작합니다. 다운샘플링·정렬·필터링 시 원본으로의 역매핑을 유지합니다.
- `instance_id`는 **회차 내** 고유 ID입니다. 서로 다른 촬영 회차에서 같은 ID라고 같은 물체는 아니며, 다른 ID여도 같은 물체일 수 있습니다.
- 모든 timestamp는 원본 영상 시작 기준 초입니다. `source_frame_index`는 디코딩 순서의 원본 프레임 0-based index입니다. VFR 영상도 `index/fps`로 시각을 임의 재계산하지 않습니다.

## 4. Task별 산출물

| 생성 task | Manifest / kind | 핵심 내용 | 소비 task |
|---|---|---|---|
| 사용자가 준비 | `video.json` / `video` | 영상 경로·projection·해상도·공간/회차 ID | 1a 또는 1b |
| 1a / 1b | `frames.json` / `frames` | 원본 영상 참조·프레임·원본 index/시각·view | 2 |
| 2 | `reconstruction.json` / `reconstruction` | frames 참조·XYZRGB PLY·카메라·깊이·좌표 단위 | 3, 이후 상위 참조 |
| 3 | `instances.json` / `instances` | reconstruction 참조·점 라벨 CSV·객체 클래스/신뢰도 | 4 |
| 4 | `objects.json` / `objects` | instances 참조·mesh·transform·bbox·근거 | 5a / 5b |
| 5a | `changes.json` / `changes` | 이전/현재 객체 참조·정합·객체 대응·변화 | 후속 시각화 |
| 5b | `keyframes.json` / `keyframes` | 객체 참조·원본 프레임 ID·시각·선별 점수 | 후속 시각화 |

### 영상 및 frames

`video.projection`은 `perspective`(1a) 또는 `equirectangular`(1b)입니다.
두 입력 경로 모두 동일한 `frames` 형식을 출력합니다.

프레임에는 `frame_id`, `timestamp_s`, `source_frame_index`, `view_id`, `image_uri`, `width`, `height`, `intrinsics`, `source_from_view_rotation`을 기록합니다.
내부 파라미터가 아직 없으면 `intrinsics=null`로 두고 Task 2에서 카메라를 추정할 수 있습니다.
Task 2의 등록된 카메라는 null intrinsics를 허용하지 않습니다.

RGB의 `view_id`는 `rgb`, `source_from_view_rotation`은 항등 행렬입니다.
360° view는 `front` 등 고유 view ID와 회전 행렬을 사용합니다. source 좌표는 x 오른쪽·y 아래쪽·z 전방으로 정의하고 panorama 중심을 +z로 둡니다.
view에서 source 방향으로 `ray_source = source_from_view_rotation @ ray_view`입니다.
equirectangular 좌표는 `longitude=atan2(x,z)`, `latitude=atan2(-y,sqrt(x²+z²))`, `u/W=longitude/(2π)+1/2`, `v/H=1/2-latitude/π`를 기준으로 하고 실제 stitching convention 변환은 parameters에 기록합니다.
하나의 panorama 원본 프레임에서 나온 뷰는 동일 timestamp/index를 가지지만 `frame_id`는 다릅니다.

### 재구성

점군 파일은 x/y/z와 red/green/blue(0~255)를 가진 PLY이며 binary little endian 또는 ASCII를 사용할 수 있습니다.
`point_count`와 vertex 수가 일치해야 합니다. 원본 점 순서는 후속 task에서 유지합니다.
각 등록된 카메라는 frames의 `frame_id`에 대응합니다. 미등록 프레임을 카메라가 복원된 것처럼 채우지 않습니다.

깊이는 필요할 때 `depth_uri`의 NumPy `.npy`, float32, H×W 배열로 저장합니다. 단위는 점군과 같고 값은 광선 거리 대신 카메라 z 깊이(`camera_z`)입니다. 유효하지 않은 깊이는 0입니다.
`confidence_uri`는 선택적 H×W float32 `[0,1]` 배열입니다. 없으면 null입니다.
깊이가 없으면 `depth_uri`와 `depth_convention` 모두 null입니다. 해당 입력을 요구하는 분할 backend는 자신의 진입점에서 확인하고 미지원으로 보고해야 합니다.
Task 3이 PLY만으로 모든 backend를 실행할 수 있다는 뜻이 아닙니다.

### 인스턴스 및 의미 정보

`point_labels_uri`는 UTF-8 CSV이며 헤더는 정확히 `point_index,instance_id`입니다.
원본 점 하나당 한 행이고, 미할당은 빈 instance_id입니다. 한 점에는 하나의 인스턴스만 부여합니다.
`instances`에는 instance_id, class_name, class_confidence, confidence, point_count, evidence_frame_ids를 기록합니다.
`class_name=null`, `class_confidence=null`은 유효한 미분류 결과입니다. 분할 영역에 임의로 클래스 이름을 붙이지 않습니다.
대형 데이터에서 CSV 성능이 문제가 되면 별도 규약 버전으로 array 포맷을 합의합니다. 임의 변경은 금지합니다.

### 객체 파일

`objects`의 각 항목은 하나의 instance_id에 대응합니다. 메시 생성에 실패한 객체는 무조건 가짜 mesh로 채우지 않고, 실행 기록에 실패 이유를 남깁니다.
`mesh_format`은 glb 또는 obj입니다. OBJ의 MTL은 `material_uri`, 부속 texture는 `texture_uris`에 기록합니다.
GLB에 모두 내장되면 두 필드는 null/빈 배열입니다.
`appearance`는 none/vertex_color/texture이며 실제 파일에 담긴 표현을 기록합니다.
관측된 부분만 저장했으면 `geometry_status=observed_partial`, 표면 복원을 적용했으면 `reconstructed_surface`입니다. 후자도 완전한 실측 형상을 보장하지 않습니다.

### 공간 정합과 변화탐지

입력은 before와 after라는 두 objects manifest입니다. 같은 scene_id, 서로 다른 capture_id여야 합니다.
`baseline_objects_manifest`가 기준 회차이고, `current_objects_manifest`가 현재 회차입니다.
`baseline_from_current`는 **현재 reconstruction → 기준 reconstruction** 변환입니다.

`p_baseline = T_baseline_current @ p_current`이며, SE3의 상위 3×3은 R, Sim3는 sR입니다.
별도 scale 필드를 중복 적용하지 않습니다. 객체 local pose를 사용할 때는 먼저 world_from_object를 적용합니다.
`rmse` 및 `displacement_in_baseline`의 단위는 기준 reconstruction 단위입니다. metric이 아니면 m로 표시하지 않습니다.
distance threshold도 기준 단위가 정해진 후 task 설정에서 정합니다. 기본값 null은 아직 임계값을 결정하지 않았다는 뜻입니다.

변화 종류는 added/removed/moved/shape_changed/unchanged/unknown입니다.
added는 before ID가 null, removed는 after ID가 null입니다. matched 변화는 양쪽 ID를 모두 기록합니다.
`displacement_in_baseline`은 정합된 현재 centroid에서 기준 centroid를 뺀 벡터입니다.
정합 status가 failed 또는 insufficient_overlap이면 행렬은 null이고 확정된 변화 대신 unknown만 출력합니다.
정합에 성공해도 가림·촬영 누락으로 관측되지 않은 객체를 자동으로 removed라고 단정하지 않습니다.

### 대표 키프레임 선별

objects → instances → reconstruction → frames → video 참조로 원본 출처를 찾습니다.
선택 항목은 instance_id, frame_id, source_frame_index, timestamp_s, view_id, image_uri, score, reason을 기록합니다.
동일 객체에 여러 프레임을 선택할 수 있으나 같은 객체/프레임 쌍을 중복 저장하지 않습니다.
image_uri는 frames에 기록된 추출 원본 프레임을 가리킵니다. 썸네일·crop이 필요하면 별도 확장 필드와 버전을 합의합니다.

## 5. 검증과 규약 변경

```bash
python -m spacewatch3d validate examples/fixtures/changes.json --check-files
python -m spacewatch3d check-fixtures --task 5a
python -m unittest discover -s tests -v
```

`validate`는 상위 manifest를 재귀적으로 읽습니다. 기본 검증은 schema와 ID 연결, `--check-files`는 참조 asset 존재와 CSV 점 라벨까지 확인합니다.
이 검증기는 영상 codec, PLY/mesh 내부 수치, 깊이 배열 shape, 회전 행렬의 정규직교성, 투영 정확도, 실제 물체 가시성을 검사하지 않습니다. 각 backend의 의미 있는 테스트에서 다룹니다.
키프레임의 원본 시각·경로, 변화 객체의 회차 ID, 원본 점 라벨 순서와 같은 task 경계 오류는 공통 검사에서 잡습니다.

규약 변경 PR에는 schema·이 문서·소비 task·fixtures·공통 테스트 변경을 함께 포함합니다.
