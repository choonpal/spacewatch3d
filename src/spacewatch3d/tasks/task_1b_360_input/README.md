# Task 1b · 360° 영상 입력

Equirectangular 영상을 여러 perspective view로 변환한다. Task 1a와 같은 frames 규약을 출력하여 Task 2가 입력 종류에 의존하지 않도록 한다.

상태: **개발 scaffold**. 실제 backend는 아직 연결되지 않았으며 `run`은 명시적으로 미구현 오류를 반환한다. 담당자 배정은 팀에서 정한다.

## 작업 범위

- 이 디렉터리의 `pipeline.py`, backend 모듈, 설정, 의존성 설명을 수정한다.
- task 전용 테스트는 `tests/task_1b_360_input/`에 추가한다.
- 공통 규약 변경은 별도 PR에서 소비 task와 합의한다.
- 브랜치 예: `task/1b-360_input/my-feature`.

## 입력과 출력

| 입력 이름 | artifact kind | 독립 개발용 입력 |
|---|---|---|
| `video` | `video` | `examples/fixtures/panorama/video.json` |

출력 kind: `frames`. 완성된 형태의 **합성 예제**: [`panorama/frames.json`](../../../../examples/fixtures/panorama/frames.json).
공통 규약: [docs/contracts.md](../../../../docs/contracts.md).

## 바로 확인

저장소 루트에서 가상환경을 활성화하고 `python -m pip install -e .`를 먼저 실행한다.

```bash
python -m spacewatch3d check-fixtures --task 1b
```

위 명령은 파일·ID·규약만 검증한다. 모델 추론이나 task 알고리즘을 실행하지 않는다.

구현을 연결한 뒤 실행할 명령:

```bash
python -m spacewatch3d run --task 1b --input video=examples/fixtures/panorama/video.json --output outputs/task-1b-trial --config src/spacewatch3d/tasks/task_1b_360_input/config.example.json
```

`run(request: TaskRequest) -> Path`를 구현하고, 출력 디렉터리를 생성한 뒤 출력 manifest 경로를 반환한다.
현재는 exit code 2의 미구현 오류가 정상이며, 결과 파일을 성공한 것처럼 만들지 않는다.

## 첫 구현 완료 기준

- [ ] 360° 원본에서 시점별 perspective image 생성
- [ ] view_id·가상 카메라 intrinsics·source_from_view_rotation 기록
- [ ] 동일 원본 프레임의 여러 뷰는 같은 source_frame_index와 timestamp_s 유지
- [ ] 자신의 결과 JSON을 `python -m spacewatch3d validate 결과.json --check-files`로 검증
- [ ] 실행 환경·모델 revision·입력·재현 명령·한계를 README에 기록
- [ ] 합성 fixture 통과와 실제 데이터 성능 평가를 구분해서 보고

## 의존성

공통 환경은 Python 3.10+와 jsonschema만 사용한다. `requirements.txt`는 task 담당자가 검증한 의존성을 기록하는 자리다.
모델별 Python/CUDA 충돌이 있으면 task별 가상환경을 사용해 manifest와 파일로 연결한다.
외부 코드는 `third_party/`, 가중치는 `weights/`, 실제 입력은 `data/`에 두며 Git에 포함하지 않는다.

---

## 실제 360° 영상의 3D map 생성 결과 (2026-10-06)

RICOH THETA X로 촬영한 `R0010003.MP4`를 Colab에서 `stella_vslam_dense`로 처리해 **RGB 포인트 327,097개의 3D point cloud**를 생성했다. 원본 파노라마를 equirectangular 카메라 모델로 직접 입력한 실험이다.

이 결과는 별도로 실행한 Colab 실험 기록이다. 위에서 정의한 Task 1b의 perspective view 생성, `frames` manifest 출력 및 `pipeline.py` backend 연결을 완료했다는 의미는 아니다.

### 입력 및 복원 설정

| 항목 | 값 |
|---|---|
| 카메라 | RICOH THETA X — 원본 MP4 메타데이터에서 확인 |
| 입력 영상 | `R0010003.MP4`, H.264, 3840 × 1920, 2:1 equirectangular |
| 길이 / 원본 프레임 수 | 컨테이너 기준 약 68.549초 / 영상 스트림 2,055프레임 |
| 원본 평균 프레임률 | 약 30.0048 fps |
| 복원 프로그램 | `stella_vslam_dense`, revision `8f91e7eb01eadca72b21984642e88bf877598ace` |
| 실행 방식 | Colab의 별도 Python 3.12 가상환경에서 headless 실행 |
| 추적 해상도 | 1920 × 960 |
| 깊이 계산 해상도 | 640 × 320 |
| 입력 간격 | `--frame-step 3`, 약 10 fps에 해당하는 간격으로 총 685프레임 처리 |
| 촬영자 제외 마스크 | 영상 하단 35% 제외 (`mask_start=0.65`), 파노라마 자체는 자르지 않음 |
| 주요 옵션 | dense reconstruction 및 loop detection 활성화, `--auto-term`, `--wait-loop-ba` |

설정 근거는 결과 폴더의 `input_metadata.json`, `run_settings.json`, `theta_x.yaml`, `operator_mask.png`다. 하단 마스크는 촬영자뿐 아니라 해당 영역의 바닥 정보도 제외한다.

### 생성 결과와 실행 기록

| 항목 | 확인 결과 |
|---|---|
| Dense point cloud | `dense.ply`, ASCII PLY, XYZ + RGB, **327,097개 점** |
| PLY 파일 크기 | 13,145,099 bytes, 약 13.15 MB |
| SLAM map database | `map.db`, 254,324,736 bytes, 약 254.32 MB |
| 프레임 처리 | **685 / 685** 완료, 로그의 `End of video.` 확인 |
| 저장된 카메라 pose | TUM 형식 683개; 로그에 내부 frame 1~2의 trajectory 누락 경고가 기록됨 |
| 저장된 키프레임 | 97개, keyframe ID 0~96 |
| 루프 폐쇄 | keyframe **2 ↔ 95** 검출, loop bundle adjustment 완료 |
| 복원 소요 시간 | 로그 첫 설정 로드부터 마지막 map 정리까지 **203.38초, 약 3분 23초** |
| 데이터 검사 | 실제 PLY 행 수와 요약값 일치, 모든 XYZ 유한값, RGB 0~255 범위 확인 |

소요 시간에는 SLAM 처리, 종료 및 결과 저장이 포함되며 설치·컴파일·영상 업로드 시간은 포함되지 않는다. `summary.json`과 실제 PLY, `reconstruction.log`, ZIP 내부 trajectory 파일을 함께 확인한 수치다.

### 결과 이미지

아래 이미지는 결과 `dense.ply`를 직접 렌더링했다. **327,097개 점을 모두 사용**했으며, 시각화를 위한 다운샘플링·이상점 제거·메시 생성은 하지 않았다. 좌표는 저장된 `preview.html`과 같이 `(X, Z, -Y)`로 표시한다. 축 값은 미터로 보정되지 않은 단안 복원 좌표다.

**3D 전체 보기**

![R0010003의 RGB point cloud 전체 보기](assets/r0010003_20261006/pointcloud_overview.png)

길게 이어진 공간의 윤곽과 일부 수직 구조를 확인할 수 있다. 표면이 비어 있는 부분과 주 구조 주변에 흩어진 점도 남아 있다.

**위에서 본 point cloud와 카메라 이동 경로**

![R0010003 point cloud의 X-Z 투영과 카메라 경로](assets/r0010003_20261006/pointcloud_top_trajectory.png)

주황색 선은 ZIP에 저장된 `trajectory/frame_trajectory.txt`의 683개 카메라 위치를 시간순으로 연결한 것이다. 초록 원은 시작, 분홍 X는 끝을 나타낸다. 경로는 식별을 위해 point cloud 위에 겹쳐 그렸으며 시작 부근으로 돌아오는 형태가 보인다. 시작점과 끝점이 정확히 일치하지 않는 것만으로 위치 오차를 산출할 수는 없다.

### 해석과 현재 한계

- 360° 영상으로부터 dense point cloud, SLAM map 및 카메라 경로를 생성하고 루프 최적화까지 수행했다.
- 전체적인 공간 구조는 시각적으로 확인되지만, 빈 영역과 흩어진 점이 있어 완전한 표면 복원이나 정밀 도면으로 간주하지 않는다.
- 단안 복원이므로 절대 스케일은 미보정 상태다. 길이·면적을 미터 단위로 사용하려면 실측 기준 길이를 이용한 스케일 보정이 필요하다.
- Ground truth 또는 실측과의 정합·오차 평가는 수행하지 않았다. 유효한 점이 생성되고 루프 폐쇄가 완료됐다는 사실만으로 기하 정확도가 검증되지는 않는다.
- 물체 instance segmentation, 객체별 export, 변화 감지 및 프로젝트 공통 artifact 규약 연결은 이 실험에서 수행하지 않았다.

### 원본 산출물 위치 및 다시 확인하는 방법

입력과 결과는 로컬에 다음과 같이 보관되어 있다.

```text
입력: /home/kdj/Desktop/capstone/360video/R0010003.MP4
결과: /home/kdj/Desktop/capstone/360video/R0010003 zip/
```

| 파일 | 용도 |
|---|---|
| `dense.ply` | RGB point cloud 원본, CloudCompare 또는 MeshLab 등으로 확인 |
| `map.db` | Stella SLAM map database |
| `preview.html` | 회전 가능한 Plotly 미리보기; 노트북 코드상 최대 50,000개 점 표시 |
| `summary.json` | 점 개수, 파일 크기, 좌표 범위, 스케일 안내 |
| `run_settings.json`, `theta_x.yaml` | 실행 명령, revision 및 복원 설정 |
| `input_metadata.json`, `operator_mask.png` | 원본 영상 메타데이터 및 마스크 |
| `reconstruction.log` | 초기화, 처리 진행, 루프 최적화, 저장 기록 |
| `R0010003_20261006_133052_202712.zip` | 위 결과와 `trajectory/`를 포함하는 전체 묶음 |

현재 결과 폴더에서는 `trajectory/`가 별도로 풀려 있지 않으며, ZIP 안에 `frame_trajectory.txt`, `keyframe_trajectory.txt`, `tracking_times.txt`가 들어 있다. 원본 MP4·PLY·DB·ZIP은 이 README에 복사하지 않고, 아래 PNG와 이미지 재생성 스크립트만 문서 자산으로 추가했다.

복원에 사용한 [Colab 노트북](https://colab.research.google.com/drive/1Ok7v3GdK-oeHMj4mz4iL1OQeZLQaQw2G?usp=sharing)에서 설치·설정 후 실행한 명령은 다음과 같다. 입력·vocabulary·설정·마스크가 존재하고 필요한 라이브러리가 설치된 Colab 환경을 전제로 한다.

```bash
export PYTHONPATH=/content/stella_vslam_dense_8f91e7e/build/lib
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
RESULT_DIR="$(mktemp -d /content/stella_theta/R0010003_replay_XXXXXX)"
mkdir -p "$RESULT_DIR/trajectory"

/content/stella-venv312/bin/python -u \
  /content/stella_vslam_dense_8f91e7e/tools/run_video_slam.py \
  -v /content/stella_theta/orb_vocab.fbow \
  -c /content/stella_theta/theta_x.yaml \
  -m /content/R0010003.MP4 \
  --mask /content/stella_theta/operator_mask.png \
  --frame-step 3 --disable-viewer --auto-term --wait-loop-ba \
  -o "$RESULT_DIR/map.db" -p "$RESULT_DIR/dense.ply" \
  --eval-log-dir "$RESULT_DIR/trajectory"
```

출력 경로는 재실행 시 기존 결과를 덮어쓰지 않도록 새 폴더로 바꿨다. 설치 노트북에는 Python 3.12 가상환경 지정, CUDA 13에서 제거된 GPU 진단 필드에 대한 조건부 컴파일, OpenCV `imgcodecs` 링크 보완이 반영되어 있다.

README 이미지의 재생성 스크립트는 [`assets/r0010003_20261006/render_results.py`](assets/r0010003_20261006/render_results.py)다. 저장소 루트에서 NumPy와 Matplotlib이 설치된 환경으로 실행한다. 원본 PLY는 읽기만 하며, trajectory는 압축을 풀지 않고 ZIP에서 읽을 수도 있다.

```bash
python3 -s src/spacewatch3d/tasks/task_1b_360_input/assets/r0010003_20261006/render_results.py \
  '/home/kdj/Desktop/capstone/360video/R0010003 zip'
```

이미지는 로컬 Python의 NumPy 1.x / Matplotlib 3.5.1 조합으로 생성했다. 위 `-s`는 이 PC의 사용자 설치 NumPy와 시스템 Matplotlib 간 충돌을 피하기 위해 사용자 site-packages를 제외하는 옵션이다.
