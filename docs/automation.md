# 영상·RGB 이미지 → 조밀 점군 자동화

`scripts/video_to_mvs.py`는 기존 수동 작업의 명령을 Python으로 순서대로 실행한다.
Python 표준 라이브러리만 사용하며 SfM과 MVS 계산은 설치된 COLMAP과 OpenMVS가 담당한다.

## 실행 환경

- Linux, Python 3.10 이상
- PATH에서 호출 가능한 `colmap`, `InterfaceCOLMAP`, `DensifyPointCloud`
- 영상 입력에는 `ffmpeg`도 필요하다. 이미지 폴더와 압축 입력에는 FFmpeg가 필요하지 않다.
- 기본 설정의 기준 환경: COLMAP 3.7, OpenMVS 2.4.0 CPU 빌드
- 실행 시 설치나 다운로드는 하지 않는다. 새 COLMAP 버전에서 옵션 이름이 바뀌면 코드의 명령 구성을 수정해야 한다. 주요 CPU 옵션 호환성은 실행 전에 검사한다.

## 전체 실행

저장소 루트에서 실행한다. 기존 수동 결과와 다른 새 출력 폴더를 지정한다.

```bash
python3 scripts/video_to_mvs.py rgbd_dataset_freiburg1_xyz-rgb.avi \
  --output runs/freiburg1_xyz_5fps \
  --fps 5 \
  --threads 8
```

기본값은 5fps, CPU 스레드 8개, 보정 이미지 최대 변 길이 640, 전체 쌍 매칭이다.
동일 카메라 내부 파라미터를 공유하는 `SIMPLE_RADIAL` 모델을 사용한다.
깊이 센서 데이터나 정답 궤적은 입력받지 않는다.

긴 영상에서 계산량을 줄이려면 `--fps`를 낮추거나 `--matcher sequential`을 선택할 수 있다.
순서 기반 매칭의 나머지 설정은 설치된 COLMAP의 기본값을 사용한다.
등록률이 낮아지면 프레임 간격과 촬영 중 장면 겹침을 다시 확인한다.

## RGB 이미지 폴더 또는 `.tgz` 입력

영상 대신 압축을 푼 `rgb/` 폴더를 첫 번째 인자로 지정한다.
FFmpeg로 프레임을 추출하지 않고, 선택한 이미지를 출력 폴더에 복사한 뒤 COLMAP 특징점 추출부터 실행한다.
원본 이미지의 파일명과 파일 내용은 유지한다. `rgb/`를 바로 아래에 가진 데이터셋 폴더를 지정해도 된다.

```bash
python3 scripts/video_to_mvs.py /path/to/dataset/rgb \
  --output runs/dataset_images --threads 8
```

`rgb/`가 포함된 `.tgz` 또는 `.tar.gz`도 직접 입력할 수 있다.
압축 내부에서 RGB 이미지 파일만 읽어 준비하고, `depth/`, 정답 궤적 등의 파일은 사용하지 않는다.

```bash
python3 scripts/video_to_mvs.py /path/to/dataset.tgz \
  --output runs/dataset_archive --threads 8
```

이미지가 많으면 `--image-step 6`처럼 파일명 순서로 6장마다 1장씩 선택한다.
기본값은 1이며 모든 RGB 이미지를 사용한다. 이 옵션은 장수 기준이며 FPS나 초 단위 샘플링이 아니다.
TUM 타임스탬프 또는 자릿수를 맞춘 일련번호처럼 파일명 정렬이 촬영 순서와 일치하는 이름을 사용한다.

```bash
python3 scripts/video_to_mvs.py /path/to/dataset/rgb \
  --output runs/dataset_every6 --image-step 6 --matcher sequential
```

이미지/압축 입력에 `--fps`나 `--duration`을 지정하면 오류로 안내한다. 선택 후 최소 3장이 필요하다.
지원 확장자는 PNG, JPG, JPEG이며 대소문자를 구분하지 않는다. 한 입력의 이미지들은 동일 카메라 내부 파라미터를 공유하는 방식으로 복원한다.

폴더 입력은 `rgb/` 직속 파일 또는 지정한 폴더 직속 파일을 사용한다.
압축 입력은 이미지가 들어 있는 `rgb/` 폴더를 우선 선택하고, 없으면 이미지가 들어 있는 폴더가 하나일 때만 선택한다.
후보 폴더가 여러 개이면 압축을 풀고 사용할 폴더를 직접 지정한다.
압축 경로를 그대로 풀지 않고 선택한 일반 이미지 파일만 저장한다. 경로 이탈, 중복 파일명과 RGB 링크 파일은 거부한다.

이미지 준비 단계는 기존 상태 파일과 동일하게 `frames`라는 이름을 사용한다.
`frames/input_images.json`에는 입력 이미지 경로와 출력 파일명의 대응 관계를 저장한다.
입력 폴더와 출력 폴더는 서로 포함되지 않는 별도 위치로 지정한다.

## 미리보기, 단계 실행, 재개

명령만 출력하고 복원하지 않기:

```bash
python3 scripts/video_to_mvs.py rgbd_dataset_freiburg1_xyz-rgb.avi \
  --output runs/freiburg1_xyz_5fps --dry-run
```

SfM까지만 실행하기:

```bash
python3 scripts/video_to_mvs.py rgbd_dataset_freiburg1_xyz-rgb.avi \
  --output runs/freiburg1_xyz_5fps --stop-after sfm
```

같은 설정으로 나머지 단계 실행하기 또는 중단 후 재개하기:

```bash
python3 scripts/video_to_mvs.py rgbd_dataset_freiburg1_xyz-rgb.avi \
  --output runs/freiburg1_xyz_5fps --resume
```

`--stop-after`는 `frames`, `features`, `matching`, `sfm`, `undistort`, `convert`, `mvs`, `validate` 중 선택한다.
기본값인 `validate`까지 실행하면 결과 검증과 COLMAP 호환 PLY 사본 생성까지 완료한다.
`--duration 8`은 영상 시작부터 8초만 처리한다. 지정했던 FPS, 이미지 선택 간격, 길이, 크기, 매칭 방식 등의 설정은 재개할 때 동일하게 전달한다.

완료 단계는 파일 SHA-256을 확인한 후 재사용한다. 실패하거나 중단된 단계의 전용 폴더는 비우고 그 단계부터 다시 계산한다.
예를 들어 MVS 도중 중단하면 MVS 전체를 다시 실행하지만 SfM과 왜곡 보정은 재사용한다.
특징점·매칭·SfM 단계는 DB 사본을 각각 사용한다. COLMAP mapper가 DB를 열면서 기록하는 변경도 이전 단계의 재개 기준에 영향을 주지 않는다.

입력, 설정, 스크립트 내용 또는 실행 파일이 바뀌면 재개를 거부한다. 변경 후에는 새 `--output`을 사용한다.
영상/압축 파일은 전체 파일 해시, 이미지 폴더는 선택된 파일 목록과 각 파일의 해시로 비교한다.
따라서 이미지 입력 지원이 추가되기 전 버전의 실행 결과도 현재 코드로 재개할 수 없으며 새 출력 폴더가 필요하다.
이 스크립트가 만든 `pipeline.json`이 없는 기존 수동 결과를 자동으로 가져오는 기능은 없다.
Ctrl+C로 중단하면 실행 중인 자식 프로세스를 종료하고 실패 상태를 저장한다.
한 출력 폴더에 두 실행이 동시에 쓰는 것은 잠금으로 방지한다.

## 출력 구조

```text
runs/freiburg1_xyz_5fps/
├── pipeline.json          # 입력·설정·단계 상태·소요 시간·결과 해시
├── logs/                  # 각 외부 명령의 stdout/stderr, 재실행 시 이어 기록
├── frames/                # 영상에서 추출한 PNG 또는 준비한 원본 RGB 이미지
├── features/database.db   # 특징점 추출 결과
├── matching/database.db   # 특징점과 매칭 결과
├── sfm/
│   ├── database.db        # mapper 전용 매칭 DB 사본
│   ├── sparse/            # COLMAP이 생성한 모든 모델
│   ├── model/             # 가장 많은 이미지가 등록된 모델의 사본
│   ├── text/              # 선택 모델의 TXT 사본
│   └── sparse.ply
├── undistort/
│   ├── images/            # 보정 이미지
│   └── sparse/            # 대응하는 보정 카메라·희소 모델
├── openmvs_input/scene.mvs
├── openmvs/
│   ├── scene_dense.mvs
│   ├── scene_dense.ply
│   ├── depth*.dmap
│   └── densify.cfg
└── result/
    ├── summary.json
    ├── scene_dense_colmap.ply
    └── view_neighbors.txt
```

`summary.json`에서 입력 종류, 입력·등록 이미지 수, 희소점·조밀점 수, 깊이 지도 수와 검증 범위를 확인한다.
COLMAP GUI에서는 `result/scene_dense_colmap.ply`를 열면 된다.
이미지 참조가 상대 경로이므로 후속 작업에 사용할 때 출력 폴더 구조를 함께 유지한다.
`runs/`는 큰 DB와 깊이 지도가 쌓이므로 Git에서 제외한다. 디스크 사용량은 이미지 수와 해상도에 따라 증가한다.

## 복원 및 검증 범위

SfM 모델이 나뉘면 등록 이미지 수가 가장 많은 모델을 선택하고, 동률이면 3D 점 수로 선택한다.
기본적으로 입력 이미지의 80% 이상과 최소 3장이 선택 모델에 등록되어야 계속 진행한다.
이 기준은 `--min-registered-ratio`로 조정할 수 있다.

OpenMVS에는 이웃 5장, 초기 추정 3회, 기하학적 검사 2회, 최소 2시점 융합을 설정한다.
깊이 지도를 보관하고 자동 ROI 자르기와 타워 모드는 끈다.

자동 검증은 다음 범위다.

- 특징점 및 기하 검증 매칭이 존재하는지 확인
- SfM 등록 비율, 보정 이미지 수, 주요 출력 파일 확인
- 저장된 `.mvs`를 다시 읽고 이웃 목록 내보내기 (재복원하지 않음)
- 조밀 PLY의 점 수에 맞는 데이터 길이와 좌표·법선의 유한값 확인
- 깊이 지도 파일의 개수와 비어 있지 않은지 확인
- 원본 PLY의 점 데이터를 유지하며 RGB 헤더 자료형 `uint8`을 `uchar`로 바꾼 호환 사본 생성

깊이 지도 픽셀 전체의 정확도, 장면 형상, 실제 미터 단위 스케일은 검증하지 않는다.
메시, 텍스처, segmentation, 촬영 회차 간 정합·변화 비교는 포함하지 않는다.

## 코드 테스트

```bash
python3 -m unittest discover -s tests -v
```

제어 흐름 테스트는 외부 복원 도구 없이 실행한다. 단계 재개, 명령 실패, 결과 손상,
설정 변경, 동시 실행, 공백·특수문자 경로, PLY 호환 변환을 검사한다.

2026-09-14 검증 결과: 자동 테스트 13개 통과. 실제 `freiburg1_xyz` RGB 영상을
1fps로 추출한 27장으로 전체 파이프라인을 실행해 27장 모두 등록, 희소점 5,614개,
깊이 지도 27개, 조밀점 121,667개를 생성했다. 매칭 단계에서 종료한 뒤 `--resume`으로
나머지 단계를 실행했고, 완료 후 재실행에서도 모든 단계가 재계산 없이 재사용됐다.
실제 실행 기록은 `runs/automation_test_1fps/pipeline.json`과 `result/summary.json`에 있다.
이는 1fps 설정의 실행 검증이며 기존 5fps 결과와 점 수를 직접 비교하는 품질 평가는 아니다.

이미지 입력 확장 후 자동 테스트는 29개가 통과했다. 기존 샘플의 RGB 이미지 폴더 27장으로
FFmpeg 없이 전체 복원을 실행해 27장 모두 등록, 희소점 5,775개, 깊이 지도 27개,
조밀점 155,329개를 생성했다. 완료 후 `--resume`으로 모든 단계를 재사용하는 것도 확인했다.
기록은 `runs/image_folder_test/result/summary.json`에 있다.
같은 이미지를 `sample_dataset/rgb/` 구조로 묶은 `.tgz`에서는 `--image-step 3`으로
9장을 선택해 실제 특징점 추출과 재개를 확인했다. 기존 영상 입력도 1초 구간에서
5fps 프레임 추출을 다시 확인했다. 사용자의 별도 데이터셋에 대한 품질 평가는 포함하지 않는다.
