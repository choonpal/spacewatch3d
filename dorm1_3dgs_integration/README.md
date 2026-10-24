# Dorm1 3DGS integration package

이 폴더는 3D Gaussian Splatting 모델을 다른 구현에 통합하기 위한 코드와 추론 데이터입니다. GitHub 전달본에는 대용량 체크포인트를 제외했습니다. USB로 받은 전체 통합 ZIP에서 아래 파일을 같은 상대 경로에 복사한 뒤 실행하십시오.

```text
model/dorm1_nerfstudio_refined_1024/splatfacto/2026-10-19_221340/nerfstudio_models/step-000059999.ckpt
```

체크포인트 크기: 782,272,513 bytes. SHA-256:

```text
9427BF3F4EB62177037E1107526AFC7F7301D76BF307CC1AB08F84D1747B54DC
```

## 포함 범위

- Nerfstudio Splatfacto 60,000-step 모델 설정 (체크포인트는 USB로 별도 전달)
- 이식 가능한 최소 추론 데이터와 학습 설정
- 정합된 256개 카메라 포즈와 원본 포즈
- 기존 맵 키프레임 20개 카메라 JSON
- 단일/일괄 4096x2048 ERP 렌더러
- 구면 포즈 정렬 기반 참조 선명화
- AUTO TOUR 360 manifest 생성기와 뷰어 소스
- 현재 20개 위치에 대응하는 원본 ERP 참조 프레임
- 결과 대조용 4K 이미지 1장

원본 1,536개 큐브면 학습 이미지와 원본 전체 영상은 포함하지 않았습니다. 체크포인트 추론에는 필요하지 않으며, 재학습이 필요할 때만 별도로 전달하면 됩니다.

## 요구사항

- Windows 10/11 x64
- NVIDIA CUDA 지원 GPU. 제작 환경은 RTX 40 계열, compute capability 8.9입니다.
- NVIDIA CUDA Toolkit 12.8
- Visual Studio 2022 Build Tools의 C++ 워크로드
- Python 3.12
- 최초 환경 설치 시 인터넷 연결

검증된 핵심 버전은 `torch 2.7.1+cu128`, `nerfstudio 1.1.5`, `gsplat 1.4.0`입니다.

## 빠른 실행

PowerShell에서 패키지 루트로 이동한 뒤 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_environment.ps1
.\scripts\render_camera.ps1 -CameraJson .\cameras\keyframes\map_keyframe_01_camera.json
```

20개 지점을 모두 렌더링하고 참조 선명화 및 투어 manifest까지 만들려면:

```powershell
.\scripts\render_all_keyframes.ps1
.\scripts\start_viewer.ps1
```

참조 이미지를 전혀 사용하지 않은 순수 3DGS 결과가 필요하면:

```powershell
.\scripts\render_all_keyframes.ps1 -SkipReferenceSharpening
```

## 새로운 가상 위치 렌더링

`docs/camera-coordinate-system.md` 형식으로 카메라 JSON을 만든 뒤 실행합니다.

```powershell
.\scripts\render_camera.ps1 `
  -CameraJson C:\path\to\new_camera.json `
  -OutputDir .\outputs\new_view `
  -FaceSize 1024 `
  -OutputWidth 4096
```

출력 ERP는 2:1 비율이며, 이미지 중심은 `look_at_glb`의 수평 방위입니다. 세계 +Y가 ERP 상단 방향이므로 파노라마 수평선이 기울지 않습니다.

## 통합 시 중요한 구분

- `render_360_keyframes.py` 출력은 전 영역이 3DGS에서 렌더링된 결과입니다.
- `reference_sharpen_erp.py`는 동일 촬영 위치가 존재하는 20개 지점에만 사용할 수 있습니다. 원본 ERP를 구면 정렬하고 구조가 일치하는 부분의 명도 고주파만 제한적으로 이식합니다.
- 완전히 새로운 가상 위치에는 대응 원본이 없으므로 참조 선명화를 적용하지 마십시오.
- `config.yml`에는 제작 PC의 옛 절대 경로가 남아 있지만, 제공된 렌더러가 `--checkpoint-root`와 `--data-dir`로 실행 시 덮어씁니다. config 파일을 직접 수정할 필요가 없습니다.

## 폴더 구조

```text
model/              checkpoint와 Nerfstudio config
inference_data/     transforms.json, 초기 point cloud, 검증 보고서
cameras/            원본/정합 포즈와 20개 키프레임
reference_frames/   현재 키프레임용 원본 ERP 20장
src/                변환·정합·렌더·선명화 코드
scripts/            환경 설치와 실행 진입점
viewer/             AUTO TOUR 360 로컬 뷰어 소스
docs/               좌표계와 통합 명세
expected_output/    결과 대조용 첫 번째 4K ERP
```

## 재학습

재학습에는 패키지에 없는 원본 ERP 전체와 1,536개 큐브면 데이터가 필요합니다. `src/prepare_nerfstudio_cubemap.py`, `src/refine_camera_poses.py`, `src/validate_nerfstudio_dataset.py`, `src/train_dorm1_refined_1024.ps1`는 재현을 위해 포함했지만 입력 데이터 경로는 새 환경에 맞춰 지정해야 합니다.
