# 기존 객체 메쉬·RGB를 받는 SceneDiff 비교 어댑터

## ViewDelta·SimSaC와 파노라마 비교 (2026-10-25)

`prepare_model_comparison.py`는 기존 메쉬 입력 SceneDiff와 동일한 view-overlap 계산으로
30개 뷰의 비교 상대를 고정한다. 변경 점수로 대응을 선택하지 않는다. 대응하는 원본 ERP
20프레임도 1024×512로 준비한다.

`run_pair_models.py --model viewdelta|simsac`는 각 모델의 실제 공개 학습 가중치를
strict=True로 로드한다. 전체 ERP 입력과 동일 224×224 방향 뷰 입력을 따로 실행하고,
비교 상대별 확률 평균 > 0.5로 후보 마스크를 만든다. 공개 모델의 기존 전처리 해상도는
ViewDelta 256×256(DINOv2 processor 포함), SimSaC 520×520과256×256이다.
일반 이미지 모델의 파노라마 적용 실험이며 구면 전용 구조로 바꾸지 않는다.

ViewDelta는 사용하지 않는 DeepSpeed import만 제외한다. SimSaC는 train.py의 원래
use_pac=False 구조와 ChangeSim 결합 체크포인트를 사용하고, CUDA/CuPy의9×9 correlation은
동일한 zero-padding·채널평균을 native PyTorch로 계산한다. 네트워크나 검출 head를
새로 초기화하여 결과를 내지 않는다. 가중치 SHA256·source commit·strict 로드와 자기
비교 대조 결과를 저장한다. 개별 쌍 확률의 저장 정밀도는 float16, 공통 뷰 평균은 float32다.

GPU 작업 폴더는 `/workspace/pano_change_models_20261025/`이며 모델을 GPU1·GPU2에서
별도 프로세스로 실행한다. 기존 환경은 수정하지 않고 의존성을 별도 deps/에 설치한다.
결과를 `outputs/scenediff-mesh-rgb-20261024/model-comparison-20261025/`에 내려받고
`python tools/scenediff_mesh/report_pair_models.py`로 비교 화면을 만든다.

SceneDiff의 통합 객체 ID 수와 새 모델의 2D 연결요소 수는 같지 않다. 주 비교는 동일
뷰의 후보 픽셀 비율과 마스크 IoU이며, 서로 다른 층에 대한 후보의 동의 정도다.
정확도·재현율·정밀도를 계산하거나 순위를 정하는 실험이 아니다.

## 원본 입력과 메쉬 입력의 후보 탐지 비교

사용자가 요청한 입력 경로 비교는 `compare_inputs.py`와 `report_inputs.py`다. 아래의
`run.py`/`report.py`는 앞서 실행한 기하/RGB 비용 절제 실험이며 입력 경로 비교와 다르다.

- A: 동일 RGB → 실제 Pi3 + 원본 수정 SAM ViT-H → SceneDiff 전체 후속 탐지.
- B: 동일 RGB → 기존 메쉬 ray depth·camera + 기존 SAM3.1 instance → 같은 후속 탐지.
- 두 경로 모두 원본 `_detect_and_merge_objects`와 `_associate_and_save_objects`를 실행한다.
  기존 ID에 점수만 붙이거나 라벨로 회차 간 대응을 제한하지 않는다.
- 기존 GPU 환경을 수정하지 않고 별도 `deps/`에 FAISS를 설치했다. 모델 소스와 가중치
  해시, 같은 RGB의 바이트 해시, 후보 RLE, 비용·가시성·입력 마스크 배열을 보관한다.
- 두 경로 공통 224×224, 구역별 두 촬영 각 5장이다. 원본 loader 기본 최대 크기 518의
  전체 해상도 벤치마크 재현은 아니다. 원본의 grid와 스케일/voxel 반환 규칙은 유지한다.
- 빈 영역/빈 비용 보호, native scatter mean, 동일 voxel 평균의 스트리밍 집계만 공통
  호환 처리한다. 메쉬 valid ray hit confidence=1이며 학습된 Pi3 confidence와는 다르다.

GPU 실행은 `/workspace/scenediff_input_ab_20261024/`의 별도 작업 폴더에서
`compare_inputs.py --stage all`로 수행한다. 이전 작업 폴더의 prepared RGB와 고정 upstream을
읽는다. 결과를 `outputs/scenediff-mesh-rgb-20261024/paired/`에 받은 뒤 다음을 실행한다.

```powershell
python tools/scenediff_mesh/report_inputs.py
```

보고서는 동일 뷰의 RGB / 원본 후보 / 메쉬 후보 / 후보 픽셀 차이를 표시한다.
최종 통합 후보의 대응은 모든 공유 뷰의 마스크 IoU≥0.5로 일대일 계산한다.
이는 두 방식의 동의 정도이며 정확도 지표가 아니다.

## 앞선 기하/RGB 비용 절제 실험

`prepare.py`는 기존 메쉬, 면별 인스턴스 ID, 원본 ERP, 카메라 포즈와 저장된 정합을 입력으로 받는다. Pi3 재구성과 SAM 분할을 실행하지 않는다. 기존 SAM3.1 결과를 재사용하며 새 객체 면을 생성하거나 누락된 형상을 보완하지 않는다.

`run.py`는 [SceneDiff 공식 코드](https://github.com/yuqunw/scene_diff)의 커밋 `62dcbd285fcfe984bb3db55dc01bdcd1a0055714`에서 다음 비교 함수를 실제로 불러온다.

- `GeometryModel.compute_geometric_costs`
- `SceneDiff.compute_rendering_cost`
- `SceneDiff._compute_cost_maps_for_image`
- `SceneDiff._compute_cluster_costs_single`
- `SceneDiff._compute_changed_region_threshold`

필요한 정의만 AST로 로드하므로 Pi3/SAM 초기화, 사용하지 않는 Open3D/FAISS, 학습 모듈은 GPU 작업에 로드하지 않는다. 원본 함수 소스와 MIT 라이선스는 실험 폴더의 `upstream/`에 보존한다.

## 현재 실행 입력

`outputs/scenediff-mesh-rgb-20261024/inputs/sam31/`은 서버의 기존 `sam31_panorama_test_20261023/delivery`에서 받은 객체 ID와 메쉬다. 각 회차의 112–143번 맵 프레임, 약 16초의 결과다. 사진은 같은 실험의 원본 4K ERP를 읽는다. 카메라와 고정 정합은 `outputs/registration-benchmark-20261019/new-meshfirst-20261022/`에서 읽는다.

면 순서·원본 RGB의 동일성과, 알려진 좌표 변환 후 모든 정점의 일치를 검사한다. 구조물별 정합을 새로 맞추지 않는다. 구역별로 저장된 변환을 메쉬와 카메라에 함께 적용한다. RGB 영상은 90도 시야각의 224×224 perspective 뷰로 변환한다. 변화 점수를 보기 전에 투영된 객체 픽셀 수와 촬영 다양성으로 구역·회차별 최대 5개 뷰를 고른다.

## 실행

로컬에서 입력을 준비한다.

```powershell
python tools/scenediff_mesh/prepare.py
python -m unittest discover -s tests -p test_scenediff_external.py -v
```

GPU 환경에서 `prepared/`, 고정 `upstream/` 소스와 어댑터를 준비한 뒤 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 python code/run.py \
  --root /workspace/scenediff_mesh_rgb_20261024 \
  --upstream /workspace/scenediff_mesh_rgb_20261024/upstream \
  --setup /workspace/scenediff_mesh_rgb_20261024/setup-status.json
```

`--setup` JSON에는 `model`과 `checkpoint_dir`가 필요하다. 이번 모델은 공개 [timm DINOv3 ViT-H+/16 QKV-bias 체크포인트](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m)다. `checkpoint_dir/model.safetensors`를 strict 로딩한다. 원본 DINOv3 RoPE의 bfloat16 period 정밀도를 복원한다. 새 가중치를 학습하지 않는다.

결과를 내려받고 로컬 보고서를 만든다.

```powershell
python tools/scenediff_mesh/report.py
```

## 비교 정의와 원본 대비 변경

- 기하 단독: 동일 입력에서 기하 비용만 사용한다.
- RGB 결합: 원본 설정의 `geometry + 0.1 × region_match + 0.5 × appearance`를 사용한다.
- 두 방식 모두 같은 투영, 객체 ID, 가시성, 복셀 집계 조건을 쓴다. 최대 엔트로피 자동 임계값은 구역·방식마다 따로 추정하므로 후보 수 차이를 정확도 향상으로 해석하지 않는다.
- Pi3 신뢰도 대신 메쉬 유효 hit를 관측 지원으로 사용한다. 수치 1은 보정된 신뢰확률을 뜻하지 않는다.
- 안정된 기존 객체 ID를 보존하므로 회차 내 신규 객체 발견·병합 단계는 건너뛴다.
- `torch_scatter` 대신 동일한 dim=0 평균을 native PyTorch로 계산하며 복셀 비용 배치를 벡터화한다.
- `align_corners=True`에 맞게 투영 grid를 `2/(W-1)`, `2/(H-1)`로 정의한다.
- 외부 ID의 누락·비연속 번호와 겹침 0인 뷰를 처리한다. 지원이 부족한 객체도 결과에서 삭제하지 않는다.
- 회차 간 RGB 대응은 같은 구역·같은 라벨에 대한 일대일 가설이며, 코사인 유사도 0.7과 차선 대비 차이 0.05를 함께 보고한다. 물리적 동일성이나 이동 확정이 아니다.

원본 SceneDiff 논문 성능을 재현한 실험이 아니라 **입력 교체와 기존 ID 보존을 적용한 비교 실험**이다. 단위는 기존 맵 단위이며 정밀한 미터 변위를 산출하지 않는다. Dorm2·Dorm3는 서로 다른 층이므로 시간 변화의 정답으로 사용하지 않는다.

## 기존 정합오차·관측검증과 SimSaC / ViewDelta 비교

```powershell
python tools/scenediff_mesh/build_focus_comparison.py
```

`focus-comparison-20261025/index.html`은 저장된 `automatic-lifecycle-v2` 판정과 두 모델의 실제 추론 캐시를 비교한다. 기존 방식의 양성 표면은 `event_groups.entry_ids` 트랙의 `face-audit.original_face_ids`만 원본 메쉬에서 first-hit 투영한다. 그룹 문맥·보류·억제·판정 연결 없음·메쉬 hit 없음은 따로 집계한다. 주변 의자 메쉬 전체를 양성으로 만들지 않는다.

같은 30개 RGB 뷰의 해시, 기존 준비 데이터와 깊이·유효 hit 일치, 후보 픽셀의 배타적 분해를 검증한다. 모델 원본 확률과 0.5 임계값은 고정한다. 기존 방식은 전체 촬영, 두 모델은 파일럿 20프레임이므로 관측량을 맞춘 정확도 비교가 아니다. `validation.json`에 검증·소스 해시, `baseline-projections.npz`에 투영된 원본 면 인덱스와 판정 bit를 저장한다.

사람 제외가 기본값이다. 기존 YOLO11l의 대상 프레임 사람 분할과 ERP seam을 감싼 검출 박스를 합쳐 두 RGB 모델 후보와 메쉬 투영에 동일하게 적용한다. RGB 사람 뒤의 메쉬 면은 관측 가능한 것으로 세지 않는다. 원본 모델 확률·기존 객체 판정은 유지하며 화면 토글을 끄면 필터 전 결과를 볼 수 있다.
