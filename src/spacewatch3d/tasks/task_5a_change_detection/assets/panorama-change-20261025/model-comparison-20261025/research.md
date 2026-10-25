# CYWS 원본 가중치 재사용과 두 촬영 영상의 변화 탐지

확인일: 2026-10-25 (Asia/Seoul). 2026-10-24 파노라마 조사 후속이다. 공식 소스·논문·가중치 배포 응답과 공개 파일 목록을 읽었으며 모델 설치·추론·원격 서버 작업은 하지 않았다. 현재 ViewDelta/SimSaC/SceneDiff 실행 결과와는 별도인 연구 기록이다.

## 1. CYWS-360에 원본 CYWS-3D 가중치를 쓸 수 있는가

**조건을 맞추면 원본 가중치를 사용하는 재학습 없는 시험 경로는 있다.** 다만 현재 CYWS-360의 `use_dinov3=False` 한 줄만으로 원본 모델과 같아지지는 않는다. 변경된 신경망 부분에 초기값이 남는 실행을 ‘사전학습 모델로 실행했다’고 세면 안 된다.

### 공개 가중치의 실제 존재

[CYWS-3D 공식 저장소](https://github.com/ragavsachdeva/CYWS-3D)의 README는 다음 학습 완료 모델과 추론 명령을 제공한다.

- [원본 `cyws-3d.ckpt.gz`](https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz): 이번 조사에서 HTTP HEAD **200**, `Content-Length: 404282479` bytes를 확인했다. gzip을 풀어 `cyws-3d.ckpt`로 사용한다.
- 공식 명령: `python inference.py --load_weights_from ./cyws-3d.ckpt`.
- 이 조사는 체크포인트를 다운로드·역직렬화하지 않았다. 응답과 메타데이터 확인이며 실제 신경망 로딩 성공 증명은 아니다. [공식 실행 안내](https://github.com/ragavsachdeva/CYWS-3D#pre-trained-model)

### 소스에서 확인한 호환 조건

| 항목 | 원본 CYWS-3D | 현재 CYWS-360 | 판정 |
|---|---|---|---|
| backbone | `vit_base_patch8_224_dino` | 기본 `vit_base_patch16_dinov3.lvd1689m` | 서로 다른 특징 표현·패치/위치 처리다. 원본 decoder에 그대로 연결한 것을 학습 완료 DINOv3 모델로 볼 수 없다. |
| 원본 backbone 분기 | DINO **v1**, 768차원, patch 8, stride 4 | `use_dinov3=False`이면 동일한 `vit_base_patch8_224_dino` | fork 주석은 DINOv2라 부르지만 실제 모델 이름은 DINOv1이다. |
| 특징 | layers `[2,11]`의 `attn.qkv`에서 K 특징 | false 분기는 같은 K 특징, true 분기는 block 출력 | 원본 특징 추출 의미를 유지해야 한다. |
| U-Net attention | `attention_type=None` | **`attention_type='scse'`** | 새 SCSE 가중치는 원본에 없다. false 분기만 바꾸면 초기화된 attention이 남는다. |
| decoder / fusion / 검출 head | 학습 완료 | shape가 맞는 키는 로드 가능 | 키·shape의 실제 전량 검사가 필요하다. |
| 기하·해상도 | perspective 등록, 기본 224×224 | 구면 back-projection과 가변 해상도 | 학습 파라미터가 없는 기하 수정 자체는 재학습을 강제하지 않지만 입력 분포는 달라진다. |

근거: [원본 model.py](https://github.com/ragavsachdeva/CYWS-3D/blob/master/modules/model.py), [원본 config.yml](https://github.com/ragavsachdeva/CYWS-3D/blob/master/config.yml), [CYWS-360 model.py](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/src/cyws360/model.py), [fork 설정](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/config.yml).

fork 설정의 `use_dinov3: True` 옆에는 기존 pretrained weights를 시험할 때 False로 바꾸라는 주석이 있다. 그러나 SCSE 변경까지 원본과 일치시키는 절차는 별도로 필요하다. 원본과 fork 모두의 `safely_load_state_dict()`는 shape가 다르면 **현재 초기값으로 치환**하고 `strict=False`를 사용한다. 따라서 예외가 없다는 사실만으로 제대로 불러왔다고 판단할 수 없다.

### 재학습 없는 시험의 입장 조건

다음은 실행 제안이며 이번 조사에서 통과를 확인한 결과가 아니다.

1. 원본 DINOv1 분기를 사용하고 새 SCSE를 제거해 원본의 `attention_type=None`과 계산 구조를 맞춘다. 원본 패치·stride·feature hook·출력 차원을 유지한다.
2. 체크포인트의 실제 최상위 구조를 확인한다. 공식 loader는 `torch.load()` 결과를 직접 state dict로 취급한다. 임의의 prefix 제거·wrapper 가정을 검증 없이 적용하지 않는다.
3. `unet_encoder.*`, `unet_decoder.*`, `feature_fusion_block.*`, `centernet_head.*`, `feature_backbone.sequence_to_spatial.*`의 **학습 파라미터가 모두 동일 키·동일 shape로 로드됐는지** 저장한다. BatchNorm의 `running_mean`, `running_var`, `num_batches_tracked`도 보존한다.
4. backbone도 동일 DINOv1 가중치로 맞추고 missing/unexpected/shape-skipped 키를 기록한다. 초기값 대체가 있으면 어떤 계산 경로에 남는지 확인한 뒤 실행 허용 여부를 결정한다.
5. 그다음 ERP 구면 등록과 해상도 적응을 시험한다. 먼저 원본 demo/perspective 입력이 재현되는지를 대조하면 이식 오류와 ERP 일반화 실패를 구별할 수 있다.

원본 CenterNet은 64채널 특징에서 heatmap(1), width/height(2), offset(2)을 낸다. fusion은 `64+768=832 → 256 → 64` 채널이다. CenterNet의 학습 가중치뿐 아니라 앞단 decoder/fusion 전체를 검사해야 한다. 검출 head만 로드되고 앞단 특징이 달라져도 원본 추론이 아니다.

**실무 판단:** 위 조건을 충족한 ‘원본 CYWS-3D + ERP 기하 어댑터’는 재학습 없이 비교 실험할 수 있다. 파노라마에 적응 학습된 모델이라고 부를 수는 없다. **현재 DINOv3 + SCSE 구조를 유지하려면 그 표현에 맞는 학습 완료 가중치가 필요하며, 없으면 적응 학습이 필요하다.** 원본 가중치만으로 해당 신구조의 신뢰할 만한 성능이 입증된 것은 아니다.

### PyTorch3D가 없는 실행 환경에서의 제한

원본과 fork의 registration 모듈은 `AlphaCompositor`, `PerspectiveCameras`, `PointsRasterizer`, `PointsRenderer`, `Pointclouds`를 직접 import한다. 실제 warper는 3D 변환한 source feature point를 **forward point splatting**한다. `radius=1.5/min(H,W)*2`, `points_per_pixel=8`, `bin_size=0`, alpha compositing, `eps=1e-5`의 렌더링 경로다. identity/2D 전략도 해당 렌더러를 사용하며, fork에는 PyTorch3D 없는 대체 경로가 없다. [원본 등록 코드](https://github.com/ragavsachdeva/CYWS-3D/blob/master/modules/registeration_module.py), [ERP fork 등록 코드](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/src/cyws360/registeration_module.py)

따라서 **PyTorch3D 부재는 현재 공개 구현을 그대로 실행할 때 실제 장애물**이다. 일반 `grid_sample` 역방향 warp로 바꾸면 투영 방향, 깊이순 가림, 반경 내 top-8 표본, alpha 합성, hole 처리가 달라질 수 있다. 이것을 동일 CYWS 계산이라고 주장할 수 없다. 동일 점 rasterizer를 다른 연산으로 재구현하는 것은 원리상 가능하지만 별도 수치 일치 검증이 필요하다. 빠른 실행을 위해 임의 warp로 대체한 결과와 원본 가중치·계산을 보존한 결과를 구별한다.

## 2. 두 회차 영상 전체를 비교하는 최신 후보

‘두 영상’에는 세 가지 구조가 있다. (a) reference/query 영상을 직접 받아 프레임 대응을 학습하는 모델, (b) 첫 영상을 3D 지도로 만든 뒤 두 번째 영상의 관측을 누적하는 시스템, (c) 선택한 다중 시점 프레임의 오프라인 변화 분석이다. 다음 연구의 대부분은 **일반 perspective RGB**를 사용하며 native ERP가 아니다.

| 방법 | 입력과 시간 처리 | 공개 구현·가중치 상태 | 현재 ERP 영상 적용 판단 |
|---|---|---|---|
| **VSCD / VSCDNet**, ICML 2026 | **비동기 reference/query RGB 영상**. frame-level alignment → query별 다중 reference 후보 → patch 대응·신뢰도 fusion → 프레임별 변화 마스크 | 공식 train/eval 코드 있음. frozen SAM ViT-B는 공개. **학습 완료 VSCDNet 체크포인트 배포는 확인 못함** | 두 회차 영상 비교라는 문제 설정이 가장 직접적이다. perspective 영상 모델이라 ERP 방향별 뷰 변환과 긴 영상 처리 확인 필요. |
| **O-SCD**, CVPR 2026 Highlight | 첫 회차의 3DGS 지도 + 두 번째 회차의 순차 RGB. PnP 등록 후 pixel/feature 변화 단서를 지도에 누적 | 실제 실행 코드 있음. **SAM2.1 tiny 공개 pretrained 로딩**과 **장면별 reference 3DGS**를 사용. 변화장은 테스트 시 최적화 | 영상 관측 누적이 목적에 맞다. 기존 메쉬를 바로 넣는 구조는 아니므로 첫 회차 3DGS 준비가 추가된다. |
| **O-SCD++**, 2026 preprint | O-SCD의 반복 replay 대신 recursive evidence fusion | 공식 프로젝트와 저장소 모두 **code coming soon**, 실제 디렉터리는 README만 확인 | 참고 연구. 현재 실행 후보로 세지 않는다. |
| **MV3DCD**, CVPR 2025 | 변화 전 여러 RGB와 변화 후 소수 RGB → change-aware 3DGS 최적화. 이후 임의 뷰 변화 마스크 | 공개 실행 코드. 사전학습 시각 특징과 장면별 최적화 사용. 범용 변화 검출 head 체크포인트 방식 아님 | 오프라인 다중 시점 비교 후보. 이미 있는 메쉬를 유지하는 단순 입력 교체와 다르다. |
| **SceneDiff**, ECCV 2026 | 두 RGB 영상에서 프레임·기하·마스크·특징을 얻고 3D voxel에 집계 → 변화 객체 마스크와 전후 ID | 실제 코드. SAM/Pi3/DINOv3 구성 모델 가중치 이용. 별도 변화 탐지기 재학습 없이 동작 | 현재 비교 실험과 직접 연결된다. native ERP는 아니며 전체 영상 길이·표본 프레임 수에 따른 메모리 비용 확인 필요. |
| **Argos / Argos-SLAM**, 2026-10 preprint | 두 방문의 RGB 시퀀스와 GFM 특징으로 joint change/reconstruction, SLAM에 온라인 통합 | 논문/저자 프로젝트의 Code·Datasets가 아직 링크 없는 표시. 공식 사용 가능 코드·학습 체크포인트를 확인 못함 | 관련도 높은 최신 연구지만 현재 실행 가능 모델로 제시하지 않는다. |

### VSCD: 프레임 대응을 모델 안에서 처리하지만 변화 head 가중치가 빠져 있음

[논문](https://arxiv.org/abs/2605.20821), [공식 코드](https://github.com/AutoCompSysLab/VSCD). 서로 다른 카메라 동선과 비동기 RGB 영상에서 query 프레임별 객체 변화 마스크를 낸다. 이동은 원위치의 사라짐과 새 위치의 나타남으로 정의하며, 같은 물체의 정밀 이동 벡터를 별도로 복원하는 모델은 아니다.

공식 README에는 SAM ViT-B 준비 후 80 epoch 학습하고 로컬 `ckpt_best.pt`로 평가하는 절차가 있다. **학습 후 저장되는 파일 이름을 공개 다운로드 링크로 세면 안 된다.** 재귀 트리에 `.pt/.pth/.ckpt/.safetensors`가 없고 releases도 비어 있었다. 저자 관련 Hugging Face 계정 조회에서도 공개 모델을 찾지 못했다. [README의 train/eval 계약](https://github.com/AutoCompSysLab/VSCD#training)

기본 스크립트는 고정 프레임 수 32를 사용한다. 따라서 논문의 영상 비교를 임의 길이 두 MP4 전체에 무제한 end-to-end 처리한다고 해석하지 않는다. 공개 checkpoint가 없으면 SAM 가중치만으로 학습 완료 VSCDNet을 실행할 수 없다.

### O-SCD: 실제 온라인 다중 시점 누적 코드와 필요한 가중치

[공식 프로젝트](https://chumsy0725.github.io/O-SCD/), [실행 코드/README](https://github.com/Chumsy0725/O-SCD/tree/main/O-SCD), [CVPR 논문](https://openaccess.thecvf.com/content/CVPR2026/papers/Galappaththige_Changes_in_Real_Time_Online_Scene_Change_Detection_with_Multi-View_CVPR_2026_paper.pdf).

실제 `oscd.py`에서 다음을 확인했다. [소스](https://github.com/Chumsy0725/O-SCD/blob/main/O-SCD/oscd.py)

- `Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny')`로 공개 학습 모델을 로드한다.
- `reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply`에서 **이미 학습된 첫 회차 Gaussian 지도**를 읽는다. 일반 메쉬 PLY와 같은 의미가 아니다.
- reference keyframe과 incoming frame을 다루며 변화용 Gaussian 표현을 학습/갱신한다.
- `run_oscd.sh`가 장면별 실행과 선택적 offline refinement를 제공한다.

즉 ‘공개 학습 완료 변화 모델 하나를 내려받아 바로 두 영상을 넣는’ 방식은 아니다. 사전학습 SAM 특징은 재사용하지만 새 공간에는 reference 3DGS 준비가 필요하고 변화장 최적화도 수행한다. 기존 메쉬·카메라 포즈가 있으면 위치 추정을 보완할 수 있으나, Gaussian map을 자동으로 대체하지는 못한다. [공개 데이터·준비 절차](https://github.com/Chumsy0725/O-SCD/blob/main/O-SCD/README.md)

### 그 밖의 직접 근거

- [O-SCD++ 저자 페이지](https://chumsy0725.github.io/O-SCD/plusplus/)와 [저장소](https://github.com/Chumsy0725/O-SCD)에서 코드 미공개 상태를 확인했다.
- [MV3DCD 공식 코드](https://github.com/Chumsy0725/MV3DCD)는 `train.py`, `train_masks.py`, `run.sh`를 제공한다. ‘Zero Labels’는 장면별 최적화까지 없다는 뜻이 아니다.
- [SceneDiff 공식 코드](https://github.com/yuqunw/scene_diff)는 두 영상 demo와 구성 모델 가중치 안내를 제공한다. 메쉬 입력과 동영상 시간 동기화를 자동으로 전제해서는 안 된다.
- [Argos 저자 프로젝트](https://www.multyxu.com/argos/)는 unposed RGB visits와 co-visibility-aware comparator를 설명하지만, Code 표시에 실제 배포 링크가 없었다. [논문](https://arxiv.org/abs/2610.10181)

## 3. 현재 영상·메쉬가 있을 때의 우선순위

1. 지금 실행하는 ViewDelta·SimSaC·SceneDiff의 동일 위치·방향별 결과를 비교해 시각 변화 후보가 기하 누락을 보완하는지 먼저 확인한다.
2. 원본 CYWS-3D 체크포인트를 실제 로드할 수 있으면, **원본 DINOv1·attention 없음·변화 head 전체 로드**를 입증한 뒤 ERP 기하 어댑터로 별도 비교한다. 원본 demo 대조를 함께 보관한다.
3. 전체 영상 수준의 누적 시스템을 확장한다면 공개 실행 코드가 있는 **O-SCD**가 구체적이다. 다만 기존 메쉬 경로에 reference 3DGS 준비를 추가하는 비용이 있다.
4. **VSCDNet**은 원하는 video-to-video 과제가 가장 직접적이지만, 공개 학습 완료 가중치가 확보되기 전에는 즉시 실행 후보로 약속하지 않는다.

Perspective 모델에는 ERP를 겹치는 tangent/cubemap 뷰로 변환하고, 각 방향의 virtual camera와 기존 mesh pose를 함께 전달하는 방식이 현실적인 어댑터 후보다. 영상 회차 사이의 장소 대응, 방향별 중복 객체의 통합, 가림/미관측 보류는 계속 필요하다. 두 영상의 같은 프레임 번호를 같은 장소로 가정하거나 검출 실패를 객체 제거로 바꾸면 안 된다. 이 적용 방향은 제안이며 현재 데이터에서 검증된 성능 결론은 아니다.
