# 파노라마 영상의 객체 변화 탐지 모델 조사

확인일: 2026-10-24 (Asia/Seoul). 대상은 두 촬영 회차의 360도 실내 RGB 영상에서 객체 추가·제거·이동 후보를 찾는 방법이다. 공식 논문·저자/구현자 저장소·설정·재귀 파일 목록·가중치 배포 링크를 조사했다. 설치·추론과 현재 영상의 성능 검증은 하지 않았다.

## 결론

**파노라마를 비교하는 변화 탐지 연구와 실제 코드는 있다.** 특히 **CYWS-360**은 ERP 입력과 구면 기하를 명시한 구현이다. 다만 조사한 후보 중 **실사 실내 360×180도 ERP 영상 두 개를 받아 객체 변화까지 처리하는 공개 사전학습 완제품**은 확인하지 못했다. 파노라마 전용 코드의 존재와 바로 내려받는 학습 완료 가중치의 존재를 구별해야 한다.

또한 다음 입력은 서로 다르다.

- **전체 ERP**: 보통 2:1 비율이며 구면의 좌우 경계와 극지 왜곡을 다룬다.
- **잘라낸 파노라마/수평 strip**: 원본은 ERP여도 상하를 제거하거나 부분 시야만 사용한다.
- **여러 perspective 사진의 stitching**: 폭과 높이가 다양할 수 있으며 전체 ERP 투영이 보장되지 않는다.
- **360도 주변에서 찍은 다중 시점 사진**: NeRF/3DGS의 360 scene이라는 표현은 360도 카메라 ERP 입력을 뜻하지 않는다.

## 후보 비교

| 후보 | 실제 입력·출력 | 파노라마 대응 종류 | 공개 코드·학습 가중치 판정 |
|---|---|---|---|
| **CYWS-360** | 전후 파노라마 두 장 + 사전 계산 깊이·상대 포즈 → 변화 객체 bounding box | **명시적 ERP 기하 지원**. DINOv3 특징·3D feature warp·UNet·CenterNet. `panorama_mode=True` | 코드 공개. **파노라마에 학습 완료된 체크포인트는 확인 못함**. `cyws-3d.ckpt`는 원본 모델 warm-start 경로다. |
| **PanoSCU / EmbSCU+PanoSAM** | 실내 전후 panorama pair → 변화 이해·설명·재배치 지시 연구 | Ai2Thor의 여러 일반 시야 이미지를 stitching. 고정 2:1 전체 ERP라고 확인할 근거는 부족 | 저자 코드 존재. **PanoSAM/파노라마 변화 이해 학습 가중치는 확인 못함**. |
| **ViewDelta-Panoramic LoRA fork** | 전후 panoramic street-view pair + 텍스트 → 변화 마스크 | 넓은 해상도·토큰 격자에 적응. 구면 ERP 기하 지원 여부는 불명확 | 코드 공개. 다운로드 안내는 **원본 ViewDelta 가중치**이며 파노라마 LoRA 학습 가중치 링크는 못 찾음. |
| **ViewDelta 원본** | RGB 두 장 + 검출할 변화 텍스트 → 이진 마스크 | PSCD 등 여러 데이터셋으로 학습한 일반 모델. 기본 입력은 256×256 resize | **공식 코드와 실제 가중치 배포 확인**. 전체 ERP 전용 모델이나 영상 모델로 분류하지 않는다. |
| **SimSaC**, CVPR 2022 | 두 이미지 → dense correspondence + 변화 마스크 | PCD 파노라마 벤치마크 평가. native ERP 전용 구조는 아니다 | 공식 코드와 저자가 연결한 pretrained Drive 폴더 확인. 현재 실내 ERP 성능은 미검증. |

## 1. CYWS-360: 현재 파이프라인과 입력 구조가 가장 가까운 공개 구현

구현자 저장소 [sevdaimany/panorama-change-detection](https://github.com/sevdaimany/panorama-change-detection)의 기본 브랜치는 `training_final`이며 README의 모델명은 **CYWS-360**이다. 원본 CYWS-3D의 제3자 확장으로, Panoramax의 360도 ERP street-level imagery에 맞췄다고 명시한다. 학회에서 검증된 독립 모델의 공식 저자 배포판이라고 소개하면 안 된다.

README·실제 설정에서 확인한 것은 다음과 같다. [README](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/README.md), [config.yml](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/config.yml)

- 두 파노라마의 DINOv3 ViT-Base 다중 스케일 특징을 구한다.
- 깊이와 상대 카메라 포즈를 이용한 **equirectangular-to-3D back-projection**으로 특징을 전후 재투영한다.
- 특징 차이를 UNet과 CenterNet head에 넣어 변화 영역의 축 정렬 박스를 낸다.
- 기본 입력 크기는 `[320,768]`, `panorama_mode=True`다. 깊이 `.npz`와 상대 포즈 `.npz`는 별도 경로에서 읽는다.
- `load_weights_from: cyws-3d.ckpt`는 원본 모델로 시작하는 설정이다. 재귀 파일 목록과 releases에는 해당 모델 또는 파노라마 학습 완료 `.ckpt/.pth` 배포를 확인하지 못했다.

추가 소스 검토에서 `use_dinov3=True` 옆에는 기존 pretrained weights를 시험하려면 `False`로 바꾸라는 주석이 있다. 따라서 원본 CYWS-3D 체크포인트를 내려받는 것만으로 현재 DINOv3 파노라마 학습 구성을 재현할 수 있다고 보면 안 된다. 실제 registration 모듈은 `depth1/2`, `K_inv_1/2`, 양방향 `Rt`와 `panorama_mode`에 따른 구면 warp를 사용한다. [등록 모듈](https://github.com/sevdaimany/panorama-change-detection/blob/training_final/src/cyws360/registeration_module.py)

별도 [Panorama-Geometry-Estimation](https://github.com/sevdaimany/Panorama-Geometry-Estimation)은 ERP→겹치는 perspective face→MapAnything/Depth Anything 3 경로와 구면 DFRM을 제공한다. 출력은 포즈·깊이·가림 마스크다. **자체로 변화 객체를 탐지하는 모델은 아니다.**

**현재 프로젝트 적용 판단:** 이미 메쉬와 카메라 포즈가 있다면 새 기하 모델을 반드시 실행할 필요는 없다. 메쉬를 원본 시점으로 raycast해 깊이를 만들고 이 구현의 깊이·포즈 입력 형식으로 변환하는 어댑터를 검토할 수 있다. 다만 공개 파노라마 학습 가중치가 확인되지 않아, 그 자체로 ‘재학습 없이 바로 추론 가능’하다고 약속할 수 없다.

## 2. PanoSCU: 실내 객체 변화에는 직접적이지만, 데이터셋·분할기·변화 모델을 구별해야 함

[PanoSCU 공식 논문](https://ieeexplore.ieee.org/document/10965672/)은 IEEE Access 2025 연구다. [대학 공식 데이터 페이지](https://ro.ecu.edu.au/datasets/148/)는 **2,650개 재배치 시나리오/파노라마 쌍, 5,300장, 실내 48개 객체 클래스**를 설명한다. 여러 개의 카메라 관측을 stitching하므로 비율과 크기가 다양하다. 전후 변화의 위치·내용을 함께 다루지만, 현재 실촬영 ERP 동영상에 즉시 적용하는 모델이라는 뜻은 아니다.

저자 코드도 실제로 확인했다.

- [EmbSCU](https://github.com/mariiak2021/EmbSCU): `pano_code/`에 학습·평가와 변화 설명 구현이 있다. README는 `checkpoints/`를 가중치 폴더처럼 설명하지만, 조사한 실제 파일은 `*_res.json` 등 **평가 결과 JSON**이었다. 실행 가능한 학습 모델 `.pt/.pth/.ckpt`나 외부 배포 링크는 확인하지 못했다.
- [SAOMv1-SAOMv2-and-PanoSAM](https://github.com/mariiak2021/SAOMv1-SAOMv2-and-PanoSAM): 저자의 **PanoSAM**은 320×64 prompt point-grid로 SAM을 파노라마 분할에 적응한 모델이다. 학습·평가 코드는 공개돼 있다. 가중치 안내는 원본 SAM으로 연결하며, 학습 완료 PanoSAM 가중치를 찾지 못했다.

PanoSAM 자체는 한 영상의 **객체 분할기**다. 객체 변화 탐지에는 전후 비교·대응·변화 설명 단계가 추가로 필요하다. 같은 이름의 다른 개발자 SAM3 wrapper와도 구별해야 한다. 논문 PDF는 이 조사에서 대학 다운로드의 403 응답으로 직접 읽지 못했으며, ERP 전체 투영 여부에 관한 판단은 공식 데이터 설명과 공개 구현에 한정한다.

## 3. ViewDelta: 가중치가 공개된 비교군과 파노라마 LoRA 확장은 서로 다름

[ViewDelta 공식 코드](https://github.com/drags99/viewdelta-scd), [공개 가중치](https://huggingface.co/hoskerelab/ViewDelta)가 있다. 두 이미지와 텍스트 조건으로 변화 마스크를 출력하며 PSCD를 포함한 여러 데이터셋을 사용한다. 기본 경로는 256×256 resize다. 따라서 현재 전체 ERP를 그대로 넣고 얻은 출력을 ‘파노라마 전용 기하를 반영한 결과’라고 설명해서는 안 된다.

제3자 [LoRA-fine-tuning-of-ViewDelta](https://github.com/sevdaimany/LoRA-fine-tuning-of-ViewDelta)의 기본 브랜치는 **`panorama`**다. `main` 브랜치만 보면 원본 코드 설명이 나오므로 브랜치를 구별했다. [파노라마 설정](https://github.com/sevdaimany/LoRA-fine-tuning-of-ViewDelta/blob/panorama/ViewDelta/model/transformer_args.py)은 기본 입력 280×672, 선택 입력 560×1344, 출력 마스크 640×1536을 사용한다. README의 ‘640×1536 입력’ 설명과 실제 설정이 완전히 같지 않다.

해당 fork는 도시별 분할과 LoRA 학습 코드를 제공하지만 다운로드는 `hoskerelab/ViewDelta/viewdelta_checkpoint.pth`인 **원본 가중치**다. 재귀 트리·README·빈 releases에서 파노라마 LoRA 가중치를 확인하지 못했다. 사유 데이터의 투영 정의와 구면 경계 처리가 충분히 명시되지 않아 native 전체 ERP 모델로 단정하지 않는다.

## 4. 고전 파노라마 벤치마크 계열의 공개 상태

아래는 PCD/PSCD에 평가했다는 사실과 실제 공개 체크포인트를 구별한 결과다.

| 항목 | 확인 내용 |
|---|---|
| [PSCD](https://sakuradaken.net/pscd/) | 원본 4000×2000 ERP에서 상하를 crop한 도시 이미지 **770쌍**. 이진·의미·인스턴스 변화 주석이 있다. **데이터셋**이다. |
| [CSCDNet / SSCDNet](https://github.com/kensakurada/sscdnet) | 코드와 PSCD 연구 연결. README에 SSCDNet/PSCD 준비 중 문구가 남아 있고 ResNet18 사전학습 안내만 확인. **ResNet 가중치를 변화 탐지 학습 가중치로 세지 않았다.** |
| [DR-TANet](https://github.com/Herrccc/DR-TANet) | PCD 평가 코드 공개. README·빈 releases에서 학습 완료 모델 배포 링크를 확인하지 못했다. |
| [D-SSCD](https://github.com/NeurAI-Lab/D-SSCD) | 공식 README에 `Model Checkpoints` 제목이 있지만 링크가 비어 있다. 즉시 사용할 학습 가중치가 공개됐다고 세면 안 된다. |
| [SimSaC](https://github.com/SAMMiCA/SimSaC) | CVPR 2022 dense correspondence·mis-correspondence+change localization. PCD 평가 및 [pretrained 폴더](https://drive.google.com/drive/folders/119FRNCyrIXxxYrZi-_kbGrGFiQ1MSkAG) 연결 확인. 구면 ERP 전용은 아니다. |
| [CCL, Machine Learning 2023](https://doi.org/10.1007/s10994-022-06288-4) | circular-symmetric correlation layer의 원통형 회전·평행이동 대응 연구. TSUNAMI 이미지 roll 실험은 전체 360 영상 객체 변화 모델과 다르다. 실제 공개 코드·학습 가중치는 확인하지 못했다. |

가중치 검증 범위도 구별했다. 이 조사 담당자는 위의 파노라마 구현 저장소·설정·재귀 파일 목록을 직접 읽었다. 병행 검토자는 SimSaC의 공식 Drive 폴더를 연결 도구로 조회해 `synth_then_joint_synth_cmu.pth`와 `synth_then_joint_synth_changesim.pth`(각 72,790,909 bytes), `synthetic.pth`(157,092,099 bytes)의 실제 파일 존재를 확인했다. ViewDelta의 Hugging Face 파일 목록에서도 약 201MB의 `viewdelta_checkpoint.pth`가 확인됐다. **파일 메타데이터 확인이며 다운로드·로딩·추론 성공 검증은 아니다.** 이들 파일이 별도의 파노라마 전용 학습 가중치라는 근거는 없다.

## 실험 방향

현재 목적에서는 **CYWS-360의 ERP+깊이+포즈 입력 구조**가 가장 가깝고, **SimSaC/ViewDelta**는 공개 가중치로 먼저 작동 여부를 비교할 수 있는 후보다. 후자들은 perspective crop 또는 겹치는 방향별 뷰로 입력을 만들고 원본 ERP/메쉬 좌표로 후보를 되돌리는 어댑터 실험을 제안한다. 이 제안은 현재 데이터에서 성능이 입증된 방법은 아니다.

조사한 주요 방법의 중심 입력은 **이미지 쌍**이다. 파노라마 동영상 두 회차 전체를 비교하려면 장소별 키프레임 대응, 여러 방향의 관측, 프레임 사이 중복 객체 통합과 가림/미관측 보류가 추가로 필요하다. 이때 두 이미지에서 보이는 차이 후보와 동일 물리 객체의 추가·제거·이동 확정을 구별해야 한다.
