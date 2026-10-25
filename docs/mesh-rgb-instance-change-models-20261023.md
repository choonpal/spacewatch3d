# 인스턴스 메쉬와 RGB를 함께 쓰는 객체 변화 탐지 후보

확인일: 2026-10-23 (Asia/Seoul). 요구 조건은 **이미 인스턴스별로 나뉜 전후 메쉬(표면 점군으로 변환 가능) + RGB 이미지**를 함께 사용하고, **사용 가능한 공개 코드와 사전학습 가중치**가 있는 방법이다. 공식 논문·저장소·입력 함수·가중치 로딩 코드를 조사했으며 설치·추론은 실행하지 않았다.

## 판단

이번에 확인한 직접 변화 탐지 후보 중 **위 입력을 그대로 받는 완제품은 확인하지 못했다**. 존재하지 않는다는 문헌 전체의 부재 증명은 아니다. 가장 가까운 공개 구현은 **SceneDiff**지만, 공식 경로는 RGB 영상 두 개에서 기하와 인스턴스를 새로 추정하므로 기존 메쉬·인스턴스를 넣으려면 입력 어댑터 작업이 필요하다. 객체 매칭 모델을 가져오는 경우에도 변화 라벨과 가시성 판정은 별도 모듈이다.

## 1. SceneDiff: 변화 탐지 코드와 구성 모델 가중치는 공개, 기존 메쉬 입력은 수정 필요

- 논문/프로젝트: [SceneDiff, ECCV 2026](https://yuqunw.github.io/SceneDiff/), [논문](https://arxiv.org/abs/2512.16908).
- 코드: [yuqunw/scene_diff](https://github.com/yuqunw/scene_diff). 장면별 변화 탐지기 재학습 없이 사전학습 구성 모델을 조합하는 방식이다.
- 공식 실행 입력: 두 RGB 영상 또는 영상에서 추출한 프레임 목록. `scripts/demo.py` → `SceneDiff.process_scene(file_list, img_1_length, img_2_length, output_dir)`로 들어간다. 이 함수는 Pi3로 깊이·포즈·point map을 추정하고 SAM 마스크를 생성한다. 기존 메쉬·3D 인스턴스를 넘기는 인자는 없다. [실제 파이프라인](https://github.com/yuqunw/scene_diff/blob/main/modules/scenediff.py), [영상 입력 스크립트](https://github.com/yuqunw/scene_diff/blob/main/scripts/demo.py)
- 변화 판정: 기하 비용, DINOv3 외형 비용, 영역 대응 비용을 결합하고 여러 뷰·3D voxel에 집계한다. 객체 유사도에 따라 전후 변화 객체 ID를 연결한다. [비용·임계값 설정](https://github.com/yuqunw/scene_diff/blob/main/configs/scenediff_config.yml)
- 출력: 단순 객체 대응만이 아니라 **변화 객체 마스크와 점군 시각화**다. `object_masks.pkl`에 전후 영상별 객체 ID·프레임 마스크·점수를 기록한다. 두 회차에 연결된 ID는 이동 객체의 전후 영역, 한쪽에만 있는 ID는 추가/제거 후보로 평가한다. 모든 객체에 `static/moved/added/removed/unknown`을 붙인 3D 상태표나 객체별 정밀 6DoF 변위가 기본 출력인 것은 아니다. [공식 출력 규격·평가](https://github.com/yuqunw/scene_diff#evaluation)

가중치 로딩은 실제 코드에서 다음을 확인했다.

| 구성 모델 | 공개 가중치 경로/방법 | 확인한 코드 |
|---|---|---|
| SAM ViT-H | 공식 `sam_vit_h_4b8939.pth` 다운로드 스크립트 | [download_sam_checkpoint.sh](https://github.com/yuqunw/scene_diff/blob/main/checkpoints/download_sam_checkpoint.sh) |
| Pi3 | `Pi3.from_pretrained('yyfz233/Pi3')`; 설정 파일의 모델 이름 | [geometry_model.py](https://github.com/yuqunw/scene_diff/blob/main/modules/geometry_model.py), [Pi3 가중치](https://huggingface.co/yyfz233/Pi3) |
| DINOv3 ViT-H+ | Meta에 가중치 접근 신청 후 URL을 `weight_url`에 입력. 기본 설정은 `TO_BE_FILLED` | [공식 설치·가중치 안내](https://github.com/yuqunw/scene_diff#download-checkpoints), [semantic_model.py](https://github.com/yuqunw/scene_diff/blob/main/modules/semantic_model.py) |

**현재 입력에 맞추려면 필요한 변경 — 구현 제안, 공식 지원 기능 아님:**

1. Pi3 추정 결과 대신 기존 메쉬를 **알려진 카메라 포즈·내부 파라미터**로 raycast해 depth/point map을 만든다. 정합된 메쉬 좌표와 카메라 좌표의 일관성을 유지한다.
2. SAM 재분할 대신 기존 3D 인스턴스의 보이는 면을 각 이미지에 투영한 ID 마스크를 전달한다. 가림을 고려한 z-buffer가 필요하다.
3. DINOv3의 RGB 특징과 다중 뷰 비용·대응·집계 부분을 재사용한다. 기존 정합 불확실성을 별도 반영하고 변화 임계값을 검증한다.

이 변경은 **재학습 없는 어댑터 실험 후보**다. 기존 RGB-only 실행이 곧 메쉬+RGB 입력 지원을 뜻하지 않는다. Pi3 신뢰도와 정규화 스케일을 사용하는 코드도 바꿔야 하므로 함수 인자 하나를 교체하는 수준으로 가정하면 안 된다. 비슷한 의자 사이 대응과 부분 가림은 별도 평가해야 한다.

## 2. CrossOver: 기존 객체의 3D·RGB 대응에 재사용할 수 있는 공개 모델

**변화 탐지 완제품이 아닌 객체 대응 구성 요소**다. [CVPR 2025 공식 코드](https://github.com/GradientSpaces/CrossOver)는 객체 점군과 객체별 RGB crop을 같은 임베딩 공간으로 옮기는 인코더를 제공한다. `InstanceCrossOverModel.forward()`는 `objects.inputs.point`와 `objects.inputs.rgb`를 처리하고 모달리티별 임베딩을 반환한다. 현재 구현에 두 특징을 융합해 이동·추가·제거를 출력하는 분류기는 없다. [실제 모델 코드](https://github.com/GradientSpaces/CrossOver/blob/main/model/instance_crossover.py)

- **체크포인트 확인:** [3RScan 인스턴스 모델](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_crossover_scan3r.pth), [전체 모델 목록](https://huggingface.co/gradient-spaces/CrossOver/tree/main). Hugging Face 공개 API에서 gated=false와 실제 파일 목록을 확인했다. 이름이 `.pth`이지만 디렉터리이며 내부 `model.safetensors`가 가중치다. 3RScan 모델의 해당 파일은 78,488,128바이트다. [체크포인트 로더](https://github.com/GradientSpaces/CrossOver/blob/main/util/torch_util.py)
- **기존 분할 재사용 경로:** 객체 메쉬에서 점군을 표본화하고, 카메라와 객체 ID를 이용해 RGB crop을 만든다. 공식 3RScan 전처리도 3D 인스턴스를 2D에 투영한 `gt-projection-seg.npz`를 사용한다. 사용자 데이터에는 로더/투영 포맷 변환이 필요하다. [전처리 안내](https://github.com/GradientSpaces/CrossOver/blob/main/DATA.md), [객체 검색 데모](https://github.com/GradientSpaces/CrossOver/blob/main/demo/demo_instance_retrieval.py)
- **추가 구현:** 전후 모달리티별 유사도 결합, 미대응을 허용하는 객체 대응, 원래 좌표계에서의 위치·자세 비교, 가림과 부재 구분이 필요하다. 모달리티를 함께 지원한다는 사실과 변화 판정을 위한 학습된 융합 모델이 있다는 사실은 구별해야 한다. README의 temporal instance matching 평가 공개 항목도 TODO로 남아 있다.
- **설치 조건:** 공식 시험 환경은 Ubuntu 22.04, CUDA 12.1, RTX 3090/4090이다. 별도 I2P-MAE 인코더 가중치 경로가 코드에 하드코딩돼 있으며, [Point-Bind의 공식 사전학습 가중치 안내](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM/blob/main/Install.md)를 따라 준비하고 경로를 수정해야 한다. DINOv2/BLIP도 구성 모델로 사용한다. 설치와 추론 성공을 검증한 것은 아니다.

현재 객체 분할과 기하 판정기를 유지하면서 **학습된 RGB·3D 동일성 증거를 보강하려는 목적**에는 검토할 가치가 있다. 그 자체를 요청한 객체 변화 탐지 모델로 세지는 않는다.

## 3. RGB-D에서 인스턴스 점군을 만드는 공개 제거 탐지기: 조건 일부만 일치

[Matsubara & Saito, Point Cloud Change Detection in Indoor Environments, APMAR 2023](https://ceur-ws.org/Vol-3467/paper3.pdf)는 [공식 코드](https://github.com/Tomoya-Matsubara/point-cloud-change-detection)가 있다. RGB-D를 Detectron2로 분할한 뒤 3D에 투영·통합하고 객체 영역의 제거를 판정한다. 구성 분할기는 `COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml`이며 `model_zoo.get_checkpoint_url()`로 공개 사전학습 가중치를 읽는다. [분할 코드](https://github.com/Tomoya-Matsubara/point-cloud-change-detection/blob/main/src/segmentation/main.py)

그러나 변화 단계는 `point cloud + label`을 읽고 인스턴스 이름을 클래스 이름으로 바꿔 kNN 의미 라벨 일치를 검사한다. **RGB 외형 특징을 전후 객체 동일성 판정에 함께 사용하지 않는다.** 출력은 `REMOVE / NO_CHANGE`이며 이동·교체·회전을 모두 분류하지 않는다. 실행 `main()`에는 정답 로딩과 평가 시각화가 결합돼 있어 사용자 데이터용 순수 추론 진입점을 분리해야 한다. 기존 인스턴스 점군을 포맷에 맞춰 넣는 것은 가능해 보이지만, 요청한 복합 모델의 완전 일치 후보로 추천하지 않는다. [변화 탐지 실제 코드](https://github.com/Tomoya-Matsubara/point-cloud-change-detection/blob/main/src/change_detection.py), [입출력·실행 안내](https://github.com/Tomoya-Matsubara/point-cloud-change-detection/blob/main/src/README.md)

## 4. 이름은 가깝지만 엄격한 후보에서 제외한 방법

| 방법 | 확인된 사실 | 제외 이유 |
|---|---|---|
| [Objects Can Move, ECCV 2022](https://github.com/katadam/ObjectsCanMove) | 정합된 `.obj`, 깊이 렌더 차이, 학습 기하 특징 대응, 강체 일관성과 그래프 최적화로 변화 영역 발견. 코드와 외부 사전학습 기하 특징 추출 안내 공개 | RGB 외형을 함께 쓰는 변화 탐지 모델이 아니다. 인스턴스 정답 파일은 GT 생성/평가와 구별해야 한다. |
| [Has Anything Changed?](https://arxiv.org/abs/2312.01148) | 3D 변화 씨앗을 SAM 색상/깊이 마스크로 전파 | [저자 저장소](https://github.com/katadam/ObjectChangeDetection)는 확인 시점 README의 `Code available soon`뿐이다. |
| [OASIS-Map](https://ori-drs.github.io/projects/oasis-map/) | SAM2·DINOv3·깊이·포즈로 객체 대응과 변화 상태 추론 | 공식 페이지 `Code Soon`; 사용 가능한 전체 공개 구현을 확인하지 못했다. |
| [POCD](https://trailab.github.io/POCD/) | RGB-D 의미 점군과 객체 TSDF로 상태 확률 갱신 | [공식 저장소](https://github.com/TRAILab/POCD)는 README/index 및 데이터셋 연결만 확인했다. |
| [FOCUS, 2026-10](https://arxiv.org/html/2610.05639v1) | 기하와 3DGS 렌더 외형의 likelihood로 PERSISTED / REPLACED / REMOVED 추론 | 기존 메쉬를 쓰는 구조가 아니며, 논문·검색에서 사용 가능한 공식 코드/가중치 배포를 확인하지 못했다. |
| [FOUND-IT](https://github.com/MIT-SPARK/FOUND-IT) | foundation model 기반 3D scene graph 연구 | 확인한 공개 저장소는 LICENSE만 있다. 실행 가능한 구현으로 추천할 수 없다. |
| LivingScenes / MORE2, CYWS-3D, 3DGS-CD, MV3DCD | 각각 객체 점군 대응 / RGB 쌍 변화 탐지 / 영상 기반 3DGS 변화 탐지에 적합 | 각각 RGB 외형 입력 부재, 기존 메쉬 입력 부재, 장면 표현 교체 때문에 요청 조건의 완전 일치 후보로 세지 않았다. |
| [Comp-PointLLM, JCDE 2026](https://academic.oup.com/jcde/article/13/5/122/8658277) | 두 객체 점군의 PointBERT 특징과 렌더 이미지의 CLIP 특징을 결합해 차이를 설명 | 실내 재촬영의 이동·추가·제거 탐지 과제가 아니며, 이번 검색에서 공식 실행 코드와 가중치 배포를 확인하지 못했다. |

## 적용 우선순위

‘현재 인스턴스 메쉬와 RGB를 모두 유지하며 실험’이 목표라면, **SceneDiff의 변화 비용·객체 집계 부분에 메쉬 기반 depth/instance projection을 연결하는 경로**가 조사한 직접 탐지기 중 가장 구체적이다. 기존 판정기를 유지하며 객체 대응을 보강하려면 **CrossOver**가 공개 코드·가중치를 갖춘 구성 요소다. **현재 입력을 그대로 받는 공개 사전학습 완제품을 찾았다**고 말할 단계는 아니다. 두 경로 모두 사용자 데이터 연결과 가시성·변화 판정의 별도 검증이 필요하다.
