# 메쉬를 선행 조건으로 두지 않는 변화 탐지: RGB-D·볼륨 표현 조사

확인일: 2026-10-15. 연구 자료만 읽었으며 코드 실행·설치·데이터 다운로드·뷰어 변경은 하지 않았다.

## 조사 범위 수정

목표는 앞으로 자체 촬영한 영상·깊이와 불완전한 자동 분할 결과에서 **이동·추가·소실과 판단을 유보할 후보를 구분**하는 기존 기술을 찾는 것이다. Rescan은 현재 시연 자료일 뿐이다. Rescan의 정답 인스턴스 ID나 현재 장면에 대한 적합성은 선정 기준에서 제외한다. 완성된 인스턴스 메쉬가 반드시 먼저 있어야 한다는 전제도 두지 않는다.

아래의 우선순위는 논문·공개 소스를 읽은 조사자의 판단이며, 자체 데이터에서 측정한 성능 순위가 아니다.

**사용자 재지적에 따른 선정 기준 정정:** 핵심 요구는 노이즈 때문에 인스턴스 분할·대응이 잘 안 되는 곳에서도 공간 단위 변화 후보를 찾는 것이다. RGB-D를 받거나 정답 분할이 필요 없다는 사실만으로 이 조건을 만족하지 않는다. 아래 Where Did I Leave My Glasses?의 우선 추천은 철회한다. 객체 기반 재관찰 우선순위 구현을, 객체 생성에 실패한 구역까지 탐지하는 구현으로 잘못 확대했다.

공개 `object_detector.py`는 검출 상자에 SAM을 적용하고 마스크가 없으면 빈 검출을 반환한다. `exploration_priority_map.py`의 입력은 `Iterable[SceneObject]`이고 객체별 stationarity를 점수로 바꾼다. 따라서 처음부터 객체로 생성되지 않은 노이즈 구역을 독립적으로 찾아 점수화하는 경로로 볼 수 없다. 과거에 검출된 객체가 현재 미검출인 경우를 관리하는 기능과, 전후 모두 분할되지 않은 구역에서 새 변화 후보를 만드는 기능을 구분한다. [검출 코드](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/core/object_detector.py), [우선순위 코드](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/grid_maps/exploration_priority_map.py)

## 확정된 미래 입력: RGB-D 영상과 카메라 위치 정보

사용자가 RGB-D 영상과 카메라 위치 정보를 확보하는 조건으로 확정했다. 여기서 기존 방법이 요구하는 **camera pose는 위치 xyz에 회전까지 포함한 자세**다. 입력이 xyz뿐이면 방향 정보가 추가로 필요하다. OASIS-Map은 명시적으로 SE(3) 자세를 요구하며, 공개 `perceive_semantix_release` 입력 타입도 4×4 pose와 3×3 intrinsics를 받는다. [OASIS-Map §III-A](https://arxiv.org/html/2607.14899v1#S3), [실제 입력 타입](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/core/input_types.py)

| 구분 | 확인해야 하는 입력 의미 |
|---|---|
| 자세 | 프레임별 위치와 회전, camera-to-world/world-to-camera 방향, 대응 시간 |
| 투영·깊이 | 초점거리·주점 등 intrinsics, RGB와 depth가 같은 픽셀에 정합되었는지, 미정합이면 센서 간 변환, 깊이 단위·무효값 규약 |
| 전후 좌표계 | 두 촬영의 자세가 같은 미터 단위 세계 좌표계인지. 촬영마다 SLAM 원점이 다르면 세션 간 정합이 별도로 필요 |

이것은 새 구현 제안이 아니라 기존 재투영·지도 갱신 방법의 입력 조건이다. RGB-D라는 이름이나 위치 파일의 존재만으로 조건 충족을 가정하지 않는다. 실제 공개 입력은 RGB·depth·intrinsics·4×4 pose·time을 함께 받으며, OASIS는 세션 공통 좌표계 정합을 외부 SLAM에 맡긴다. [공개 raw 입력 인터페이스](https://github.com/learnsyslab/perceive_semantix_release#readme), [OASIS-Map 입력 정의](https://arxiv.org/html/2607.14899v1#S3)

## 후보 비교

| 기존 방법 | 실제 요구 입력·표현 | 정답 인스턴스 사전 필요 여부 | 변화·불확실성 출력 | 공개 구현 확인 및 판단 |
|---|---|---|---|---|
| **Where Did I Leave My Glasses? (RA-L 2026)** | posed RGB-D, intrinsics·시간. 객체 점군 지도와 missing-object library; prior map 또는 빈 지도에서 시작 | GT는 불필요하지만 예측 검출·분할로 객체 생성 필요 | 생성된 객체의 stationarity와 재관측 priority map | 실제 코드 공개. **분할 실패 구역의 독립적 변화 후보 검출기로는 우선 추천 철회.** 객체 생성 이후의 관리 참고용. [논문](https://arxiv.org/html/2509.19851v2), [공식 코드](https://github.com/learnsyslab/perceive_semantix_release) |
| **OASIS-Map (2026 preprint)** | RGB 키프레임, 깊이, 두 세션의 공통 미터 좌표계 카메라 자세. TSDF 또는 누적 점군으로 기하 지도 구성 | 필요 없음. SAM2 마스크와 DINOv3 특징으로 자체 객체 지도 구성 | static / moved / appeared / disappeared / unknown | 공식 페이지는 **Code Soon**, 논문은 공개되어 있고 under review. 기능 목표가 가장 직접적으로 겹치지만 재현 가능한 공개 코드 후보로 확정할 수 없음. [논문 §III](https://arxiv.org/html/2607.14899v1#S3), [공식 프로젝트](https://ori-drs.github.io/projects/oasis-map/) |
| **Panoptic Multi-TSDFs (ICRA 2022)** | RGB-D, 외부 카메라 자세, 예측 panoptic 분할. 객체·배경·자유공간을 여러 TSDF submap으로 저장 | 필요 없음. 공개 코드에 Detectron 예측 입력 경로가 실제 존재 | persistent / absent / unobserved 등의 지도 상태. **동일 물체의 먼 위치 이동 대응은 별도 문제** | 저자 공개 코드 존재. 자동 분할과 미관측 상태를 포함한 볼륨 지도 기준으로 유력하나 이동·추가·소실을 모두 연결하는 완결된 탐지기로 보아서는 안 됨. [논문 §III](https://arxiv.org/html/2109.10165v2#S3), [저자 코드](https://github.com/ethz-asl/panoptic_mapping) |
| **Langer et al. (2022), Where Does It Belong?** | RGB-D로 만든 국소 지지면 점군 재구성, 참조 객체 지도. 배포 탐지 코드는 재구성 이후부터 | GT instance 불필요. 지지면 제거·군집화·PPF 매칭 | static / moved / removed / novel; 명시적 unknown 없음 | 실제 공개 코드 존재. 안정된 탁자·선반 위 물체 변화에는 관련되나 큰 가구·지지면 이동은 참조 재생성 조건. [논문 §3](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full), [저자 코드](https://github.com/edith-langer/tidyupvisionpipeline) |
| **Objects Can Move (ECCV 2022)** | 정합된 두 3D 모델, 자세·내부 파라미터. **배포 코드는 메쉬에서 깊이를 렌더링** | 변화 탐지에는 필요 없음. 특징 대응·강체 변환·supervoxel graph로 객체를 발견 | changing points와 연결된 객체 성분. 공개 결과 파일을 세 가지 사건 유형·unknown 출력으로 해석하면 안 됨 | 저자 핵심 코드 존재. 사전 인스턴스 분할 없이 객체를 찾는 비교 후보지만, 공개 파이프라인 자체가 메쉬 없는 입력은 아님. [논문](https://arxiv.org/abs/2208.09870), [저자 README](https://github.com/katadam/ObjectsCanMove) |

## 1. OASIS-Map: 가려짐·불완전 분할과 판단 유보를 직접 다룸

논문 §III-B는 RGB에 SAM2를 적용한 뒤 깊이·자세로 마스크를 3D에 올리고, 겹침과 의미 유사도에 따라 여러 관측을 합친다. 두 세션은 **별도 SLAM으로 같은 미터 좌표계에 정합되어 있어야 한다**. 깊이는 센서·LiDAR 투영·학습 추정 중 하나일 수 있지만, RGB 동영상만 주면 자세와 스케일까지 해결하는 방법은 아니다. [논문 §III-A/B](https://arxiv.org/html/2607.14899v1#S3)

객체 전체 특징 하나 대신 DINOv3 패치 대응을 모은다. §III-C 식 (6)의 평균 대응 점수와 대응 비율을 함께 사용하므로, 소수 패치만 강하게 맞는 부분 가림을 충분한 객체 대응과 구분한다. 식 (9)는 상대 세션에서 관측된 부피 비율이 0.5 미만이면 unmatched 객체를 바로 추가·소실로 단정하지 않고 **unknown**으로 남긴다. 이 점수는 특징 유사도이며 보정된 변화 확률이 아니다. [논문 §III-C, 식 (6)–(9)](https://arxiv.org/html/2607.14899v1#S3.SS3)

실내 결과에서 Table I의 **점 단위 moved F1은 0.353**, Table III의 **객체 대응 moved F1은 0.667**이다. 서로 다른 과제의 수치이므로 섞어 쓰면 안 된다. 자동 분할 오류를 전부 고친다는 증거도 아니다. 공식 페이지의 “Code Soon”과 “Under review” 표기를 확인했으며, 공개 preprint는 이미 존재한다. [논문 §IV, Tables I·III](https://arxiv.org/html/2607.14899v1#S4), [공식 상태](https://ori-drs.github.io/projects/oasis-map/)

**해석:** unknown은 관측 부족을 유보하는 데 직접 유용하다. 다만 이를 검토 우선순위가 매겨진 ‘유심관찰구역’ 알고리즘이나 검증된 확률로 바꿔 부를 근거는 아직 없다. 실내 임의 물체·임의 촬영에서의 범용 해법으로 확정하지 않는다.

## 2. Panoptic Multi-TSDFs: 메쉬보다 관측·자유공간을 유지하는 공개 기반

입력 분할은 연구자가 만든 완성 인스턴스가 아니라 예측 마스크여도 된다. 논문은 GT와 Detectron2 예측을 구분해 평가한다. TSDF submap에 기하와 분할 증거를 누적하며, 최근 관측 지도와 과거 객체 표면을 비교한다. 표면 일치, 관측된 자유공간과의 충돌, **아직 보지 못한 곳**을 구별하여 persistent / absent / unobserved를 갱신한다. 여러 관측과 가중치를 이용하므로 단일 메쉬 표면 차이만 보는 것보다 관측 근거를 보존한다. [논문 §III-A/C/D 및 §IV-B](https://arxiv.org/html/2109.10165v2)

저자 저장소의 [rio_detectron.yaml](https://github.com/ethz-asl/panoptic_mapping/blob/main/panoptic_mapping_ros/config/mapper/rio_detectron.yaml)은 `id_tracker: type: detectron`을 사용한다. [common.h](https://github.com/ethz-asl/panoptic_mapping/blob/main/panoptic_mapping/include/panoptic_mapping/common/common.h)의 실제 변화 상태는 `New`, `Matched`, `Unobserved`, `Absent`, `Persistent`다. **Moved 상태는 없다.** 원래 자리의 absent와 다른 자리의 new만으로 동일 물체 이동을 식별했다고 주장할 수 없다.

실험은 합성 flat과 실제 실내 반복 촬영에서 지도 재구성·유지 성능을 다룬다. 실제 예측 분할을 사용한 실험은 불완전한 입력을 다룬 근거지만, 모든 변화 유형의 객체 단위 precision/recall을 완결적으로 검증한 결과와는 다르다. [논문 §IV](https://arxiv.org/html/2109.10165v2#S4)

저자 코드는 실제 공개되어 있고 ROS 기반 mapper·설정·예제가 있다. README의 분류 confidence는 **누적 의미 분류의 확신**이며 변화 확률이 아니다. 개발·테스트 환경은 Ubuntu 18.04/20.04와 ROS이며, 현재 Windows 뷰어에 즉시 연결 가능하다는 뜻은 아니다. 빌드와 실행은 확인하지 않았다. [저자 README](https://github.com/ethz-asl/panoptic_mapping#readme)

## 3. Objects Can Move: 미리 분할된 객체 없이 강체 변환으로 객체 발견

사전 객체 마스크 대신 깊이 차이로 변화 seed를 얻고, 점 특징 대응에서 공통 강체 변환을 찾은 뒤 graph 최적화로 같은 움직임을 하는 부분을 묶는다. 추가·소실과 같이 대응 변환이 없는 표면도 변화 증거로 다루지만, 공개 최종 산출물은 `final_changing_points.npy`와 발견된 객체 성분이다. 사건 유형별 완제품 출력과 구분해야 한다. [저자 README](https://github.com/katadam/ObjectsCanMove), [논문 §§3.1–3.3](https://arxiv.org/abs/2208.09870)

README의 `create_GTchanges.py`는 **평가 정답 생성** 단계다. 이를 탐지의 GT 인스턴스 의존성으로 오해하면 안 된다. 반면 정합된 `.obj`와 깊이 렌더링을 요구하는 전처리는 실제 탐지 경로다. [initialDetection.py](https://github.com/katadam/ObjectsCanMove/blob/main/initialDetection.py)는 두 깊이 중 하나가 없는 픽셀을 제외한다. 이는 없는 관측을 곧바로 변화로 쓰지 않는 처리지만 모든 가림을 해결하는 보장은 아니다.

공개 예제는 사전학습 DGCNN 특징을 이용하며 다른 특징으로 교체할 수 있다. ‘객체 발견이 비지도’라는 표현을 학습 모델이 전혀 없다는 의미로 사용하지 않는다. README는 실험용 파라미터가 하드코딩되어 있고 RANSAC 무작위성도 남는다고 설명한다. 현재 사용자 데이터용 범용 실행기까지 확인된 것은 아니다. [저자 README의 특징·재현 설명](https://github.com/katadam/ObjectsCanMove#readme)

## 4. Where Did I Leave My Glasses?: 점수와 재관측 구역을 갖춘 공개 후보

RA-L 2026 논문의 공식 저장소는 논문의 객체 검출·scene belief·priority map 부분을 실제 공개한다. raw 파일과 ROS 입력 모두 있으며, 기존 지도 초기화도 가능하다. SAM·CLIP 후보를 기하·의미 정보로 대응하고 필요할 때 ICP를 적용해 객체 이동·소실·재등장에 따라 지도를 갱신한다. GT 인스턴스 메쉬는 필요 없다. [공식 README와 raw 인터페이스](https://github.com/learnsyslab/perceive_semantix_release), [논문 §III·IV-A/B](https://arxiv.org/html/2509.19851v2)

중요한 구분은 stationarity가 **마지막 위치에 계속 있을 것이라는 믿음**이라는 점이다. 미관측 객체에도 의미 prior에 따른 synthetic change를 주입해 점수를 감쇠시킨다. 따라서 낮은 값이 관측된 기하 변화의 확률을 뜻하지 않는다. maintenance priority는 재관측할 위치를 정하는 설계이며, 논문의 Beta 파라미터(5,6)와 현 공개 코드 기본값(4,2.8)도 다르다. 논문은 2D 바닥 탐색 지도에서 평가했고 코드는 3D 경로도 포함하지만, 3D 검토 구역 성능까지 검증되었다고 확대하지 않는다. [논문 §IV-B2/C](https://arxiv.org/html/2509.19851v2#S4), [실제 priority 구현](https://github.com/learnsyslab/perceive_semantix_release/blob/main/perceive_semantix_lib/src/perceive_semantix_lib/grid_maps/exploration_priority_map.py)

초록의 평균 95%는 시간 제한을 둔 **능동 지도 유지** 결과다. 이동은 추가와 소실 두 건으로 세므로, 세 변화 유형을 구분한 95% 정확도로 인용하지 않는다. 합성 Khronos 데이터의 별도 4D Changes F1은 apartment 89.3, office 53.0이며 실제 자체 촬영 결과가 아니다. 입력·공개 코드·재관측 점수의 조합만 보고 우선 후보로 선정한 판단은 철회했다. 분할 실패 구역의 독립 탐지가 확인되지 않았기 때문이다. [논문 §V-A2/Table II, §V-B3/Table V](https://arxiv.org/html/2509.19851v2#S5)

## 5. Langer 2022: 공개 사건 분류 코드가 있는 제한된 대안

저자 README는 ElasticFusion의 국소 표면 재구성을 입력으로 지지면을 제거하고 군집·PPF 대응을 수행한다고 명시한다. 즉 RGB-D 영상·pose를 그대로 받는 완성 실행 경로와는 구분한다. 논문은 카메라 추적 자세와 로봇 odometry 자세를 결합해 재구성을 보완하고, 부분 매칭·region growing으로 일부 합쳐진 객체를 분리한다. [공개 README](https://github.com/edith-langer/tidyupvisionpipeline/blob/365812063d1753c4bcb344512c759edd6e818f13/README.md), [논문 §§3.4–3.5](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full)

그러나 주요 지지면·가구가 크게 바뀌면 참조 지도를 다시 만드는 범위이고, 실험용 ObChange 국소 재구성에는 시각적 확인과 필요한 수동 보정도 들어갔다. 따라서 원시 RGB-D의 임의 노이즈·분할 실패에 대한 완전 자동 보증으로 쓰면 안 된다. 네 가지 사건의 공개 구현을 읽을 수 있는 대안이며, 임의 실내 전체의 유보 후보까지 제공하는 대안은 아니다. [논문 §§3.2, 4.1.1, 4.2 discussion](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.828732/full)

## RGB-D 확정 후 SceneDiff 우선순위 수정

**SceneDiff를 RGB-D 조건에서도 계속 1순위로 유지할 근거는 확인되지 않았다.** 논문 §4.1과 현재 공개 `SceneDiff.process_scene` 경로는 RGB들에서 Pi3로 depth·pose·intrinsics를 다시 추정한다. 확인한 demo와 기본 경로에는 실측 depth·외부 pose를 입력하는 인터페이스가 없다. 이를 받도록 바꾸는 일은 새로운 수정·검증 과제이므로 현재 조사에서 가능한 기능으로 간주하지 않는다. [논문 §4.1](https://arxiv.org/html/2512.16908v1#S4.SS1), [실제 추정 호출](https://github.com/yuqunw/scene_diff/blob/main/modules/scenediff.py), [demo 입력](https://github.com/yuqunw/scene_diff/blob/main/scripts/demo.py)

SceneDiff는 공개 RGB 기반 이동·추가·소실 비교 후보로 남긴다. 그러나 현재 핵심 선별 조건은 **객체 검출·분할이 실패한 공간에서도 변화 증거를 남기는가**다. Where Did I Leave My Glasses?의 priority, OASIS-Map의 unknown, Panoptic의 TSDF 상태 중 어느 것도 이름만으로 그 조건 충족을 단정하지 않는다. 분할 이전의 점·복셀·깊이 변화 및 노이즈·가림 처리를 확인하는 방향으로 조사 기준을 수정한다. 지지면 위 물체가 주 대상일 때 Langer 2022를 별도 비교 후보로 유지한다.

## 이번 조사에서 확정할 수 있는 내용

- **완성된 인스턴스 메쉬가 없어도 시작하는 기존 방법이 있다.** OASIS-Map과 Panoptic Multi-TSDFs는 RGB-D 관측과 자동 분할을 누적하여 지도를 만든다. 다만 깊이와 정합된 자세라는 입력 조건은 남는다.
- **불확실한 곳을 변화와 분리하는 선행 개념이 있다.** OASIS-Map의 unknown과 Panoptic Multi-TSDFs의 unobserved가 해당한다. 두 개념 모두 ‘변화 가능성 순위가 검증된 검토 구역’과 동일하지 않다.
- **입력 적합성·분할 실패에 대한 대응·코드 공개 여부를 구분한다.** Where Did I Leave My Glasses?는 실제 공개 코드가 있지만 분할되지 않은 구역을 독립적으로 탐지하는 방법으로 추천하지 않는다. OASIS-Map은 객체 지도와 자동 마스크를 사용하므로 분할 실패 전체를 해결했다고 단정할 수 없다. Panoptic은 이동 동일성 대응이 부족하고 Langer는 지지면 조건이 있다. 어느 것도 자체 촬영에서 실행하거나 우승 방법으로 검증한 것은 아니다.
- 위 내용을 근거로 새 점수식이나 구현을 제안·적용하지 않았다. RGB/영상 계열 후보와의 종합은 [상위 조사 노트](change-detection-without-mesh.md)에서 다룬다.
