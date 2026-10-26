# 분할 성공에 의존하지 않는 RGB-D·점군 변화 후보: 1차 자료 조사

확인일: 2026-10-15. 목표는 **노이즈가 있는 전체 장면에서 객체 분할이 실패해도 변화 의심 공간을 찾고 수치화하는 기존 기술**을 확인하는 것이다. RGB-D와 카메라 자세를 사용할 수 있다. 연구·소스 열람만 수행했으며 설치·검출 실행·새 알고리즘 구현·데이터 변경은 하지 않았다.

여기서 ‘분할 불필요’는 semantic/instance 마스크나 객체 검출이 **변화 후보 생성의 필수 선행 조건이 아님**을 뜻한다. 후보를 찾은 뒤 잡음을 지우기 위한 기하 군집화까지 없다는 뜻은 아니다. Rescan의 정답 라벨·현 장면 적합성은 선정 기준으로 삼지 않는다.

## 결과 요약

| 기존 방법 | 분할 이전에 만드는 공간 증거 | 수치의 정확한 의미 | 가림·미관측 처리 | 공개 저자 코드 확인 |
|---|---|---|---|---|
| **Underwood et al., ICRA 2013** | 두 range scan의 점이 상대 scan의 관측된 자유공간을 침범하는지 검사 | 각도·거리 임계값에 따른 기하 판정. 배포 앱은 변화점·대응 ray 끝점을 내보내며 변화 확률은 아님 | 각도 이웃과 관측 coverage 조건으로 그림자·hole·탐색 경계의 겉보기 차이를 배제. 앱이 별도 unknown 지도를 내보내는 것은 아님 | **실제 `points-detect-change.cpp` 확인**. `acfr/snark`는 현재 archived이며 빌드·실행 미확인 |
| **Fehr et al., ICRA 2017** | 정합된 TSDF의 voxel 차이. 객체 분할은 이 증거를 얻은 뒤 수행 | 절대 TSDF 차이(m), voxel 관측 weight, 임계값에 따른 dynamic label. posterior 아님 | weight=0인 미관측과 관측된 자유공간을 구분 | 공식 저장소는 **데이터·README·preview**. detector 구현 재확인되지 않음 |
| **Wellhausen et al., SSRR 2017** | 현재 점과 reference occupancy voxel 사이 Mahalanobis 거리 | 센서 오차로 정규화한 D²; 이후 RF의 change-cluster scored label과 반복 관측 근거. 점수 보정 확률 검증은 아님 | raycast로 관측 여부 확인. 미관측은 distance=0으로 후보에서 제외 | 원문·저자 출판 목록 확인. detector 저자 코드는 미확인 |

출처: [Underwood 저자 프로젝트](https://www.acfr.usyd.edu.au/papers/icra13-underwood-changedetection.shtml), [Fehr 원문](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf), [Wellhausen 원문](https://cesarcadena.ethz.ch/files/SSRR2017_lwellhausen.pdf). 아래는 논문 근거와 현재 구현 확인 범위를 구분한 상세 기록이다.

## 1. Underwood: 객체가 무엇인지 몰라도 관측된 자유공간과 모순되는 점 탐지

**논문:** *Explicit 3D Change Detection using Ray-Tracing in Spherical Coordinates*, J. P. Underwood, D. Gillsjö, T. Bailey, V. Vlaskine, ICRA 2013. 입력은 정합된 range scan과 센서 자세다. 상대 센서 좌표계에서 각도 이웃의 range와 비교하며, 전후 순서를 바꾸어 addition/subtraction을 얻는다. 이동 물체의 두 위치를 동일 인스턴스로 연결하는 단계는 아니다. [저자 원문, §§III–IV](https://www.acfr.usyd.edu.au/papers/pdfs/Underwood-ICRA13-ChangeDetection.pdf)

관측 범위·hole과 가림을 검사하므로, 단순 nearest-neighbor 거리보다 ‘실제로 비어 있는 것으로 관측했던 곳’의 모순을 이용한다. 다만 자세·센서 오차를 확률적으로 전파하지 않고 각도/거리 tolerance로 다룬다. 움직인 센서에서 calibration·정합·경계 오차에 더 취약하다는 실험 결과도 있다. [원문 §VI](https://www-personal.acfr.usyd.edu.au/tbailey/papers/icra13underwood.pdf)

**실제 실내 잡음 결과:** cluttered lab에서 작은 상자를 네 위치에 두고 세 센서 자세에서 총 24 scans를 얻었다. Table II의 전체 pair 실험은 다음과 같다. 각 변형의 **최적 파라미터**를 사용한 pooled point-level 결과이며, 임의 RGB-D 촬영에서의 성능이 아니다.

| Table II 실험 | Precision | Recall | F-score |
|---|---:|---:|---:|
| Lab, pair 비교, 군집 후처리 없음 (line 2) | 0.7497 | 0.6739 | 0.7098 |
| Lab, pair 비교, 군집 후처리 포함 (line 6) | 0.9758 | 0.6955 | 0.8121 |

[원문 §V-B·VI 및 Table II](https://www.acfr.usyd.edu.au/papers/pdfs/Underwood-ICRA13-ChangeDetection.pdf). 유리의 시점 의존 반사는 GT에서 제외되었다. 표의 높은 accuracy는 대부분이 불변점이기 때문이므로 F-score와 구별해야 한다.

**소스 확인:** [공식 안내](https://www.acfr.usyd.edu.au/papers/icra13-underwood-changedetection.shtml)는 core 변화점 검사와 잡음 억제 clustering 예제를 구분한다. 실제 [points-detect-change.cpp](https://github.com/acfr/snark/blob/master/point_cloud/applications/points-detect-change.cpp)의 `trace()`는 각도 이웃·range threshold·bearing/elevation coverage를 검사하고, 성립한 점과 reference 끝점을 출력한다. 객체 마스크 입력이나 사전학습 모델은 없다. 따라서 **분할 실패와 독립적인 공간 후보를 찾는 공개 비교 후보**라는 판단은 가능하다. 다만 연속 변화확률 지도가 이미 제공된다고 주장하지 않는다.

[저자 저장소](https://github.com/acfr/snark)는 2026-10-15 확인 시 archived 상태다. 소스 존재를 확인했을 뿐, 현재 환경에서 설치·실행 가능하다는 뜻은 아니다. RGB-D 적용에는 프레임별 깊이로 얻은 ray와 센서 원점이 필요하다. 여러 시점에서 합친 전체 점군을 한 센서 원점의 단일 scan처럼 해석하면 안 된다는 것은 입력 모델로부터의 해석이다.

## 2. Fehr: TSDF 차이 자체는 선행 객체 분할과 독립적

**논문:** *TSDF-based Change Detection for Consistent Long-Term Dense Reconstruction and Dynamic Object Discovery*, Fehr et al., ICRA 2017. §III-B 식 (4)는 공통 관측 voxel에서 `δ = |f_new − f_static|`을 계산한다. 작은 관측 weight를 제외하고, dynamic label에 3D erosion/dilation을 적용한다. 객체의 mesh region growing·connected components는 **그 이후** 단계다. 즉 분할 결과가 없어도 거친 변화 공간 증거는 생긴다. [저자 원문, §III-A/B/C](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf)

TSDF weight=0은 미관측이고, 양의 truncation 값과 weight>0은 관측된 자유공간이다. 따라서 없는 표면점을 소실로 바로 간주하지 않는다. 다만 정확한 pose-graph 정합에 의존하며, 저자는 지역적·비강체 reconstruction misalignment가 단일 전역 ICP로 해결되지 않음을 인정한다. posterior·정합 covariance·점수 calibration을 제공하는 방법으로 해석하면 안 된다. [원문 §III-B4·V](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf)

**현실성의 범위:** 실제 Google Tango RGB-D 실내 자료이며 living-room 9회, office 4회, lounge 10회다. lounge는 약 50–100% 중첩과 부분 관측을 포함한다. 이는 현실적 잡음·불완전 관측 근거지만 통제된 pose/noise sweep은 아니다. Table II의 recall 98/100/92%는 **최종 객체 변화 개수** 평가이며 voxel 의심구역 precision/recall이 아니다. 후단의 평면 객체 제거 때문에 누락된 사례도 있어 그 수치를 거친 후보 지도의 성능으로 사용하지 않는다. [원문 §IV/Table II](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf), [공식 자료 설명](https://github.com/ethz-asl/change_detection_ds)

**코드 재확인:** [공식 `change_detection_ds` 최상위](https://github.com/ethz-asl/change_detection_ds)는 README와 previews뿐이다. 저자 [출판 목록](https://cesarcadena.ethz.ch/publications.html)은 논문 draft를 제공하지만 해당 detector 코드 링크는 확인되지 않았다. 원문에 등장하는 OpenChisel은 기반 재구성기와 관련된 것으로, Fehr의 차이 검사·static-map 갱신·후처리 전체 공개 구현을 의미하지 않는다. ‘공개 detector 사용 가능’으로 소개하지 않는다.

## 3. Wellhausen: 센서 불확실성을 반영한 점별 변화 거리와 후보 점수

**논문:** *Reliable Real-time Change Detection and Mapping for 3D LiDARs*, Wellhausen et al., SSRR 2017. §III-C 식 (1)은 점/occupancy voxel 차이에 대한 Mahalanobis D²를 쓰고 voxel 경계 양자화 오차를 보정한다. semantic/instance 검출이 후보 생성의 전제는 아니다. 이후 HDBSCAN으로 변화 후보를 묶고, 32개 기하·거리 특징의 RF 분류와 반복 관측으로 오검출을 거른다. RF는 학습이 필요하지만 객체 인식기가 아닌 **변화 후보 군집 분류기**다. [저자 원문, §III-C–F](https://cesarcadena.ethz.ch/files/SSRR2017_lwellhausen.pdf)

핵심 한계는 원문이 **robot pose uncertainty가 무시 가능하다고 가정**한다는 점이다. Σ는 주로 range와 beam 방향·spread의 오차다. 미관측 공간은 D²=0으로 제외하며, 이는 ‘변화가 없다고 관측되었다’와 구별해야 한다. D²는 정규화된 모순 크기이고 RF score도 calibration이 검증된 공간 변화확률은 아니다. [원문 §III-C, Algorithm 1·2](https://n.ethz.ch/~cesarc/files/SSRR2017_lwellhausen.pdf)

실제 실험은 저속 sweeping LiDAR의 urban/폐발전소 자료다. Power Plant의 **최종 mapped cluster** P/R은 반복 관측 2회에서 .91/.84, 3회에서 1.0/.72다. 이것은 point-level score의 정확도나 일반 RGB-D 실내 성능이 아니다. 원문은 raycast 오류, 얇은 구조, 0.5m 미만 물체와 군집 대응 실패를 보고한다. [원문 §IV/Table I](https://cesarcadena.ethz.ch/files/SSRR2017_lwellhausen.pdf)

**코드 상태:** 저자 [출판 목록](https://cesarcadena.ethz.ch/publications.html)과 공개 저장소 검색에서 해당 detector 구현·RF 모델을 확인하지 못했다. ‘코드가 없다’는 단정이 아니라 **이번 조사에서 저자 공개 구현이 확인되지 않은 상태**다.

## 이번 요구에 대한 판단

- **분할 없이 먼저 공간 변화 후보를 만드는 기존 기술은 확인됐다.** Underwood의 ray/free-space 모순과 Fehr의 TSDF 차이는 객체가 무엇인지·어디까지 분할되는지를 후보 생성의 전제로 삼지 않는다. Wellhausen은 이 공간 증거를 센서 잡음에 맞춘 수치로 표현한다.
- **공개 구현을 확인한 것은 Underwood의 core다.** 오래된 archived 코드와 실제 잡음 실내 실험이라는 조건을 함께 봐야 한다. Fehr와 Wellhausen은 연구 근거는 있으나 이번 확인으로는 바로 사용할 공개 detector가 확보된 상태가 아니다.
- **후보점수와 변화 확정·사건 분류는 별개다.** 여기의 자유공간 위반·TSDF 차이·Mahalanobis 거리·RF score를 ‘객체가 실제 이동했을 확률’로 바꿔 부르지 않는다. 이동의 동일 객체 대응은 이 세 공간 후보 방식만으로 완결되지 않는다.
- **Bayesian/evidential이라는 이름만으로 조건을 충족한다고 보지 않는다.** 이번 확인 범위에서 실내 RGB-D 잡음 실험·분할 없는 수치 후보·저자 detector 코드를 함께 확인한 evidential occupancy 방법은 추가 확보하지 못했다. [OctoMap의 `enableChangeDetection`](https://github.com/OctoMap/octomap/blob/devel/octomap/include/octomap/OccupancyOcTreeBase.h)은 scan 삽입 중 바뀐 voxel key 추적 API다. 그 API나 occupancy wrapper만을 검증된 전후 변화 탐지 알고리즘이라고 부르지 않는다.

새 점수 결합식이나 구현을 제안·적용하지 않았다. 위 판단은 1차 자료 조사 결과이며 자체 데이터 성능 검증은 아니다.
