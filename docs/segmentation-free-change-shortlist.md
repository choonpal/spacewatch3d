# 노이즈가 포함된 3D 메쉬맵의 기하 변화와 불확실 영역 탐지

확정된 메쉬 입력·실제 실내 근거·두 등급 출력·공개 코드를 재비교한 최신 종합 판단은 [noisy-indoor-mesh-method-comparison.md](noisy-indoor-mesh-method-comparison.md)에 기록했다. 아래 후보 발견 기록은 그 입력·출력 제한과 함께 읽는다.

조사일: 2026-10-15. 아래 문제 정의는 사용자가 최종 확정한 조사 기준이며 이전의 객체 중심 해석보다 우선한다. Rescan은 임시 자료일 뿐 선정 조건이 아니다. 구현·설치·검출 실행·뷰어 수정은 하지 않았다.

## 사용자가 확정한 문제 정의

> 서로 다른 시점에 재구성된 노이즈가 포함된 3D 메쉬맵에서, 객체 분할 결과에 의존하지 않고 국소적인 기하학적 차이를 분석하여 실제 변화 영역과 변화 여부가 불확실한 영역을 어떻게 신뢰성 있게 탐지할 수 있는가?

**주 입력:** 시점 t0, t1의 3D 메쉬맵 두 개. 울퉁불퉁한 표면, 재구성 노이즈, 인접 물체의 메쉬가 합쳐진 상태를 포함한다. 기존 instance label의 정확성을 가정하지 않는다. RGB-D 및 카메라 자세는 이용 가능한 보조 관측이며, 모든 방법에 필수라고 가정하지 않는다. 메쉬 표면을 점으로 샘플링해 비교하는 방법은 주 입력에서 파생된 표현으로 구분한다.

**목표 출력:**

- 변화 영역: 실제 변화가 발생했다는 근거가 충분한 공간 영역.
- 변화 의심 영역: 메쉬 간 차이는 있으나 재구성 노이즈·정합 오차 등의 영향 때문에 판단이 불확실한 공간 영역.

이 두 출력은 사용자 요구다. 특정 기존 논문이 그대로 제공하는 출력인지, 논문의 거리·신뢰도·유의성에서 별도 후처리로 만들어야 하는 출력인지는 조사 결과에서 구분한다. 통계적 비유의성을 곧바로 변화 의심으로 해석하거나 모든 노이즈 영역을 의심 영역으로 분류하지 않는다.

| 우선순위 | 기술 | 조사할 핵심 |
|---|---|---|
| 1 | Mesh-to-Mesh / Point-to-Point Change Detection | 두 메쉬의 국소 형상 차이와 실제 입력 요건 |
| 2 | Multi-scale Geometric Comparison | 표면 노이즈와 구조적 변화의 구별, 작은 변화의 누락 |
| 3 | Uncertainty-aware Change Detection | 재구성·정합 오차의 모델과 변화/불확실 판단 근거 |
| 4 | Visibility-aware Change Detection | 관측 누락·가림 처리, 메쉬 두 개만으로 가능한 범위와 원본 depth/pose가 필요한 범위 |

Instance Segmentation 부정확성은 문제의 발생 배경이다. **분할 개선, 객체 재식별, 객체 매칭은 우선 조사 대상에서 제외한다.** 현재 연구의 중심은 메쉬맵 자체의 공간·기하 변화와 그 불확실성이다.

## 이번의 필수 조건

객체 단위로 나뉘지 않은 메쉬 표면에서도 공간 변화 증거를 먼저 만들 수 있어야 한다. 점·복셀·국소 표면의 비교와 후보 생성 이후 기하 군집화는 허용한다. GT 마스크가 필요 없다는 말만으로 노이즈가 낀 메쉬의 불확실성을 처리한다고 판정하지 않는다.

이번에 확인한 방법들은 **서로 다른 입도와 의미의 수치**를 만든다. 점수 지도가 있다는 사실과, 연결된 구역별 검토 순위가 완성되어 있다는 사실도 구별한다.

| 방법 | 객체 분할 전 증거와 수치 | 노이즈·미관측 처리 | 실내 근거 및 구현 상태 |
|---|---|---|---|
| ROI Similarity, Electronics 2026 | 기하 거리 seed → 공간 군집 → normal/구조 network 기반 ROI 유사도 | ICP/SOR, 밀도 적응 seed, regional similarity로 후보 선별. 기하 군집도 잡음에 영향받음 | 실제 지하주차장 추가 대상 8개 검출·오탐 5구역. 공개 detector 코드 미확인 |
| Palma et al., CGF 2016, multi-scale GLS | 정합된 두 전체 점군의 국소 implicit surface descriptor 차이. 점별 `ch_p` 변화 값 | LoOP 이상점 제거, 밀도에 따른 이웃 크기, 여러 scale의 표면 형상 비교, 기하 일관성·전파 후처리 | 실제 사무실 의자/책상 장면 포함. detector 저자 코드 미확인 |
| Wellhausen et al., SSRR 2017 | 센서 오차로 정규화한 점별 Mahalanobis D² → 후보 군집 → RF score | raycast 관측 범위, voxel 경계 보정, 반복 관측. 로봇 자세 오차는 무시 가능하다고 가정 | 실제 폐발전소·도시 LiDAR. detector/RF 모델 코드 미확인 |
| M3C2 / M3C2-EP | 국소 표면 거리와 LoD/uncertainty, 유의 변화 여부 | roughness/count/정합 오차 또는 sensor·registration covariance. 상대 관측이 없으면 NaN | 공개 CloudCompare/py4dgeo. 실제 실내 RGB-D에서 만능이라는 근거는 없음 |
| Fehr et al., ICRA 2017 | 객체 추출 전 복셀 TSDF 절대 차이와 관측 weight | 저신뢰 복셀 제외, 3D erosion/dilation, 관측된 자유공간과 미관측 구별 | 실제 Tango RGB-D 실내. 공식 저장소는 데이터이며 detector 코드 미확인 |
| Underwood et al., ICRA 2013 | 점과 상대 관측 ray의 자유공간 모순 | range/angular tolerance, coverage와 가림 검사 | 실제 실내 lab + 실제 저자 C++ core 공개. 배포 출력은 thresholded change points이며 확률/구역순위가 아님 |

출처와 더 자세한 관측·코드 상태: [RGB-D/ray/TSDF 조사](segmentation-free-rgbd-change-research.md), [점군·불확실성 조사](segmentation-free-point-change-research.md).

## ROI Similarity와 학습 후보 재확인

ROI Similarity는 객체 semantic/instance 마스크 없이 적응적 거리로 변화 seed를 얻은 뒤 RBNN 기하 군집의 normal 및 구조 유사도를 계산한다. 낮은 유사도가 변화 근거다. 실제 주차장의 추가 대상 8개 검출·오탐 구역 5개는 한 쌍의 장면 결과다. 합성 터널의 2cm 관측 noise 실험과 실제 결과를 섞지 않는다. 기하 군집 자체의 분절·불균일 표면 문제는 남으며 공개 실행기와 일반 실내 RGB-D 성능은 확인되지 않았다. 이는 **기존 구역 점수 방식**의 근거이지 우리 장면의 검증된 detector라는 뜻이 아니다. [원문](https://www.mdpi.com/2079-9292/15/20/4629)

최근 SPICD-Net도 정합된 전체 점군 tile에서 instance 없이 softmax score를 만들지만, 저자가 coordinate jitter/yaw 오차에서 큰 성능 저하를 보고하고 공개 코드는 요청 제공 방식이다. 따라서 segmentation-independent라는 이유만으로 noise-robust 후보로 추천하지 않는다. [원문](https://www.mdpi.com/2673-2688/7/4/141)

## 새로 확인한 GLS: 객체가 분할되지 않아도 점마다 변화 값 생성

논문 *Detection of Geometric Temporal Changes in Point Clouds*는 정합된 두 점군으로 시작한다. 국소 밀도에 따라 query grid와 이웃 반경을 정하고, 20개 scale의 Growing Least Squares 표면 descriptor를 비교한다. 식 (4)는 주변 query의 descriptor 차이를 가중 평균해 점별 `ch_p`를 만든다. 논문은 이 무차원 값을 [0,1]로 설명하고 0.05 threshold를 사용하지만, calibration 평가를 확인하지 못했으므로 이를 관측 빈도와 일치하는 변화 확률로 부르지 않는다. [저자 원문 §§3–5](https://openportal.isti.cnr.it/data/2016/359538/2016_359538.postprint.pdf)

실제 OFFICE 장면의 Fig.1에서는 이동한 의자와 책상·바닥의 다른 sampling 방향을 비교한다. 최단거리 4mm threshold보다 교차/중첩부의 분류와 배경 speckle이 개선된 결과를 제시한다. 합성 실험과 실제 장면 결과를 구별하며, 이를 자체 RGB-D 데이터의 정확도 수치로 바꾸지 않는다. 이미 정합된 입력을 가정하고 normal/local surface 추정이 남으며, 큰 scale은 작은 변화를 놓칠 수 있다. 이는 임의 수준의 잡음·drift를 해결한다는 보장이 아니다. [저자 소개](https://perso.telecom-paristech.fr/boubek/papers/ChangeDetect3D/), [출판 정보](https://doi.org/10.1111/cgf.12730)

공개 [Change Vis](https://vcg.isti.cnr.it/activities/visChange//)에는 이 방법으로 미리 계산한 change field와 시각화 코드가 있다. **이 코드는 detector 공개 구현과 다르다.** 해당 페이지는 준비 단계에서 vertex별 field를 먼저 계산하라고 하므로, viewer zip을 GLS detector 코드로 소개하지 않는다. 논문의 점별 change field를 구역 하나의 검토 점수로 집계하는 방법도 따로 정해야 한다.

## 비교에서 실수하면 안 되는 부분

- **점수 의미:** GLS 형상 차이, M3C2 표면 분리/LoD, TSDF 차이(m), Mahalanobis D², RF 분류 score는 같은 척도가 아니다. 단순 0–100 변환으로 같은 변화 확률이 되지 않는다.
- **가림:** 점이 없다는 것과 실제로 빈 공간을 관측했다는 것은 다르다. ray/TSDF 방법은 depth와 센서 원점을 유지하는 이유가 여기에 있다. 누적 점군 전체에 단일 가상 센서 원점을 적용하면 원래 관측 모델을 잃는다.
- **자세:** 분할 불필요가 정합 불필요를 뜻하지 않는다. 특히 Wellhausen의 sensor covariance를 pose uncertainty까지 처리하는 것으로 확대하지 않는다.
- **지역 점수:** GLS/M3C2는 점·표면 수준 증거다. Wellhausen의 RF는 후보 군집 score를 직접 다루지만 change-cluster 학습이 필요하고 공개 모델은 미확인이다.
- **이동:** 변화 후보를 독립적으로 남기는 것과 이전/새 위치를 같은 물체로 대응해 moved로 확정하는 것은 별개다. 후자의 실패로 전자의 후보가 사라지지 않아야 한다는 것이 사용자 요구다.

## 판단

**기존 방법은 존재한다.** ROI Similarity는 기하 후보 구역의 유사도, GLS는 점별 표면 차이, Wellhausen은 Mahalanobis 증거와 변화 군집 score, M3C2는 오차 대비 표면 차이의 통계적 근거를 준다. RGB-D와 자세가 남아 있는 조건에서는 Fehr/Underwood의 관측된 자유공간 처리가 중요한 비교 축이다.

즉시 활용 가능성과 기능 적합성은 분리한다. 공개 코드 확인 기준으로 M3C2와 Underwood core가 남고, 변화 후보·연속 점수라는 기능 기준으로 GLS/Wellhausen/Fehr를 연구 참고에 둔다. **노이즈 실내 RGB-D, 미분할 공간 후보, 신뢰도 있는 구역 순위, 모든 사건 대응, 현대적 공개 실행기**를 한 번에 충족하는 단일 완제품은 이번 조사에서 확인하지 못했다. 이는 위 연구들이 후보 탐지에 쓸 수 없다는 뜻이 아니라 통합 기능을 확대 해석하지 말아야 한다는 뜻이다.

새 결합 점수식·탐지기·실험은 작성하거나 실행하지 않았다.
