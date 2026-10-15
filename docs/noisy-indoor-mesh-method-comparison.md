# 노이즈 실내 메쉬맵의 변화·의심 영역: 기존 방법 재비교

조사일: 2026-10-15. 연구 자료·공개 소스만 확인했다. 탐지기 실행, 패키지 설치, 메쉬 처리 실험, 뷰어 변경은 하지 않았다.

## 확정된 문제와 판단

주 입력은 t0/t1의 재구성된 3D 메쉬 두 개다. 울퉁불퉁한 표면, 물체 사이에 잘못 생긴 연결부, 정합 오차를 포함할 수 있다. Instance ID, 객체 검출·매칭은 판단의 전제가 아니다. 사용자가 원하는 출력은 충분한 변화 근거를 가진 영역과, 차이는 있지만 판단이 불확실한 의심 영역이다.

**현재 확인한 후보 중 이 두 출력을 노이즈 실내 메쉬에서 신뢰성 있게 구분하고, 공개 탐지 코드까지 제공하는 단일 완제품은 확인하지 못했다.** 거리·유사도·유의성·미관측 상태를 사용자 출력과 구별해야 한다.

판단은 다음과 같다.

1. **공개 구현으로 시작할 기하 비교 기준:** CloudCompare의 C2M과 M3C2. 실제 메쉬를 받거나 메쉬 표면점으로 비교할 수 있다. 다만 불확실성의 근거까지 메쉬에서 복원되는 것은 아니다.
2. **노이즈와 구조 변화 구별의 연구 비교 후보:** 다중 스케일 GLS. 실제 실내 스캔 근거가 있지만 detector 공개 코드는 미확인이다.
3. **메쉬 직접 불확실성 모델의 참고:** Moenck & Schüppstuhl 2024. 면 단위 불확실성을 지원하지만 외부 오차 모델이 필요하고 실제 방 재구성에 대한 검증·공개 코드는 미확인이다.
4. **주 후보에서 후순위:** PlaneSDF의 평면·blob 가정, ROI Similarity의 점군/기하 군집 실험, Fehr의 추가 TSDF 관측 정보 요구를 고려하면 범용 메쉬 두 개의 두 등급 출력을 위한 즉시 선택으로 보기 어렵다.

위 순위는 입력·출력·구현 접근성에 대한 조사자의 판단이다. 우리 메쉬에서 측정한 정확도 순위가 아니다.

## 같은 기준의 비교

| 방법 | 메쉬 두 개로 입력 가능? | 실제 노이즈 실내 근거 | 원래 출력과 변화/의심 구분 | 공개 탐지 구현 |
|---|---|---|---|---|
| C2M / Hausdorff | 가능. 비교 메쉬의 vertex 또는 표면 sample과 상대 triangle을 비교 | 범용 거리 도구이며 이번 조사로 실내 재구성 noise 분리 성능을 확보한 것은 아님 | 거리 field. 변화/의심 판정과 오차 모델은 없음 | CloudCompare / MeshLab 공개 |
| M3C2 | 표면 sampling 후 가능 | 실제 실내 RGB-D benchmark가 있으나 surfel 점군이며 정리·정합·고정 threshold 조건 | 거리, LoD, 유의성. 비유의는 안정/높은잡음/낮은지지를 함께 포함. 두 출력은 별도 판단 | CloudCompare / py4dgeo 실제 코드 확인 |
| M3C2-PM / EP | 표면 형상 외 외부 정밀도/센서·정합 오차 자료 필요 | 이번 확인으로 generic noisy room mesh 실험을 확보하지 못함 | 외부 오차를 반영한 거리 불확실성/LoD. 의심 영역 native 출력 아님 | 공개. PM과 EP의 추가 입력은 서로 다름 |
| 다중 스케일 GLS, Palma et al. | 메쉬 표면 sample로 적용 가능하다는 것은 입력 모델에 따른 추론. 원 실험은 점군 | 실제 사무실·실험실 ToF scans, 다른 표본 밀도·촬영 방향·표면 교차 | 점별 0–1 change value와 change/no-change. calibrated probability/명시적 의심 등급 아님 | 저자 detector 코드 미확인. 공개 Change Vis는 시각화기 |
| PlaneSDF | 완성 점군 경로는 있음. 메쉬 sampling의 성능은 미검증 | 실제 RGB-D/Voxblox 실내 5종. 소파 noisy boundary와 높이방향 가림 실패 보고 | SDF feature/blob similarity로 binary 변화 판정. 미관측 mask는 의심 등급과 다름 | 저자 detector 코드 미확인 |
| ROI Similarity | 점군 변환은 가능하나 실험의 원래 입력은 LiDAR 점군 | 실제 지하주차장 1쌍 + 별도의 합성 터널 noise 실험 | ROI similarity와 thresholded change. 통계적 confidence/명시적 두 등급 아님 | 저자 detector 코드 미확인 |
| Moenck & Schüppstuhl 2024 | **mesh-to-mesh를 명시적으로 지원**, 면별 불확실성을 추가 지정 | noisy ToF 기하 모형, 실제 Artec Eva 항공기 동체 mesh. 일반 실내 방은 아님 | 확률분포 간 Bhattacharyya distance와 thresholded change/no-change. BD 자체로 significance를 보장하지 않음 | 저자 detector 코드 미확인 |
| Fehr TSDF | **최종 mesh 두 개만으로 원래 방법 재현 불가**. TSDF 값·weight·관측공간 필요 | 실제 Tango RGB-D living room/office/lounge | voxel TSDF 차이, dynamic labels, 후단 영역. 원래 출력은 충분변화/의심 두 등급 아님 | 공식 repo는 데이터. detector 코드 미확인 |

## 1. 바로 비교 가능한 공개 구현의 범위

CloudCompare는 두 메쉬를 선택할 수 있지만 실제 계산은 비교측 vertices에서 reference triangles까지의 거리다. vertices가 성기면 표면 sampling을 공식 문서가 권한다. 따라서 단순 vertex 차이·triangle index 차이·메쉬 개수 차이를 측정하는 방법과 다르다. 양방향으로 계산하는 경우에도 confidence나 원인 분류가 자동 생기지 않는다. [공식 C2M 문서](https://www.cloudcompare.org/doc/wiki/index.php/Cloud-to-Mesh_Distance), [MeshLab 공개 구현](https://github.com/cnr-isti-vclab/meshlab)

M3C2는 국소 평균 표면의 normal 방향 차이와 LoD를 준다. 공식 출력에는 distance uncertainty, change significance가 있으며 대응 이웃이 없으면 NaN이다. 표면 샘플을 늘리면 새로운 센서 관측 없이 표준식의 n만 늘어날 수 있다. **메쉬 재샘플링이 통계적 신뢰를 자동 높이지 않는다**는 것은 원 식·코드로부터의 통계적 해석이며 논문이 mesh 실험으로 검증한 결론은 아니다. [공식 문서](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)), [현재 계산 코드](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp)

PM은 외부 sigmaX/Y/Z를 normal 방향으로 전파하고 표준식처럼 n으로 나누지 않는다. 그러나 해당 sigma가 없으면 mesh만으로 입력 조건이 충족되지 않는다. EP는 scan origin/촬영출처, range/angle 오차, 변환·정합 covariance 등 더 많은 정보를 요구한다. 둘을 mesh noise를 자동 추정하는 버튼으로 소개하지 않는다. [PM 공식 설명](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)#Precision_maps), [EP 공식 튜토리얼](https://py4dgeo.readthedocs.io/en/stable/m3c2ep.html)

## 2. 실제 실내 결과를 어떻게 읽어야 하나

**M3C2:** MultiChange3D Open-space RGB-D 요약 F1 52.2%, P 51.8%, R 55.4%다. 이것은 RealSense/BAD-SLAM surfels를 1cm downsample하고 안정영역 정합·gross artifact 정리를 수행한 실험이다. 평가는 고정 10cm 거리 threshold이며 LoD로 의심 등급을 평가한 것이 아니다. 이 수치로 noisy mesh의 두 등급 성능이나 모든 M3C2 구성의 우열을 단정할 수 없다. [발표본 Tables 3–5](https://isprs-annals.copernicus.org/articles/XI-2-2026/639/2026/isprs-annals-XI-2-2026-639-2026.pdf), [실제 평가 코드](https://github.com/3DOM-FBK/multichange3d/blob/main/evaluation/compute_metrics.py)

**GLS:** 점군의 국소 밀도를 반영한 이웃과 여러 scale의 implicit surface descriptor 차이를 사용한다. 실제 OFFICE Fig.1에서 의자 이동과 desk/floor의 sampling 차이를 비교하며 단순 nearest distance의 speckle과 temporal surface intersection 문제를 줄인 결과를 보인다. 이미 정합된 입력·normal 추정이 필요하고 큰 이웃은 작은 변화에 불리하다. 원 논문의 point field를 mesh sample에 쓰는 것, 공간 구역 단위로 집계하는 것, 불확실 등급을 정의하는 것은 각각 구분해야 한다. [저자 프로젝트](https://perso.telecom-paristech.fr/boubek/papers/ChangeDetect3D/), [원문](https://openportal.isti.cnr.it/data/2016/359538/2016_359538.postprint.pdf)

**PlaneSDF:** 실제 실내 Table I 평균 F1 0.76은 작은 YCB 추가 객체 평가이며 이동한 가구·장식은 평가에서 제외했다. 소파의 noisy boundary, 높이방향 가림, 붙어 있는 planar part의 누락이 보고된다. 이 결과는 노이즈가 있는 방 재구성을 다룬 중요한 증거지만 임의의 붙은 메쉬 표면에 대한 범용성을 보증하지 않는다. `isChanged(blob)`는 similarity threshold 판정이고 feature similarity는 uncertainty posterior가 아니다. [원문 §IV/Table I](https://arxiv.org/html/2207.08323v2)

**ROI Similarity:** 실제 garage에서 추가 대상 8개 검출·오탐 5구역이며 2cm observation noise는 별도 합성 터널 실험이다. 점군 seed 이후 기하 군집을 만든다는 점은 인스턴스 ID 의존성을 피하지만, 굴곡이 큰 표면에서 ROI가 조각날 수 있다. mesh-to-mesh 실험이나 신뢰도별 두 출력 검증은 확인하지 못했다. [원문](https://www.mdpi.com/2079-9292/15/20/4629)

## 3. 메쉬 자체의 불확실성을 다루는 직접 사례

Moenck & Schüppstuhl, *Geometric digital twins of long-living assets: uncertainty-aware 3D images from measurement and CAD data*, Procedia CIRP 2024는 point/mesh 조합 네 가지를 명시한다. 면별 또는 전체 mesh의 불확실성을 지정하고, 면에서 샘플을 뽑아 대응 분포의 Bhattacharyya distance를 비교한다. 오차 분포는 센서 calibration, 측정 또는 tolerance 등 외부 근거로 공급한다. 메쉬 형상만 보고 완전한 uncertainty를 복원하는 방법은 아니다. [출판사 초록](https://doi.org/10.1016/j.procir.2024.08.369)

실험은 low-cost ToF Lenovo Phab2Pro의 noisy 기하 모형과 Artec Eva 동체 구조물 mesh다. 동체에서 bracket 5개 추가·1개 제거를 제시하지만, Fig.8의 측정 artifact를 novelty로 오인하고 해상도/threshold 때문에 경계가 뜨는 현상도 보고한다. 일반 주거·사무실 재구성의 PR/F1이나 arbitrary reconstruction-noise sweep은 없다. 저자는 BD 자체의 통계적 유의 threshold가 정해지지 않는다고 명시한다. 최종 출력도 change/no-change이며 의심 등급이 따로 검증된 것은 아니다. [저자기관 원문 §§4–6](https://tore.tuhh.de/bitstreams/44c4653c-a4b9-49b9-a597-d8e06a5ed8ae/download)

가림 처리는 Region of View 및 viewpoint/sensor metadata를 사용하며 예제에서는 bounding box와 Plane of Sight를 수동 지정했다. 따라서 visibility-aware 부분까지 mesh 두 개만으로 된다고 확대하지 않는다. 출판사·저자 페이지·공개 repository 검색에서 저자 탐지 코드를 확인하지 못했다. 이것은 부재 증명이 아니다.

## 변화와 의심을 나누는 데 남는 작업

기존 방법의 native 출력을 사용자 출력과 구분하는 원칙은 다음과 같다. 이는 이번 조사에서 정리한 해석 기준이며 검증된 새 detector가 아니다.

- 차이가 충분히 크고 오차 모델·관측 지지가 적절하면 변화 근거가 강해진다. 큰 거리 하나만으로 실제 변화 확정이 되지는 않는다.
- 차이가 있으나 오차와 분리되지 않거나, 큰 scale에서만 나타나거나, 경계·합쳐진 면·불안정 normal·불충분 관측에 의존하면 불확실성이 남는다.
- **비유의는 의심과 동의어가 아니다.** 실제로 안정된 표면도 비유의다. M3C2 NaN 역시 미관측/검색실패 등이 섞인 상태다.
- 잘못 생성된 mesh 연결부나 구멍 메움은 실제 표면 추가와 같은 형상 차이를 만들 수 있다. 원 관측·재구성 uncertainty가 소실된 mesh만으로 두 원인을 항상 구별한다고 주장하지 않는다.
- 여러 scale에서 차이가 지속되는 것은 유용한 기하 근거지만, 여러 scale에서 지속되는 reconstruction drift도 가능하므로 실제 변화의 독립 증명이 아니다.

## 우리에게 맞는 선택

**현 시점의 공개 구현 기준으로는 M3C2를 기본 표면 차이/오차 비교 기준으로 두는 것이 가장 실행 가능하다.** C2M은 mesh 직접 거리 비교의 기준선이다. 그러나 mesh 표면 sampling의 sample count와 재구성 오차를 고려하지 않은 표준 LoD를 그대로 95% 물리 변화 confidence로 보여주어서는 안 된다. 외부 vertex precision이 확보되면 PM, 더 완전한 acquisition/registration metadata가 있으면 EP를 별도로 검토할 수 있다.

**연구 목표에 가까운 기하 보강 비교는 GLS**, mesh-native uncertainty의 설계 참고는 Moenck 2024다. 두 방법 모두 현재 공개 detector를 확보했다고 말할 수 없다. PlaneSDF는 평면/지지면 중심의 특화 비교로 후순위다. Fehr/ray 방식은 RGB-D/pose 또는 기존 TSDF를 보존할 때 visibility의 보조 비교축이며, mesh-only의 직접 대체물이 아니다.

사용자의 최종 두 등급 출력까지 완성된 방법을 찾았다는 결론은 내리지 않는다. 현재 추천은 **기존 공개 기하 비교를 선택하고, 그 evidence와 uncertainty를 구분해 연구하는 출발점**에 대한 판단이다.

## 근거 확인 및 코드 범위

- CloudCompare C2M 문서의 두 mesh/vertex-to-triangle 동작을 확인했다. M3C2/PM 공식 코드와 EP API는 별도 통계 비교 노트에 기록했다.
- GLS 저자 페이지는 논문을 제공한다. Change Vis zip은 precomputed change field를 입력받는 시각화 코드이며 GLS detector로 간주하지 않는다. [Change Vis](https://vcg.isti.cnr.it/activities/visChange//)
- PlaneSDF 저자 publication page·repository 검색, Fehr 공식 dataset repository는 이번에 재확인했다. [volumetric 상세 노트](mesh-change-volumetric-comparison.md)
- Moenck 원문 6쪽을 추출하고 관련 전체 3–5쪽을 렌더링해 입력 오차, mesh 대응, 실제 mesh와 오검출 Fig.8을 시각 확인했다. 해당 source는 `outputs/research-sources/mesh-comparison/Moenck-2024.pdf`에 보관했다.
- [통계 비교 상세](mesh-change-statistical-comparison.md), [이전 후보 조사](segmentation-free-change-shortlist.md)는 보충 자료다. 이 문서가 이번의 mesh 기준 추천과 한계를 종합한다.
