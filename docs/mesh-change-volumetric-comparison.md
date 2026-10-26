# Noisy mesh 두 개 조건에서 PlaneSDF·Fehr TSDF 재평가

조사일: 2026-10-15. 주입력은 **t0/t1의 울퉁불퉁하고 일부 물체가 붙은 3D 메쉬 두 개**다. 목표는 geometry-only 변화 근거가 충분한 영역과 불확실한 의심 영역을 구분하는 기존 방법이며, 분할 개선·객체 재식별·객체 대응은 범위에서 제외한다. 자료와 소스만 읽었고 설치·실행·새 점수식·구현·데이터 변경은 하지 않았다.

## 먼저 내릴 수 있는 판단

**두 방법 모두 현재 요구의 직접적인 1순위로 추천하기 어렵다.** PlaneSDF는 완성 점군 입력 경로가 있어 메쉬에서 얻은 점을 적용할 가능성은 있지만, 이는 검증되지 않은 입력 변환이며 평면·object-blob 처리에 의존한다. Fehr는 TSDF의 관측 정보가 필요하므로 메쉬 두 개만으로 원래 방법을 재현할 수 없다. 어느 논문도 ‘변화 근거 충분 / 의심’이라는 두 증거 등급을 native 출력으로 검증하지 않았다.

위는 논문 요구 조건과 사용자 목표를 비교한 **조사자의 적합성 판단**이며 성능 실험 결과가 아니다.

| 확인 항목 | PlaneSDF (RA-L/IROS 2022) | Fehr TSDF (ICRA 2017) |
|---|---|---|
| 원래 입력 | 공통 좌표계의 두 scene point cloud. 완성 점군과 순차 스트림 경로 모두 설명 | 관측값·weight가 보존된 정합 TSDF volume. 생성 단계는 RGB-D와 자세 사용 |
| 메쉬만 직접 입력 | 논문에서 검증하지 않음. 메쉬→표면 샘플 점군은 적용 가능성에 대한 추론 | 지원 근거 없음. 메쉬에는 TSDF 관측 weight·free/unknown 이력이 보존되지 않음 |
| raw depth·poses 필요성 | 완성 점군 경로에서는 detector 필수 입력 아님. 이미 공통 좌표계여야 함 | TSDF 생성에는 필요. 올바른 TSDF를 이미 보관했다면 변화 검사 때 raw 자료를 다시 요구하지 않음 |
| 기하 수치 | height 차이, SDF descriptor similarity, blob 평균 similarity와 임계값 | 절대 TSDF 차이(m), 관측 weight와 dynamic/static 임계값 |
| native 확실/의심 등급 | 없음. 논문은 ternary 2D mask·미관측 height 상태를 두지만, 최종 blob 판정은 change 여부 | 없음. 관측/미관측 volume 상태와 static/dynamic label은 별개 정보 |
| 현재 공개 구현 | 저자 구현 미확인 | 저자 detector 구현 미확인. 공개 데이터 저장소만 확인 |

방법 근거: [PlaneSDF §III-A–D](https://arxiv.org/html/2207.08323v2), [Fehr §III-A–C](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf).

## PlaneSDF: 메쉬를 샘플링해도 원래의 한계는 남음

논문 §III-A는 완성 scene point cloud에서 평면을 찾는 경로를 명시한다. 따라서 원본 RGB-D나 카메라 궤적이 없다는 이유만으로 PlaneSDF를 배제하면 부정확하다. 다만 **triangle mesh에서 샘플링한 점군을 사용한 실험은 확인되지 않았다.** 메쉬의 평활화·구멍 메움·붙은 표면도 샘플에 반영된다는 것은 입력 변환에 따른 해석이지 논문의 robustness 결과가 아니다. [원문 §III-A/B, PDF pp. 2–3](https://arxiv.org/pdf/2207.08323#page=2)

수평·수직 지지평면과 height-map 연결 성분으로 객체 또는 가까운 객체 묶음을 만든다. GT instance segmentation은 필요 없지만, 완성 방법은 이 **기하 blob 단위 검증**에 의존한다. 사용자가 제외한 분할 복구 방법으로 추천할 수 없으며, 붙은 물체를 각각 구별해야만 실행되는 방법도 아니다. 대신 붙은 부분이 한 blob으로 평가될 수 있다. [원문 §III-A/D](https://arxiv.org/html/2207.08323v2)

식 (6)의 `sim = 1 / (1 + α‖f−f′‖₂)`는 local SDF feature 유사도다. 식 (7)은 blob의 histogram 평균 `Havg`가 임계값보다 작으면 changed로 판정한다. **낮은 유사도가 변화확률이고 중간 유사도가 의심 확률이라는 정의는 없다.** height=-1의 미관측 상태와 논문이 부르는 ternary mask도, 변화가 실제인지 불확실한지를 보정한 확신 등급과 동일하지 않다. [원문 §III-A 및 식 (6)–(7), PDF pp. 3·5](https://arxiv.org/pdf/2207.08323)

실제 ASUS Xtion/Voxblox 실내 재구성의 구조 어긋남·노이즈 경계에서 평가했다. 그러나 소파의 noisy/deforming 경계, 높이 방향 가림, 평면에 붙은 skillet 바닥 등에서 누락을 보고한다. 실제 표의 평균 F1=0.76은 새로 놓인 YCB 물체 대상이고 움직인 가구·장식 변화는 평가에서 수동 제외했다. 따라서 임의 noisy mesh 전체 변화와 의심 영역 분리 성능의 근거로 확대하면 안 된다. [원문 §IV-A/E, Table I 및 실패 사례, PDF pp. 5–8](https://arxiv.org/pdf/2207.08323#page=7)

**구현 fresh check:** 저자 [현재 출판 페이지](https://jiahui-fu.github.io/)의 PlaneSDF 항목은 Paper·Video만 제공한다. 논문 제목/방법명에 대한 공개 저장소 검색에서도 실행 가능한 저자 detector를 확인하지 못했다. 이는 공개 구현이 어디에도 없다는 단정이 아니다.

## Fehr TSDF: 저장된 TSDF와 메쉬는 다른 입력

§III-A/B는 RGB-D와 정합된 자세로 만든 volume의 signed distance `f`와 관측 weight `w`를 사용한다. `w=0`은 미관측, `f=truncation` 및 `w>0`은 관측된 자유공간이다. TSDF를 완성한 뒤에는 raw 카메라 자료가 필요 없다는 논문 설명을 **메쉬만 있어도 된다는 의미로 읽으면 안 된다.** [원문 §III-A/B, PDF pp. 2–4](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf)

메쉬에서 새 SDF를 계산하더라도 원래 어떤 공간을 관측했는지와 얼마만큼 관측했는지를 되살리는 것은 아니라는 점은 입력 정보로부터의 추론이다. 이 손실 때문에 ‘메쉬→SDF’만으로 해당 논문의 가림·미관측 처리까지 동일하다고 주장할 수 없다.

변화 검사는 공통 관측 voxel의 `|f_new−f_static|`와 임계값, weight 필터, erosion/dilation을 사용한다. 이후 mesh region growing·connected components가 따라온다. 최초 voxel 증거는 객체 분할 성공과 독립적이지만, **native 출력은 static/dynamic이고 별도 의심 영역이나 confidence calibration은 없다.** [원문 §III-B/C, 식 (4)–(10)](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf)

실제 Tango 실내 반복 촬영에서 노이즈·부분 관측을 다뤘으며, lounge는 10회 촬영과 약 50–100% 중첩을 포함한다. 원문은 지역적·비강체 정합 오차와 두 관측만 있을 때의 겹친 변화·객체 경계 실패를 인정한다. 최종 객체 검출·분할 지표는 voxel별 확실/의심 분리 평가가 아니다. [원문 §IV/V·Fig. 9](https://tisl.cs.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf), [공식 데이터 설명](https://github.com/ethz-asl/change_detection_ds)

**구현 fresh check:** 공식 [change_detection_ds](https://github.com/ethz-asl/change_detection_ds)의 최상위는 README와 previews다. [저자 출판 목록](https://cesarcadena.ethz.ch/publications.html)과 공개 저장소 확인에서 detector 코드가 재확인되지 않았다. 기반 재구성기 OpenChisel의 공개 여부를 Fehr 변화 탐지 전체의 공개 구현으로 바꿔 말하지 않는다.

## 이번 두 후보의 우선순위

1. **PlaneSDF는 조건부 문헌 비교 대상으로만 유지한다.** 두 후보 중 완성 표면 자료 입력에는 더 가깝지만, arbitrary noisy mesh·붙은 표면·확실/의심 영역 출력이라는 세 조건을 직접 충족한 증거는 없다. 저자 구현도 미확인이다.
2. **Fehr는 메쉬 두 개 입력의 우선 후보에서 제외한다.** 관측 정보가 보존된 원본 TSDF가 별도로 있을 때 재검토할 수 있는 방법이다. 지금 그 자료를 확보하거나 새 volume을 만들자는 제안은 아니다.
3. **두 논문의 유사도/차이 임계값을 임의로 두 등급으로 나누어 기존 기능처럼 소개하지 않는다.** 요구한 출력의 후보를 찾으려면, 표면 비교의 불확실성 또는 판정 유보를 실제로 정의·평가한 방법과 별도로 비교해야 한다.

이 파일만 새로 작성했다. 기존 시연·코드·데이터·다른 연구 노트는 수정하지 않았다.
