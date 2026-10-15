# 정합 이후 잔여 오차 비교: 가구 전 방향 / 공통 구조면 / 수동 ROI 기록

2026-10-16. 로컬 브랜치 `task/5a-change-detection/residual-roi-comparison`. 원격 푸시는 수행하지 않았다. 기본 `mesh-comparison.html`과 고정 2cm 산출물은 유지한다. 새 화면은 [residual-comparison.html](http://127.0.0.1:8768/residual-comparison.html)이다.

## 최신 실행: 설정을 고정한 새 Bonn 촬영 적용

[새 촬영 뷰어](http://127.0.0.1:8768/residual-comparison.html?dataset=bonn-placement). 상단 촬영 쌍 선택에서 기존 Rescan과 새 Bonn 촬영을 전환한다. Rescan 기본 결과와 기존 Bonn 제거 결과는 byte 동일하게 유지했다.

### 실행 전에 고정한 내용

- 새 촬영은 공식 [Bonn 배포 페이지](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)의 `rgbd_bonn_placing_nonobstructing_box`다. 기존에 사용한 `removing_nonobstructing_box`와 다른 기록이며, 다운로드 전에 사용 이력과 선택 이유를 기록했다. **같은 Bonn 실험 공간의 새 촬영**이며, 완전히 새로운 방으로의 일반화 검증은 아니다. 한 연속 영상의 전후 시간 구간을 각각 복원한 쌍으로, 독립적인 두 방문 스캔도 아니다.
- 다운로드/복원/탐지 전 `outputs/frozen-transfer-bonn-placement-20261016/frozen-protocol.json`을 저장했다. SHA256 `78d112ab658521c231249af6006ad4095b41f05088f59b0d0ccf1aea1a51cf4b`. 이전 Rescan 설정 전체, 사용 이력, 선택 순서, 전처리 규칙과 기존 소스 6개의 SHA256가 포함된다.
- **기존 바닥 기준 3.7857808 / 3.6151132 / 3.4235026cm까지 그대로 전달**했다. 새 촬영에서 바닥 P95를 다시 추정하거나 결과를 보고 기준을 조정하지 않았다. 표면 sample 간격 3.5cm, 코어 간격 8cm, 반경 8/16/28cm, 최소 차이 6cm, 두 크기 이상 초과, source 지지 4.5cm, 연결 반경 16cm, 최소 9점·0.04m²도 그대로다.
- `all_surface_change.py`, `structural_reference.py`, `compare_mesh_methods.py`, `build_m3c2_independent.py`, `run_all_surface_comparison.py`, `prepare_bonn_pair.py`는 고정 시점과 hash가 같다. 새 adapter는 기존의 `clearance_measurements`, `classify_clearance`, `candidate_components`, 구조면 추출·대응, raw-depth vote와 사진 연결 함수를 호출한다. 실행 시작과 prediction 기록 전에 소스 hash를 검사하며 변경되면 중단한다.
- 새로 작성한 `run_frozen_indoor_transfer.py`는 입력·좌표·공개 파일 형식을 연결한다. 기존 Bonn RGB-D 복원 adapter와 공개 카메라 보정, TSDF 1.5cm / truncation 6cm를 재사용했다. Rescan의 Poisson 복원 자체를 새로운 RGB-D에 적용했다는 주장은 하지 않는다.
- 시간 구간은 **처음/마지막 20%, 촬영 양 끝에서 0.5초 제외**라는 사전 규칙으로 자동 선택했다. 실제 전 0.50–4.82596초 / 후 19.30382–23.62978초, 각 26 RGB-D 프레임이다. 탐지 결과를 보고 구간을 고르지 않았다.
- 객체·instance·변화 정답 파일은 탐지에 사용하지 않았다. Bonn의 `groundtruth.txt`는 **카메라 위치·자세**이므로 좌표 복원에 사용한다. 이것을 변화 정답 라벨과 혼동하지 않는다.

원본 [공식 ZIP](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_placing_nonobstructing_box.zip)은 400,775,291 bytes, SHA256 `8eda7eabf1f71217d202d12f438201146a1488c1eb3664e21f490701934b9bcc`다. `data/bonn/placing_nonobstructing_box/`에 입력과 source manifest를 보존했다.

### 고정 설정의 첫 실행 결과

실제 실행은 `outputs/frozen-transfer-bonn-placement-20261016/run-v1/` 한 번이다. 결과를 본 뒤 검출 규칙을 수정하거나 재실행하지 않았다.

| 항목 | 결과 |
|---|---:|
| TSDF 삼각형 전 / 후 | 356,172 / 200,859 |
| 전체 공유 코어 | 5,662 |
| 대응 구조면 | 바닥 1 + 수직 구조 2 |
| 후보에서 제외한 구조면 코어 | 2,046 |
| 구조면 밖 코어 중 비교 가능 | 3,472 / 3,616 |
| 최종 연결 후보 | 6패치, 1,367코어 |
| 전후 사진이 모두 연결된 후보 | 3 / 6패치 |
| depth의 자기 시점 존재 / 반대 시점 빈 공간 지지 후보 | 78코어 |

후보별 결과는 다음과 같다. 전후 면적은 삼각형을 가까운 공유 코어에 배분한 근사 표면 면적이다.

| 후보 | 코어 | 후보 면적 전 / 후(m²) | 전후 사진 수 | depth 존재/빈 공간 지지 코어 |
|---|---:|---:|---|---:|
| 01 | 91 | 0.000 / 0.496 | 4 / 4 | 68 |
| 02 | 260 | 1.107 / 0.000 | 4 / 0 | 0 |
| 03 | 799 | 4.591 / 0.460 | 4 / 0 | 10 |
| 04 | 190 | 0.755 / 0.147 | 3 / 1 | 0 |
| 05 | 17 | 0.025 / 0.064 | 4 / 4 | 0 |
| 06 | 10 | 0.099 / 0.009 | 3 / 0 | 0 |

**후보 01은 원본 사진의 새 상자 위치에 잡혔다.** 탐지 후 원본 전 프레임 130(4.36초)의 빈 바닥과 후 프레임 605(20.27초)의 놓인 상자를 확인했다. 해당 위치의 raw-depth 지지는 68/91점이다. 알고리즘이 객체 종류를 상자로 인식하거나 모든 후보를 실제 추가/소실로 분류했다는 뜻은 아니다.

02/03/06에는 이후 사진이 연결되지 않았고, 새 맵의 관측 범위 밖 또는 가림/복원 차이가 포함될 수 있다. 04/05도 전후 사진이 있다는 사실만으로 실제 변화를 보장하지 않는다. 모든 6패치를 그대로 제공하며, 불리한 후보를 숨기거나 다른 사진으로 채우지 않았다. **6패치는 정답 변화 6개가 아니고, 이번 실행은 precision/recall이나 과적합 부재를 증명하지 않는다.**

왼쪽 지도는 원본 메쉬, 오른쪽 지도는 같은 메쉬에 후보를 표시한 결과다. 이전/이후 버튼으로 두 맵을 함께 전환하며 전후 원본 사진은 나란히 유지한다. 처음 열 때 전후 사진이 있는 첫 후보를 선택하지만 전체 후보 목록은 그대로다. raw PNG 원본은 보존하며, 화면용 JPEG는 기존 adapter의 변경 없는 변환을 사용했다. 사진 상자는 제공 카메라 포즈 및 보정된 depth로 연결하고 원본 RGB 왜곡을 반영한다.

재현:

```powershell
$env:OMP_NUM_THREADS='1'
python tools/run_frozen_indoor_transfer.py --data data/bonn/placing_nonobstructing_box/rgbd_bonn_placing_nonobstructing_box --protocol tools/frozen-bonn-placement-protocol.json --output outputs/frozen-transfer-bonn-placement-new-run
```

현재 산출물: `frozen-protocol.json`, `acquisition-selection.json`, `structural-extraction.json`, `selected-originals.json`, `prediction.npz`, `comparison.json`, `published/`. 다운로드 전에 고정한 protocol 원문은 배포용으로 `tools/frozen-bonn-placement-protocol.json`에도 보존한다. 공개 JSON에 protocol/prediction SHA256를 남겼고 raw depth/NPZ/protocol 파일은 웹 라우트에서 차단했다. 52개 입력 RGB hash와 공개용 JPEG 35장의 동일 복사, 고정 소스/설정 일치, 이전 3개 결과 API의 byte 일치를 확인했다. 도구 63 / 뷰어 Python 19 / Node 60 / 공통 Python 19 = **161개 통과, 건너뜀 없음**, fixture 검사도 통과했다. 감사 결과·테스트 로그·실제 지도/사진 화면은 `outputs/frozen-transfer-bonn-placement-20261016/`에 있다.

## 이전 수정: 바닥 기준 유지, 가구는 모든 방향의 표면 비교

Rescan 선택 화면은 `outputs/all-surfaces-rescan-v2-20261016/published`를 사용한다. 이전 구조면 실험과 수동 ROI 실험은 아래에 보존했다. 실제 입력은 동일한 Rescan `scene_e_0 → scene_e_2`이며, 정합·복원·카메라를 다시 최적화하지 않았다.

### 변경한 계산

- 이전 공통 바닥 오차 모델의 P95 **3.786 / 3.615 / 3.424cm**를 유지한다. 바닥 1면과 수직 구조면 3면의 배경 마스크도 그대로다. 이전에 벽이라고 표시했던 수직 면에는 창문이 포함되므로 현재 UI는 수직 구조면이라고 부른다. 객체 라벨로 배경 마스크를 수정하지 않았다.
- 각 query의 원래 국소 법선을 복원하고, **그 query 자체의 법선**과 50° 이내인 같은 시점의 이웃을 모은다. 바닥과 평행한 가구 표면만 허용하던 전역 방향 제한은 없다. 옆면·등받이·기울어진 면도 비교한다.
- 상대 mesh에 평행한 표면층이 없는 소실·측면 이동을 놓치지 않도록, 각 원본 표면 sample에서 상대 mesh의 실제 삼각형까지의 최단 거리를 전후 양방향으로 계산한다. 반경 8/16/28cm 이웃별 중앙값을 사용한다. **이번 새 경로는 정식 M3C2의 signed 법선 거리/LoD만으로 판정하는 경로가 아니며, 3D 표면 거리로 확장한 기하학적 후보 검출기다.**
- query의 원래 표면을 10cm 안에서 국소 평면 적합하고 `max(1.4826*MAD, (P90-P10)/2.563)`로 요철을 추정한다. 판정 기준은 `max(바닥 P95, 1.96*sqrt(2)*국소 요철)`이다. 두 분산을 더하지 않고 mesh 표본 수의 제곱근으로 줄이지도 않는다. 바닥 P95를 다른 모든 방향에 대해 검증된 공분산이나 95% 신뢰도로 해석하지 않는다.
- 같은 시점의 원본 점군/mesh에서 4.5cm 이내 지지, 반경별 이웃 6점 이상, 두 반경 이상에서 **6cm 및 노이즈 기준 동시 초과**, 16cm 연결 반경에서 9점/0.04m² 이상을 요구한다. 원본 점군의 instance/class 값은 이 과정에 넣지 않는다. 중앙값·다중 크기·국소 요철·원본 지지·연결성으로 작은 재구성 차이와 고립점을 걸러낸다.
- 두 시점의 depth가 모두 같은 위치를 occupied로 지지해야 하는 조건도 제거했다. 객체가 이동/소실하면 그 조건 자체가 성립하지 않을 수 있기 때문이다. 대신 자기 시점 occupied와 상대 시점 free 관측을 후보 정렬 점수에 기록한다. 가림/결측을 추가·소실 확정으로 바꾸지 않는다. 점수는 순위용 휴리스틱이며 확률이 아니다.

이전→수정 비교는 위 조건들이 함께 바뀐 **구현 버전 비교**다. 방향 조건만 바꾼 통제 실험이나 M3C2 단독 성능 비교로 표현하지 않는다.

### 탐지 후 기존 12개와 대조

설정은 `frozen-config.json`에 저장하고 `prediction.npz`를 먼저 완성한 뒤 `changes.json`의 12개 항목을 읽는다. 기존 항목의 source instance ID는 **평가와 사진 표시만** 담당한다. 각 query가 해당 시점의 실제 instance 표면에서 4cm 이내로 지지되는지 확인하므로, 인접한 객체의 상자가 겹친다는 이유로 포착 수에 넣지 않는다.

포착은 해당 가구의 원래 표면에 **9점 이상 및 0.04m² 이상의 연결 후보가 한 곳 이상** 존재하는 경우다. 일부 점만 있으면 일부 포착, 없으면 미포착이다. 이동 전후 위치를 모두 완전히 복원하거나 종류를 자동으로 맞혔다는 기준이 아니다. 임계값은 이전 연결성 규칙에서 유지했으며 12개 결과에 맞추어 조절하지 않았다.

| 기존 항목 | 기존 종류 | 이전 포착 | 수정 포착 | 수정 후보 / 대상 표면점 |
|---|---|---|---|---:|
| 테이블 #9 | 이동 | 포착 | 포착 | 235 / 320 (73.4%) |
| 테이블 #10 | 소실 | 미포착 | 포착 | 88 / 141 (62.4%) |
| 기타 가구 #11 | 이동 | 미포착 | 포착 | 340 / 392 (86.7%) |
| 기타 가구 #13 | 이동 | 미포착 | 포착 | 214 / 292 (73.3%) |
| 기타 가구 #18 | 이동 | 미포착 | 포착 | 299 / 458 (65.3%) |
| 의자 #19 | 이동 | 포착 | 포착 | 452 / 602 (75.1%) |
| 의자 #20 | 이동 | 미포착 | 포착 | 325 / 568 (57.2%) |
| 의자 #21 | 이동 | 포착 | 포착 | 417 / 579 (72.0%) |
| 소파 #25 | 이동 | 미포착 | 포착 | 1,846 / 2,635 (70.1%) |
| 기타 가구 #27 | 이동 | 미포착 | 포착 | 238 / 250 (95.2%) |
| 의자 A | 추가 | 포착 | 포착 | 264 / 285 (92.6%) |
| 의자 B | 추가 | 포착 | 포착 | 265 / 273 (97.1%) |

**이전 5/12 → 수정 12/12 위치 포착.** 구조면 밖 11,923점 중 비교 가능 표본은 529→11,763점, 최종 후보 표본은 106→8,852점이다. 새 후보는 연결 패치 10개이며 한 패치에 여러 가구가 연결될 수 있다. 따라서 10패치를 10객체로 세지 않는다.

이 결과는 광범위한 후보 검출이다. **8,852점 중 3,869점은 기존 12개 항목 밖**이다. 이 점들을 모두 정답이나 모두 오탐으로 분류할 근거는 없으며, 새 추가 후보와 관측 차이/재구성 오탐이 섞일 수 있다. 이 중 raw depth의 occupied/free 지지까지 있는 후보는 1,140점이다. 전후 mesh 형상의 차이만으로 실제 물체 변화인지 확정할 수 없다. 기존 12개도 추정된 이동/객체 연결을 포함하므로 완전한 정답셋이나 전체 precision 측정으로 해석하지 않는다. **12/12는 정확도 100%가 아니다.**

### 화면과 산출물

- 왼쪽은 이전 구조면 실험의 공개 field binary를 그대로 사용하고, 오른쪽은 모든 방향의 새 3D 표면 비교를 사용한다. 같은 mesh에 두 판정을 덧씌우며 시점은 동기화한다.
- 기본 목록은 12개 항목이며, 각 항목의 이전/수정 포착·면적·연결점 수·표면 비율과 원본 정지 사진을 표시한다. 아래 결과 표에서 항목을 누르면 해당 맵/사진으로 이동한다. 전체 연결 후보와 구조 기준면도 별도 목록으로 볼 수 있다. 기존 종류는 참고 라벨로 명시하며 새 자동 종류 판정처럼 표시하지 않는다.
- 12개 모두 전후 원본 사진이 연결된다. 원본 JPEG 28장은 byte 동일하게 복사했다. 소파 #25 이후와 기타 가구 #27 이전은 기존 depth 기반 사진 연결만 있어 **3D 연결 검토 필요**라고 표시하고 검증되지 않은 투영 상자는 그리지 않는다. 다른 시점의 사진으로 채우지 않았다.
- `frozen-config.json`, `prediction.npz`, `reference-evaluation.json`, `comparison.json`, `published/`를 새 디렉터리에 기록한다. prediction과 evaluation SHA256를 공개 JSON에 남긴다. private NPZ/별도 평가 파일은 웹 라우트로 노출하지 않는다. 기존 main, 기본 Rescan/Bonn 화면 산출물은 그대로다.

재현:

```powershell
$env:OMP_NUM_THREADS='1'
python tools/run_all_surface_comparison.py --comparison-dir outputs/mesh-comparison-final2-20261015 --review-dir outputs/rescan-three-changes --frames-dir outputs/m3c2-frame-recovery-20261015 --previous-dir outputs/structural-reference-rescan-v5-20261016 --normal-reference-dir outputs/residual-roi-rescan-v3-20261016 --output outputs/all-surfaces-rescan-new-run
```

검사: 도구 Python 61 / 뷰어 Python 18 / Node 58 / 공통 Python 19 = **156개 통과, 건너뜀 없음**. 회전된 면의 동일 판정, 작은 노이즈·고립점 제거, 국소 요철·원본 지지 부족, 라벨과 공간 상자 혼동 방지, 평가 종류가 탐지에 영향을 주지 않음을 확인했다. fixture 검사와 `git diff --check`도 통과했다. 실제 출력에서 원본 mesh/이전 field/28개 JPEG byte 동일, 배경 8,920점의 후보 제외, 12개 전후 사진의 브라우저 로드를 확인했다. 감사 결과와 화면은 `outputs/all-surfaces-qa-20261016/`에 기록했다.

## 이전 구현: 작은 ROI 대신 공통 구조면

이전 화면은 `outputs/structural-reference-rescan-v5-20261016/published`를 사용했다. 아래 수동 ROI 실험도 이전 실패 기록이며 현재 판정 경로가 아니다.

- 각 시점의 면적 가중 mesh 표본에서 큰 수평/수직 평면을 추출하고, 바닥 높이·크기·법선·겹치는 범위로 전후 면을 대응한다. 객체/변화 라벨이나 기존 객체 범위를 읽지 않는다. 이번 실제 쌍은 **바닥 1면과 벽 3면**이 대응됐다. RANSAC에서 노이즈로 나뉜 낮은 바닥 조각은 합치되 책상/의자 상판은 바닥에 합치지 않는다.
- 기준 계산은 두 시점에 공통으로 지지되고 원본 depth 관측이 있는 면 표본만 사용한다. 후보 제외는 이미 대응된 구조면 위에 실제로 놓인 표본에 적용한다. 반대 시점에서 가구에 가려진 바닥도 바닥 자체의 변화로 만들지 않는다. 바닥 상자 안 전체를 지우지 않으며 표면 법선 및 가까운 구조면 표본과의 법선 방향 거리 2.5cm 이내 조건으로 낮은 가구 받침까지 지우는 것을 막는다.
- 기준과 후보의 원통 이웃은 같은 방향의 표면만 사용한다. 기준은 해당 평면의 표본으로 더 제한하므로 바닥-의자 다리-상판이 한 층으로 연결되지 않는다. 반경 8/16/28cm는 유지한다.
- 면별로 기준 표본의 signed 거리에서 위치 `(u,v)`에 따른 1차 편차를 추정한다. 맵·pose를 움직이는 정합이 아니라, 안정면에서 기대되는 잔차를 모델링하는 것이다. 모든 추정은 fit 영역만 사용한다. 임의의 고정 오차 대신 1.5m 공간 블록을 제외해 예측한 잔차의 공간 균형 P95를 사용한다. 각 제외 블록 주변 56cm도 학습에서 뺀다. 단순 fit 잔차보다 공간 예측 오차를 과소평가하지 않도록 직접 fit P95와 그중 큰 값을 취한다.
- 최종 확인 구간은 별도 1.5m 띠로 나누고 추정/확인 중심 사이 60cm 간격을 둔다. **실제 면 표본 교집합은 전후/반경별 모두 0**이다. 확인 구간 잔차는 편차식이나 P95 추정에 쓰지 않는다.
- 후보 기준은 `max(면별 예측 잔차 P95, 국소 robust spread envelope)`이며 두 분산을 다시 더하지 않는다. 기존 비교군과 동일하게 최소 변위 6cm, 같은 방향으로 두 반경 이상 초과를 요구한다. 이웃 반경 16cm에서 9점 이상 연결되고 평균 전후 면적 0.04m² 이상인 **표면 패치**만 후보로 공개한다. 전체 객체가 바뀌었다거나 추가/소실/이동 종류를 확정한 결과는 아니다.

실제 결과:

| 항목 | 결과 |
|---|---|
| 공통 면 | 바닥 1 / 벽 3 |
| 후보에서 제외한 구조면 | 8,920코어, 이전 45.911 / 이후 21.863m² |
| 오차 추정에 쓸 수 있는 전후 공통 구조면 코어 | 3,563; 이 중 depth/공간 분할을 통과한 표본으로 적합/확인 |
| 바닥 fit / 별도 check | 296 / 156코어 |
| 바닥 예측 잔차 기준 P95 (8/16/28cm) | 3.786 / 3.615 / 3.424cm |
| 바닥 check 기준 초과 | 10/156 = 6.41% |
| 벽의 기준 적용 | 표본/분리 구간 부족으로 3면 모두 보류. 대응된 벽 표면의 후보 제외는 유지 |
| 기존 고정 2cm 표면 초과 / 수정 후 연결 후보 표면 | 640 / 106코어 |
| 최종 후보 | 5개 연결 표면 패치, 전후 원본 JPEG 29장 중 위치별 최대 각 4장 연결 |

기존 640과 새 106은 이웃 오염 방지·배경 제외·공간 오차 모델·연결성 조건이 함께 바뀐 결과다. 감소량을 정답 기반 오탐 감소율로 해석하지 않는다. 예전 수동 ROI check 38개와 새 check 156개도 다른 표본 집합이며 76.3%→6.4%를 같은 평가셋의 성능 향상으로 표현하지 않는다. 전체의 유효율을 분모로 판정한 요약 상태는 여전히 관측 부족이고, 지원되는 5개 국소 표면만 후보로 보여준다. 안정 구조면이라는 가정 자체는 자동 semantic 정답이 아니다.

재현:

```powershell
$env:OMP_NUM_THREADS='1'
python tools/run_structural_comparison.py --comparison-dir outputs/mesh-comparison-final2-20261015 --review-dir outputs/rescan-three-changes --frames-dir outputs/m3c2-frame-recovery-20261015 --previous-dir outputs/residual-roi-rescan-v3-20261016 --output outputs/structural-reference-rescan-new-run
```

추가 파일은 `tools/structural_reference.py`, `tools/run_structural_comparison.py`, `tools/tests/test_structural_reference.py`다. 기존 화면 JS/HTML의 별도 실험 모드와 server의 선택적 결과 경로만 갱신했다. 산출물에는 `plane-extraction.json`, 면별 `*-reference.npz`, `reference-models.json`, `fit-check-support-audit.json`, 최종 `measurements.npz`와 공개용 `published`를 분리했다. 원본 mesh/카메라, 기존 Rescan/Bonn 결과, main/origin/main은 유지했다.

최종 확인: 도구 Python 50, 뷰어 Python 18, Node 56, 공통 Python 19 = **143개 통과(건너뜀 없음)**. fixture 규약 검사와 `git diff --check`도 통과했다. 예전 check 38개는 모두 배경으로 제외되고 새 후보에 바닥 높이 표본이 없음을 실제 NPZ에서 확인했다. 같은 38개를 면 제한 후 다시 측정한 큰 반경의 원시 P95는 2.473cm였다. 전후 사진·후보 5곳·binary 길이·실제 표면 표본 독립성, 기본 Rescan/Bonn API의 원본 바이트 일치를 확인했다. 브라우저에서 기준면/후보 선택, 이전/이후 메쉬, 사진 탐색/확대 및 console 오류 없음도 확인했다. 화면 증거는 `outputs/structural-reference-qa-20261016/maps-and-originals-final.png`, `candidate-after-final-full.png`, `candidate-before-full.png`, `common-floor-full.png`이다.

## 이전 실패의 재현과 원인

보관된 `outputs/reverted-error-baseline-20261015/new-files/tools/tests/test_stable_surface_baseline.py`의 공통 8cm 편차 검사를 다시 실행했다. **변화가 없는 공통 편차인데 전 지점이 변화로 판정**되는 실패가 재현됐다. 이전 기준은 약 `0.07999961m`, 판정 거리는 `0.08000000m`였다. 자동 기준 추정의 평균 법선과 국소 판정의 법선/이웃 연산이 달라 약 0.4μm의 차이가 전 지점 초과를 만들었다. 임의의 1mm 여유를 더해 검사를 통과시키는 수정은 하지 않았다.

이번에는 `mesh_change_noise.surface_pair_measurements` 하나를 기준/확인/대상 모두에 사용한다. 반경 8/16/28cm, 축 방향 반길이 65cm, 지지된 가장 가까운 표면층의 법선 방향 중앙값이라는 계산이 동일하다. 법선도 동일한 사전 계산 배열에서 읽는다. 각 크기별로 같은 측정값 분포와 비교하며, 등호 주변에는 16 ULP 또는 1pm 중 큰 부동소수점 수치 허용치만 둔다. 물리적 오차를 덮는 추가 여유가 아니다.

이전 전역 자동 배경 기준은 다른 위치/방향의 재구성 차이를 한 P95로 묶어 Rescan에서 약 14cm가 됐다. 그것이 순수 정합 오차이거나 모든 객체에 전달 가능한 기준이라는 근거는 없었다. 양쪽에서 표면이 관측된다는 사실만으로 해당 표면이 변하지 않았다고 보장되지도 않는다. 이번에는 자동 저잔차 선택 대신 명시적 수동 ROI와 원본 사진, 위치/형상/요철/밀도 조건, 별도 확인 영역을 기록한다.

## 입력과 좌표

- 실제 Rescan `scene_e_0 → scene_e_2`의 기존 복원/정합 캐시: `outputs/rescan-three-changes/cache/{0,1}-depth8.ply`.
- 공유 코어/원래 지도: `outputs/mesh-comparison-final2-20261015`.
- 기존 RGB/depth 카메라: `outputs/rescan-three-changes/review.json`, `outputs/m3c2-frame-recovery-20261015/cameras.json`.
- 입력 메쉬 SHA256와 코어 SHA256, 카메라/리뷰 metadata/config SHA256를 결과에 기록한다.
- **새로운 ICP, 객체별 정합, pose 보정은 하지 않는다.** 원래 정합 캐시와 제공된 추정 카메라를 그대로 읽는다. 원본 사진 연결은 기존 mesh raycast/project 함수를 사용한다.
- ROI 좌표는 `PLY 세계 좌표 @ comparison.display_rotation.T`이며, `display_origin`을 빼기 전 절대 좌표다. 높이는 +Z, 단위는 m. 화면 좌표는 여기서 `display_origin`을 뺀다. 원본 카메라로 갈 때 기존 회전을 역으로 적용한다.

## 수동 ROI와 독립성

정의 파일은 `tools/residual-roi-rescan.json`이다. 아래 footprint는 **새 잔차 값을 확인하기 전에** 원본 RGB와 양쪽 depth 관측 범위로 정했다. 카펫 바닥이 안정적이라는 검토 가정을 사용하며, 변화 정답 라벨이 아니다. 높이 범위는 모두 -0.12~0.22m이고, 실제 적합에는 법선/평면성/양쪽 depth 조건을 추가한다.

| ROI | 역할 | X 범위(m) | Y 범위(m) | 실제 상태 |
|---|---|---|---|---|
| floor-east-fit | 중앙 바닥 기준 추정 | 0.55~1.20 | 0.60~1.20 | 95코어 중 조건/관측 충족 72코어, 전후 사진 연결 |
| floor-east-check | 중앙 바닥 독립 확인 | 1.85~2.45 | 0.60~1.20 | 101코어 중 비교 가능한 양쪽 관측 38코어 |
| floor-west-fit | 서쪽 바닥 기준 추정 | -3.60~-3.00 | -0.60~0.00 | 38코어 측정 가능, 이후 RGB 연결 부재로 기준 미승인 |
| floor-west-check | 서쪽 바닥 독립 확인 | -1.80~-1.20 | 0.60~1.20 | 서쪽 기준 미승인 상태로 전달하지 않음 |

기준/확인 query footprint의 반경 버퍼를 검사한다. 기울어진 원통 이웃까지 고려하기 위해 실제로 선택된 표면 sample ID도 추적한다. 최종 실행에서 **모든 fit/check 조합 × 두 epoch × 세 반경의 공유 sample 수는 0**이다. `fit-check-support-audit.json`에 기록한다. 확인 영역은 어떤 기준의 적합에도 들어가지 않는다.

중앙 기준 사진은 전 `frame-0-000690`(대안 1560/1545/1530), 후 `frame-1-000000`(대안 15/30/150) 등에 연결했다. 중앙 확인은 전 1500/1515/1530/1545, 후 180/195/210/2220이다. 사진은 기존의 추정 카메라를 사용하며 정확한 GT 대응을 주장하지 않는다. 서쪽의 사진 누락을 다른 위치 사진으로 대신 채우지 않는다.

## 판정식과 적용 조건

각 반경에서 표면 차이 `d = median(layer_after) - median(layer_before)`를 계산한다. 표면 요철은 기존 `max(1.4826*MAD, (P90-P10)/2.563)`이고, 메쉬 표본 수로 나누지 않는다.

경험적 기준 `T_r`는 fit ROI의 **개별 |d_r| 표본**에 대한 공간 균형 P95다. 16cm 공간 블록마다 같은 총 가중치를 주며, 블록별 중앙값으로 분포를 먼저 좁히지 않는다. 낮은 거리만 골라 적합하거나 큰 잔차를 보고 잘라내지 않는다.

국소 적용에는 다음 조건을 요구한다.

- 기존 정상 방향/크기로 계산한 거리. 바닥 방향 ±20°, 양쪽 국소 평면성 및 법선 일치 조건.
- fit ROI와 비슷한 표면 요철/지지 표본 수 범위(기록한 분위수와 0.5~2배 조건).
- 기준 중심에서 2.5m 이내, 법선 축 높이 차이 25cm 이내. 복잡한 의자나 벽에 바닥 기준을 무조건 적용하지 않는다.
- 실제 깊이에서 각 시점 표면 위치가 두 번 이상 occupied로 지지. 가림/결측은 유효 비교가 아니다.
- 기준 최소 20코어/6공간 블록. 확인 영역 최소 10유효 코어, 전체 ROI 대비 유효율 50% 이상, 같은 방향의 2개 이상 크기에서 잠정 기준을 넘는 비율 20% 이하. 이 20%는 명시적인 초기 승인 규칙이며 95% 신뢰도 주장이 아니다.

승인된 기준에서 `|d_r| > T_r + numerical_tolerance`가 같은 방향으로 두 반경 이상 유지돼야 초과 표본이다. **잔차 기준을 M3C2 registration sigma에 넣지 않는다. 국소 분산을 다시 더하지도, 전체 RMSE를 빼지도 않는다.** P95 자체가 이 측정 조건의 정합/재구성 잔차를 함께 포함한다.

초기 적용 조건의 요철/밀도 특징은 세 반경의 중앙값(요철은 전후 중 큰 값)으로 요약한다. 반경별 조건을 완전히 따로 보정한 모델은 아니다. 고정 비교군에는 기존 6cm 최소 변위 규칙도 유지하며, 경험적 방식은 승인된 측정 P95를 직접 기준으로 삼는다. 따라서 이 실험은 완전히 동일한 최종 객체 분류기의 오차 항 하나만 바꾼 성능 비교가 아니다.

공간 패치는 객체 마스크 없이 80cm 격자로 만든다. 제공된 객체 범위 12개도 별도 집계하지만 라벨을 변화 정답으로 쓰지 않는다. `changes.json`이 없어도 국소 패치 경로는 실행된다. 영역은 유효 비교 50% 이상, 유효 표본 중 초과 20% 이상, 연결 반경 16cm에서 최소 9코어, 초과 평균 epoch 면적 0.04m² 이상이어야 변화 후보가 된다. 한 점만으로 객체를 변화로 확정하지 않는다.

출력은 변화 후보 / 변화 근거 부족 / 관측 또는 기준 부족으로 판단 불가다. 미관측/대응 부재를 추가·소실로 변환하지 않는다. 단위면적은 각 삼각형 면적을 가까운 공유 코어의 꼭짓점 몫으로 분배한 근사이며 전후 값을 별도 보고한다.

## 실제 실행 결과

**현재 수동 기준 두 개 모두 승인되지 않았다.** 따라서 새 경험적 방식은 이번 입력의 변화 후보를 확정하지 않고 보류한다. 0점/0m²는 변화가 없다는 결론이나 오탐 감소 성공이 아니다.

| 중앙 바닥 | 반경 8cm | 반경 16cm | 반경 28cm |
|---|---:|---:|---:|
| fit 공간 균형 P95 | 1.436cm | 1.524cm | 1.322cm |
| 독립 check 원시 표본 P95 | 3.040cm | 2.747cm | 47.530cm |

중앙 check는 38/101코어(37.6%)가 유효하며 그중 29/38(76.3%)가 잠정 기준을 초과한다. 관측 비율과 전달성 모두 승인 규칙을 만족하지 않는다. 특히 큰 반경의 먼 표면층 혼입/복원 누락 가능성을 포함해 원인을 따로 검토해야 한다. 이 check 잔차를 적합에 다시 넣어 기준을 높이지 않았다.

서쪽 fit의 계산된 P95는 약 0.39~0.54cm지만 이후 원본 사진으로 검토할 수 없어 승인하지 않았다. 한쪽 RGB 누락을 깊이 관측만으로 덮지 않았다.

| 전체 기준/확인 ROI를 제외한 20,507코어 | 고정 2cm 표면 비교군 | 경험적 기준 |
|---|---:|---:|
| 유효 비교 코어 | 2,470 (12.0%) | 0 (승인 기준 부재) |
| 두 반경 이상 초과 코어 | 639 | 판정 보류 |
| 초과 면적 이전/이후 | 1.737 / 1.789m² | 판정 보류 |
| 최대 연결 초과 코어 | 97 | 판정 보류 |

이 고정 비교군은 **동일 거리 연산·분석 크기·양쪽 관측 조건 위에서 기존 2cm 오차 식을 적용한 표면 단계 비교**다. 기존 뷰어의 전체 객체 분류 파이프라인을 다시 실행한 이벤트 개수가 아니다. 기본 뷰어의 Rescan 새 항목 0 / Bonn 상자 소실 1 결과는 원래 파일 그대로 유지한다.

144개 국소 패치와 제공된 객체 범위 12개, 기준/확인 ROI 4개를 추적할 수 있다. 새 방식의 144패치는 모두 판단 불가다. 화면의 `거리 / 기준`은 **미승인 기준의 잠정 비율을 참고로만** 표시한다. 이것을 새 변화 탐지 결과로 부르지 않는다.

## 재현과 파일

```powershell
python tools/run_residual_roi_comparison.py --comparison-dir outputs/mesh-comparison-final2-20261015 --review-dir outputs/rescan-three-changes --frames-dir outputs/m3c2-frame-recovery-20261015 --output outputs/residual-roi-rescan-new-run
```

계산 의존성은 기존 `tools/mesh-comparison-requirements.txt`를 사용한다. 기존 출력 경로를 덮어쓰지 않는다. 최종 실제 출력은 `outputs/residual-roi-rescan-v3-20261016/`이다.

- `roi-selection.json`: 설정·좌표·단위·원본 연결.
- `fit-check-support-audit.json`: 기준/확인 실제 표면 sample 교집합.
- `comparison.json`: 기준 분포·P95·조건·확인 실패·구역별 면적/표본/연결성/보류 사유.
- `measurements.npz`: 공유 코어별 위치/법선/크기별 거리/요철/표본 수/관측/실제 및 잠정 기준/판정.
- `samples.csv`: 코어별 좌표·역할·크기별 거리와 기준·보류 사유.
- `published/`: 공개할 mesh/field binary, comparison JSON, 기존 원본 JPEG 29장. raw depth/npz/CSV는 웹 라우트로 공개하지 않는다.

새 페이지에는 두 지도 시점 동기화, 전후 메쉬 선택, ROI·객체·국소 패치 필터, 원본 사진 앞뒤 탐색/확대, 분포/확인 비교, 결과 JSON 다운로드가 있다. 기본 페이지 HTML/JS와 기본 산출물은 바꾸지 않았다.

## 확인

도구 Python 39 / 뷰어 Python 18 / Node 55 / 공통 Python 19 = **131개 테스트 통과**, 공통 fixture 규약 검사도 통과했다. 합성 검사는 공통 8cm 편차, 국소 큰 변화, 불일치 형상/밀도/요철, 관측 결측, fit/check 누출, 확인 실패 시 보류, 한 점의 객체 판정 금지를 확인하는 구현 회귀 검사다. 실제 재촬영의 precision/recall 증명이 아니다.

기본 Rescan/Bonn API의 원래 JSON 파일 바이트 일치와 별도 라우트의 private 파일 차단도 확인했다. 실제 브라우저에서 두 WebGL 지도, 메쉬 시점 전환, 잠정 비율의 참고 표시, 원본 사진 이동, 빈 후보 목록의 오래된 사진 제거를 확인했고 console 오류는 없었다.

화면 증거는 `outputs/residual-roi-qa-20261016/independent-check-full.png`와 `maps-and-originals.png`다. 기존 뷰어 탭은 유지하고 실험을 별도 탭에 열었다.

변경 파일:

- `tools/mesh_change_noise.py`: 거리 연산 추출/공유, 고정 2cm 기본 규칙 보존, 실제 표면 sample 지지 집합 감사.
- `tools/empirical_residual.py`: 명시적 ROI 적합, 공간 균형 P95, 조건별 적용, 독립 확인/보류, 영역 판정.
- `tools/residual-roi-rescan.json`: 좌표·단위·수동 ROI·승인/영역 조건.
- `tools/run_residual_roi_comparison.py`: 실제 데이터 실행/구역 집계/원본 연결/공개 묶음/CSV·NPZ.
- `apps/autotour360/residual-comparison.{html,js,css}`, `residual-context.js`: 별도 실험 화면.
- `apps/autotour360/server.py`: 실험 전용 경로와 선택적 `--residual-dir` 추가. 기존 기본 경로 유지.
- `tools/tests/test_empirical_residual.py`, `apps/autotour360/tests/residual-context.test.mjs`, `apps/autotour360/tests/test_server.py`: 새 회귀/격리 검사.
