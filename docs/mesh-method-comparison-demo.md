# M3C2 / GLS core 실제 메쉬 비교 데모

2026-10-15. `task5-a`의 별도 실험 뷰어. 실제 Rescan `scene_e_0 → scene_e_2`의 기존 복원 메쉬를 입력으로 사용한다. **M3C2/GLS 계산은 객체 라벨, instance ID, 기존 변화 마스크를 읽지 않는다.** 사용자가 무엇이 바뀌었는지 이해할 수 있도록 화면에서는 기존 전후 비교의 추가·소실·이동 추정과 사진 연결을 맥락 정보로 함께 읽는다. 이 변화 종류를 M3C2/GLS가 직접 분류했다고 표시하지 않는다. 기존 전후 객체·사진 뷰어도 유지한다.

## 실행

계산용 별도 Python 환경에 `tools/mesh-comparison-requirements.txt`를 설치한다. 공통 scaffold 환경이나 앱 서버는 Open3D를 요구하지 않는다.

```powershell
python tools/compare_mesh_methods.py `
  --before outputs/rescan-three-changes/cache/0-depth8.ply `
  --after outputs/rescan-three-changes/cache/1-depth8.ply `
  --output outputs/mesh-comparison-final2-20261015
python apps/autotour360/server.py --port 8768 `
  --rescan-dir outputs/rescan-three-changes `
  --mesh-comparison-dir outputs/mesh-comparison-final2-20261015
```

위 명령은 저장소 루트 기준이다. 출력 폴더가 이미 차 있으면 새 경로를 사용한다. 기존 계산이나 입력을 덮어쓰지 않는다. 서버가 실행 중이면 필요한 경로 인자로 재시작한다. URL은 `http://127.0.0.1:8768/mesh-comparison.html`이다. 실제 데이터/생성 binary는 Git에 포함하지 않는다. 현재 PC의 원본은 기존 `data/rescan/scene_e_0-2/`이며, [공식 데이터 출처](https://rescan.cs.princeton.edu/)는 이전 데모와 같다.

## 동일 입력과 공통 좌표

- 이미 공통 좌표계로 정합된 캐시 PLY 두 개를 읽는다. after 정합 변환을 중복 적용하지 않는다.
- 삼각형 면적 비례 barycentric sampling 후 3.5cm voxel당 한 표본을 남긴다. 정점 개수나 삼각형 번호의 차이를 변화로 비교하지 않는다. 고정 난수 seed는 `20261015 + epoch`다.
- 이전 80,545 / 이후 48,621 표면 표본, 공통 코어 20,843곳이다. 두 검출 결과를 같은 코어에서 평가하고 화면 메쉬 정점에 최근접 전달한다.
- 인스턴스 기반 마스크, 객체 매칭, 기존 변화 결과를 검출이나 구역 선택에 사용하지 않는다. 입력 SHA256과 설정은 `comparison.json`에 저장한다.

## M3C2 범위

원통 내 두 epoch 표본을 법선 방향으로 투영하고 평균의 차이 `d`를 계산한다. 법선은 각 표면의 16/24cm 이웃에서 PCA planarity가 더 나은 방향을 선택한다. 원통 반경 10cm, 양방향 반길이 65cm, 최소 epoch당 표본 6개다.

`LoD95 = 1.96 × (sqrt(s0²/n0 + s1²/n1) + 0.02m)`.

직접 구현한 기본 M3C2 거리/nominal LoD이며 CloudCompare 또는 py4dgeo 자체를 실행한 것은 아니다. 2cm는 데모의 정합 오차 가정이다. 메쉬 표본의 상관이나 Poisson 재구성 오차를 보정한 95% coverage, M3C2-EP의 covariance propagation을 주장하지 않는다. [CloudCompare 공식 M3C2 설명](https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)), [py4dgeo 공식 문서](https://py4dgeo.readthedocs.io/en/latest/).

화면 규칙: `|d| >= 6cm`, `|d|/LoD >= 1`, PCA planarity >= .25이면 빨강. 차이가 3.5cm 이상이며 ratio >= .5 또는 법선이 약하면 노랑(빨강 조건 우선). 표본 부족은 비교 불가다. 통계적으로 유의하지 않은 모든 점을 의심 영역으로 바꾸지 않는다.

## GLS scoring core 범위

[Palma et al. 2016 원문](https://perso.telecom-paristech.fr/boubek/papers/ChangeDetect3D/ChangeDetect3D.pdf)의 Eq.2–4에 따른 scoring core를 재구현한다. [원문·Ponca 수식 대조](gls-core-reproduction-source-check.md)에 원식의 상세를 기록했다.

- APSS oriented sphere fitting, Pratt-normalized offset `tau/t`, curvature `t*kappa`, fit alignment의 차이제곱을 합산한다. weight는 `(1 - distance²/radius²)²`다.
- 수식은 공개 [Ponca gls.h](https://github.com/poncateam/ponca/blob/fd5150bf9079aabdb82a6f05e419b76111c1f788/Ponca/src/Fitting/Basket/extensions/gls.h), [orientedSphereFit.hpp](https://github.com/poncateam/ponca/blob/fd5150bf9079aabdb82a6f05e419b76111c1f788/Ponca/src/Fitting/Basket/fit/orientedSphereFit.hpp)와 대조했다. 해당 재구현 파일은 MPL-2.0 notice를 보존한다. 코드 복제본을 detector로 실행한 것은 아니다.
- 표면 union의 occupied grid를 쓰되 spacing은 **30cm로 고정**한다. 원문의 Gabriel radius/평균 density 기반 spacing은 구현하지 않는다. 1,412개 query, 선형 20 scale(3–60cm)이다.
- Eq.3: 각 표면점에 대해 query마다 그 점을 포함하는 **최소 scale 하나**를 선택한다. Eq.4: 선택한 delta들을 `(maxScale - query 거리)`로 가중 평균한다. 모든 scale delta의 단순 평균과 다르다.
- normal convention이 반대인 fit은 부호를 맞춘다. 불안정 fit 또는 지원 query 3개 미만은 값을 채우지 않는다. 점수는 raw `ch_p`이고 0–1 clipping/확률 변환을 하지 않는다.
- 자체 표시 규칙: 점수 >= .08, 충분한 query 6개 이상, weight의 65% 이상에서 delta > .05, 반대편 최근접 표면 거리 >= 3.5cm일 때 빨강. 점수 >= .025이나 강한 근거 조건을 충족하지 않으면 노랑이다. 이 threshold/등급은 논문 자체의 출력 등급이 아니다.
- LoOP outlier 제거, Gabriel-radius 전처리, 원문 consistency check·proximity graph propagation/consolidation·visibility 추론을 포함하지 않는다. **Palma 전체 알고리즘 재현이나 저자 detector 실행 결과라고 부르지 않는다.**

## 시각화와 해석

상단에서 추가 2개·소실 1개·이동 9개의 기존 변화 추정 목록을 고른다. 대상 선택 시 전후 원본 정지 키프레임·위치 표시·메쉬 확대 위치가 함께 바뀐다. 각 epoch의 사진 후보를 개별적으로 넘기거나 원본 전체 프레임을 큰 화면으로 볼 수 있다. 사진은 자동 재생하지 않으며 이벤트 선택이 빠르게 바뀌면 이전 사진 로드 결과를 취소한다.

기본 지도는 선택한 위치 주변을 추가(파랑)·소실(주황)·이동(보라)로 표시한다. 이 색상의 **종류는 기존 변화 추정**, 해당 색상으로 강조되는 **표면은 독립적으로 계산한 방법별 차이**다. 노랑은 불확실한 표면 차이다. 선택 상자 바깥은 원래 메쉬 색상을 어둡게 표시해 맥락을 남긴다. 각 지도에 선택 대상과 표면 비교 상태를 구분해 표시한다. 원본 전체 표면의 연속 차이 점수도 선택할 수 있으며 이 모드에는 객체 라벨에 따른 마스킹이 없다.

M3C2와 GLS의 카메라는 동기화하고 이전·이후 메쉬 전환 시 유지한다. raw 점수의 척도는 다르다. M3C2 heatmap은 ratio 0–3, GLS heatmap은 raw score 0–.25 범위를 색에 매핑하며 이 범위 이상은 포화된다. 원래 float 값은 `core-fields.npz`에 보관한다.

계산 결과 JSON에는 연결된 change-evidence core들을 14cm 그리드로 묶고 넓은 구역을 수평 1.6m 타일로 나눈 정보도 보존한다. 이는 객체 분할이나 독립적인 변화 사건이 아니다. 현재 화면의 주 목록은 이 타일 대신 사진이 연결된 기존 변화 추정이다.

방법별 카드의 상태는 선택 위치의 상자 안 삼각형 면적을 정점별 1/3로 가중해 요약한다. 비교 불가 표면이 70%를 넘거나 해당 표면이 없으면 '사진 확인 필요', 차이 표면이 12% 이상이면 '변화 감지', 차이와 의심 표면 합이 10% 이상이면 '변화 의심', 그 외는 '표면 차이 작음'이다. 이는 자체 표시 규칙이며 변화 종류나 확률을 새로 추론하지 않는다. 천장 자르기는 표시 기능이다. 추가/소실은 visibility 없이 확정하지 않으므로 기존 종류에도 '추정'을 표시한다. 큰 기하 차이가 실제 변화인지 재구성 아티팩트인지는 이 결과만으로 확정할 수 없다. 현재 데모는 가중치를 GT로 맞추거나 실제 노이즈에 대한 성능 우열을 검증한 실험이 아니다.

## 확인

```powershell
python -m unittest discover -s tools/tests -v
python -m spacewatch3d check-fixtures
python -m unittest discover -s tests -v
python -m unittest discover -s apps/autotour360/tests -p 'test_*.py' -v
node --test apps/autotour360/tests/*.test.mjs
```

계산 테스트는 평면의 알려진 12cm 이동, 누락 표면의 undefined 처리, 평면 APSS offset, 단위구의 curvature, Eq3 최소 포함 scale/Eq4 weighting, normal 부호 불변성을 확인한다. 정확도 benchmark를 수행하지 않는다. 서버 테스트는 새 endpoint의 Range 지원과 공개 파일 경계를 확인한다. 원본·기존 출력은 보존했다.

실행 결과: 계산 테스트 7개, 공통 contract 19개와 fixture 7종, 앱 Python 15개(새 endpoint 포함), Node 39개가 통과했다. 브라우저에서 두 WebGL 지도의 실제 픽셀 출력/GL error 0, 이전 촬영 전환, 구역 확대, 연속 점수, 휠 확대를 확인했다. 기존 Rescan 뷰어도 객체 12개와 전후 원본 사진이 정상 로드됐다. 실제 화면 캡처는 로컬 `outputs/mesh-comparison-qa/`에 보관했다. 이 확인은 구현 동작 점검이며 검출 정확도 검증이 아니다.

사진·변화 종류 UI 수정 후 Node 45개 및 서버 테스트 4개가 통과했다. 새 테스트는 전후 좌표 원점 변환, 잘못된 epoch/외부 URL/누락 사진 연결 제외, 다른 촬영 쌍 거부, 공간 내 표면 요약과 변화 종류의 분리, 미관측 표면의 소실 오판 방지를 확인한다. 사용자가 열어 둔 실제 탭에서 세 종류 필터, 대상 선택, 양쪽 원본 JPEG, 후보 사진 넘기기, 전후 사진 확대 창을 확인했다.
