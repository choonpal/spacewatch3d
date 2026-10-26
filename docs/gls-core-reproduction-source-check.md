# GLS core reproduction: Palma 2016 수식·Ponca fitness 확인

2026-10-15. 좁은 원문·소스 대조만 수행했다. detector·실험·뷰어 코드를 수정하지 않았다.

**주 문헌:** G. Palma, P. Cignoni, T. Boubekeur, R. Scopigno, *Detection of Geometric Temporal Changes in Point Clouds*, Computer Graphics Forum **35(6), 33–45 (2016)**, DOI [10.1111/cgf.12730](https://doi.org/10.1111/cgf.12730). 식 위치는 [CNR 저자 원고 PDF](https://iris.cnr.it/retrieve/80f9e1d7-2244-4ce0-86fb-8563e309b96b/prod_359538-doc_126694.pdf)의 PDF 페이지 기준이다. PDF 머리말의 임시 `Volume 0 (1981)`은 서지정보로 사용하지 않는다. [저자 프로젝트](https://perso.telecom-paristech.fr/boubek/papers/ChangeDetect3D/)도 학술지·연도·권·쪽수를 확인한다.

## Eq. (2): 한 scale의 descriptor 차이

PDF p. 3, §2:

\[
\delta(s_a,s_b)=
\left(\frac{\tau_a}{t_a}-\frac{\tau_b}{t_b}\right)^2
+(t_a\kappa_a-t_b\kappa_b)^2
+(\phi_a-\phi_b)^2.
\]

`τ`는 Pratt-normalized algebraic offset, `κ`는 signed sphere curvature, `φ`는 scalar field와 입력 normal의 alignment다. **normal 방향 `η`는 이 비교에서 제외한다.** orientation-independent 차이를 위한 식이며, 별도 normal-angle penalty를 더하면 원식과 달라진다. [원문 Eq. (2)](https://iris.cnr.it/retrieve/80f9e1d7-2244-4ce0-86fb-8563e309b96b/prod_359538-doc_126694.pdf#page=3)

## Eq. (3)–(4): 모든 scale의 평균이 아님

PDF pp. 4–5, §5·Algorithm 1: 두 점군 union bounding box의 nonempty grid cell 중심을 공통 query `c`로 쓴다. `cellSide=5·mean(두 점군의 평균 point radius)`, `maxScale=2·cellSide`, `t_i=maxScale·i/20`으로 **20개 scale**을 구성한다. 각 query·scale에서 두 점군 descriptor를 비교한다.

원본 sample `p`로 점수를 옮길 때 각 query에서 **p를 포함하는 최소 scale 하나**만 선택한다:

\[
E_p=\{\langle\delta_i,t_i\rangle_c\in DIF_c:
t_{i-1}<\|p-c\|\le t_i\}. \tag{3}
\]

그 뒤 여러 query에서 선택한 차이를 거리 가중 평균한다:

\[
ch_p=\frac{\sum_{\langle\delta_i,t_i\rangle_c\in E_p}
\delta_i(\mathrm{maxScale}-\|p-c\|)}
{\sum_{\langle\delta_i,t_i\rangle_c\in E_p}
(\mathrm{maxScale}-\|p-c\|)}. \tag{4}
\]

따라서 **p 하나를 중심으로 모든 scale 차이를 평균/max하는 계산은 Eq. (3)–(4) 재현이 아니다.** 원문의 0.05 threshold는 이 `ch_p`에 대한 실험 설정이다. [§5·Eq. (3)–(4), PDF pp. 4–5](https://iris.cnr.it/retrieve/80f9e1d7-2244-4ce0-86fb-8563e309b96b/prod_359538-doc_126694.pdf#page=4)

논문은 `ch_p`를 [0,1] 값과 probability로 설명하지만 Eq. (2)–(4)에 clipping·sigmoid·exponential mapping은 명시하지 않는다. 별도 변환을 붙이면 raw score와 표시 score를 구별해야 한다. calibration된 변화확률이라는 검증으로 확대하지 않는다. 빈 `E_p`나 불안정 fit의 대체 규칙도 이 수식만으로 정의되지 않는다.

## φ와 pinned Ponca fitness는 보완값이지만 차이제곱은 같음

기반 논문 N. Mellado, G. Guennebaud, P. Barla, P. Reuter, C. Schlick, *Growing Least Squares for the Analysis of Manifolds in Scale-Space*, CGF **31(5), 1691–1701 (2012)**, DOI [10.1111/j.1467-8659.2012.03174.x](https://doi.org/10.1111/j.1467-8659.2012.03174.x). Fitness 절, printed p. 1694/PDF p. 4는 다음을 정의한다:

\[
\phi=\frac{\sum_i w_i\nabla s_u(q_i)\cdot n_i}{\sum_i w_i}
=\|u_l\|^2-4u_cu_q.
\]

`φ=1`은 완벽한 field/normal alignment다. [SGP 2012 원문, Fitness 절](https://projet.liris.cnrs.fr/imagine/pub/proceedings/SGP-2012/pdf/v31i5pp1691-1701.pdf)

반면 Ponca commit `fd5150bf9079aabdb82a6f05e419b76111c1f788`의 [gls.hpp `finalize()`](https://github.com/poncateam/ponca/blob/fd5150bf9079aabdb82a6f05e419b76111c1f788/Ponca/src/Fitting/Basket/extensions/gls.hpp)는 **normalization 이전의** `fitness=1−prattNorm2()`를 보관한다. [gls.h `compareTo()`](https://github.com/poncateam/ponca/blob/fd5150bf9079aabdb82a6f05e419b76111c1f788/Ponca/src/Fitting/Basket/extensions/gls.h)는 normalized τ·κ와 fitness의 차이제곱을 합한다.

따라서 동일 fitting·weighting 조건에서는 `fitness=1−φ`이며,

\[
[(1-\phi_a)-(1-\phi_b)]^2=(\phi_a-\phi_b)^2.
\]

**Ponca fitness 차이제곱을 Eq. (2)의 세 번째 항으로 사용하는 것은 대수적으로 동등하다.** 다만 fitness 자체를 alignment 또는 confidence로 표시하면 방향이 반대다. normalization 이후 norm²=1에서 다시 계산하면 정보가 사라지므로 저장 시점도 중요하다. 이 동등성은 APSS fit 및 가중치·normals까지 동일하다는 전제이며, 전체 detector 재현을 뜻하지 않는다.

## 구현 이름의 정직한 범위

- Eq. (2)와 APSS/Ponca parameter 계산만 따르고 scale·query mapping을 바꿨다면 **“GLS 기반 다중 scale APSS 차이 core”** 또는 **“Palma 2016에서 착안한 GLS core”**가 적절하다.
- Eq. (2)–(4)의 query·scale 선택·mapping을 실제로 따랐다면 **“Palma Eq. (2)–(4) scoring core 재구현”**이라고 범위를 명시할 수 있다.
- 원문 §4의 LoOP·Gabriel radius 전처리, §5 consistency check, §6 proximity graph/propagation/consolidation을 생략했다면 **“Palma 2016 전체 알고리즘 재현”이나 저자 코드 실행**이라고 부르지 않는다. 메쉬 sampling·독자 scale·독자 구역화·의심 등급 역시 별도 구현 조건이다.

원문 PDF는 텍스트 수식과 페이지 위치를 확인했으며, 이번 좁은 점검에서는 수식을 렌더링한 페이지의 시각 대조는 하지 않았다. 코드·출력·패키지 변경은 없다.
