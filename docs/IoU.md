# IoU 기반 Segmentation 성능 평가

## 1. IoU란?

IoU(Intersection over Union)는 이미지 Segmentation 결과가 정답 영역과 얼마나 잘 겹치는지를 평가하는 지표이다.

예측한 객체 영역과 실제 정답(Ground Truth) 영역의 교집합을 두 영역의 합집합으로 나누어 계산한다.

\[
IoU = \frac{Prediction \cap GroundTruth}
{Prediction \cup GroundTruth}
\]

즉, SAM2가 예측한 Mask와 사람이 직접 만든 정답 Mask가 얼마나 일치하는지를 0~1 사이의 값으로 표현한다.

예를 들어 IoU가 0.9라면 예측 Mask와 정답 Mask의 겹침 정도가 상당히 높은 것을 의미한다.

---

## 2. IoU를 사용하는 이유

본 프로젝트에서는 2D 이미지에서 Segmentation을 수행한 뒤, 해당 결과를 3D Point Cloud의 각 점에 객체 정보로 전달할 예정이다.

예를 들어 의자 영역으로 판단된 픽셀에 대응되는 3D Point에는 `chair` 라벨을 부여하게 된다.

따라서 2D Segmentation 결과가 부정확하면 잘못된 3D Point에도 객체 라벨이 전달될 수 있다.

이 때문에 SAM2가 객체 영역을 얼마나 정확하게 분할하는지를 수치적으로 평가할 필요가 있으며, 이를 위해 IoU를 사용한다.

단순히 결과 이미지를 보고 "잘 분할되었다"고 판단하는 것은 주관적이므로, 사람이 직접 만든 Ground Truth Mask와 SAM2의 예측 Mask를 비교하여 객관적인 성능을 확인한다.

---

## 3. IoU 계산 방식

다음과 같이 두 개의 Mask를 사용한다.

- Ground Truth Mask
  - 사람이 직접 Labelme 등을 이용하여 만든 정답 영역

- Prediction Mask
  - SAM2가 생성한 Segmentation 결과

예를 들어 다음과 같은 경우를 생각할 수 있다.

```text
Ground Truth 영역 : 10,000 pixel
Prediction 영역   : 10,500 pixel
두 영역의 교집합   : 9,000 pixel
두 영역의 합집합   : 11,500 pixel