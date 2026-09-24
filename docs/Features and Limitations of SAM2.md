# SAM2 특성 및 프로젝트 적용 가능성 분석

## 1. SAM2를 테스트한 이유

본 프로젝트의 최종 목적은 영상에서 객체 및 공간 영역을 구분한 뒤, 2D 이미지의 각 픽셀과 대응되는 3D Point에 의미 정보를 부여하는 것이다.

현재 목표로 하는 Semantic Class는 다음 4개이다.

- chair
- floor
- wall
- ceiling

최종적으로는 다음과 같은 처리가 필요하다.

```text
영상 Frame 입력
      ↓
2D Semantic Segmentation
      ↓
각 Pixel의 Class 결정
      ↓
2D Pixel ↔ 3D Point 대응
      ↓
3D Point에 Semantic Label 부여
```

초기 Segmentation 모델 후보로 SAM2를 선정하고 실제 이미지에서 객체 분할 성능을 테스트하였다.

---

## 2. SAM2의 특징

SAM2는 이미지 또는 영상에서 사용자가 지정한 영역을 분할하는 Prompt 기반 Segmentation 모델이다.

대표적으로 다음과 같은 Prompt를 사용할 수 있다.

- Point Prompt
- Box Prompt
- Mask Prompt

이번 테스트에서는 Point Prompt를 사용하였다.

예를 들어 사용자가 의자 내부의 한 지점을 클릭하면 다음과 같은 의미로 입력된다.

```text
"이 Point가 포함된 영역을 Segmentation해라."
```

중요한 점은 다음과 같은 명령을 입력하는 것이 아니라는 것이다.

```text
"사진 속 chair 전체를 찾아라."
```

즉 SAM2의 주 역할은 Semantic Class를 판단하는 것보다, 주어진 Prompt와 관련된 Mask를 생성하는 것에 가깝다.

---

## 3. SAM2가 직접 제공하지 않는 정보

SAM2가 생성한 결과는 기본적으로 다음과 같다.

```text
Input Image
      +
Point / Box Prompt
      ↓
SAM2
      ↓
Segmentation Mask
```

하지만 본 프로젝트에서 필요한 결과는 다음과 같다.

```text
Input Image
      ↓
chair / floor / wall / ceiling
      ↓
각 Class별 Segmentation Mask
```

예를 들어 SAM2가 특정 영역에 대한 Mask를 생성하더라도 그 Mask가 chair인지, wall인지, floor인지, ceiling인지 직접 판단해 주는 것은 아니다.

따라서 SAM2 단독으로 사용할 경우 별도의 객체 인식 또는 Semantic Classification 과정이 추가로 필요하다.

---

## 4. 실제 테스트 환경

실제 실내 이미지에서 의자를 대상으로 테스트하였다.

사용 모델과 조건은 다음과 같다.

```text
Model      : SAM2.1 Hiera Tiny
Device     : CUDA
Image Size : 3024 × 4032
Prompt     : Positive Point
Class GT   : chair
```

Ground Truth는 Labelme를 이용하여 사람이 직접 제작하였다.

Ground Truth에는 단순히 의자의 좌석만 포함한 것이 아니라 다음과 같은 실제 의자 전체 영역을 포함하였다.

```text
chair
 ├─ 좌석
 ├─ 레버
 ├─ 지지 기둥
 ├─ 발 받침
 └─ 하단 받침대
```

따라서 SAM2 Prediction과 사람이 정의한 전체 chair 영역을 직접 비교하였다.

---

## 5. 실제 SAM2 테스트 결과

의자의 상단 좌석 영역에 Positive Point 1개를 입력한 결과는 다음과 같다.

```text
Positive Point : (1535.7, 2012.2)

--------------------------------------------------
SAM2 Prediction
--------------------------------------------------
Prompt Points : 1
SAM2 Score    : 97.74%
Inference Time: 147.72 ms
```

SAM2 자체에서 출력된 Score는 매우 높았다.

```text
SAM2 Score = 97.74%
```

하지만 사람이 제작한 Ground Truth와 실제 Prediction Mask를 비교한 결과는 다음과 같았다.

```text
============================================================
SAM2 Segmentation Evaluation
============================================================
Image : sample3.png
Class : chair
Prompt Points : 1
------------------------------------------------------------
Segmentation IoU : 51.94%
Dice             : 68.37%
Precision        : 99.54%
Recall           : 52.07%
Pixel Accuracy   : 98.24%
SAM2 Score       : 97.74%
Inference Time   : 147.72 ms
------------------------------------------------------------
TP : 232499
FP : 1086
FN : 214013
TN : 11745170
============================================================
```

주요 결과만 정리하면 다음과 같다.

| 평가 지표 | 결과 |
|---|---:|
| SAM2 Score | 97.74% |
| IoU | 51.94% |
| Dice | 68.37% |
| Precision | 99.54% |
| Recall | 52.07% |
| Pixel Accuracy | 98.24% |
| Inference Time | 147.72 ms |

---

## 6. SAM2 Score는 실제 정확도가 아님

이번 실험에서 가장 주의해야 할 부분이다.

SAM2에서 다음과 같은 값이 출력되었다.

```text
SAM2 Score = 97.74%
```

처음에는 이 값을 Segmentation 정확도가 약 97.74%라고 생각할 수 있다.

하지만 Ground Truth와 실제 비교한 결과는 다음과 같았다.

```text
IoU = 51.94%
```

두 값에는 큰 차이가 존재한다.

이는 SAM2 Score가 사람이 만든 Ground Truth와 비교하여 계산한 실제 정확도가 아니기 때문이다.

SAM2 Score는 SAM2가 생성한 Mask 후보의 품질에 대해 모델 내부적으로 예측한 값이며, Ground Truth 기반 평가 결과와 구분해야 한다.

따라서 본 프로젝트에서는 SAM2 Score가 아니라 다음과 같은 Ground Truth 기반 지표를 실제 성능평가에 사용해야 한다.

- IoU
- Dice
- Precision
- Recall

특히 대표 성능지표로 IoU를 사용한다.

---

## 7. Precision은 높은데 IoU가 낮은 이유

이번 결과에서 특히 특징적인 부분은 다음과 같다.

```text
Precision = 99.54%
Recall    = 52.07%
IoU       = 51.94%
```

Precision은 거의 100%인데 Recall과 IoU는 약 50% 수준이다.

이는 SAM2가 잘못된 영역을 많이 선택한 것이 아니라, 의자의 일부만 매우 정확하게 선택했기 때문이다.

실제 결과를 단순화하면 다음과 같다.

```text
Ground Truth

[좌석]       ← chair
[레버]       ← chair
[기둥]       ← chair
[발 받침]    ← chair
[하단 받침]  ← chair
```

반면 SAM2의 결과는 대략 다음과 같았다.

```text
SAM2 Prediction

[좌석]       ← 대부분 정확하게 검출

[레버]       ← 미검출
[기둥]       ← 미검출
[발 받침]    ← 미검출
[하단 받침]  ← 미검출
```

따라서 SAM2가 선택한 픽셀 대부분은 실제 chair였기 때문에 Precision이 매우 높게 나타났다.

실제로 False Positive는 다음과 같이 매우 작았다.

```text
FP = 1,086 pixel
```

하지만 실제 chair 영역 중 SAM2가 검출하지 못한 영역은 매우 많았다.

```text
FN = 214,013 pixel
```

즉 실제 Chair Pixel 중 SAM2가 약 절반 정도만 검출한 결과가 나타났으며, 이 때문에 Recall은 다음과 같이 낮아졌다.

```text
Recall = 52.07%
```

---

## 8. IoU 51.94%가 나온 이유

IoU는 TP, FP, FN을 이용하여 계산한다.

```text
TP = 232,499
FP =   1,086
FN = 214,013
```

따라서,

```text
IoU
= TP / (TP + FP + FN)

= 232499 / (232499 + 1086 + 214013)

≈ 0.5194
```

최종 IoU는 다음과 같다.

```text
IoU = 51.94%
```

즉 SAM2가 선택한 영역 자체는 상당히 정확했지만, Ground Truth로 정의한 의자 전체 중 상당 부분을 놓쳤기 때문에 전체 객체 Segmentation 성능은 낮아졌다.

---

## 9. 왜 이런 결과가 발생했는가?

사람이 이미지를 볼 때는 다음 구조를 모두 하나의 chair로 인식한다.

```text
        좌석
         │
       지지대
         │
      발 받침대
         │
      하단 받침대
```

하지만 이미지 자체에서는 이 영역들의 특성이 상당히 다르다.

예를 들어,

```text
좌석
→ 빨간색
→ 넓은 영역
→ 부드러운 재질

기둥
→ 은색
→ 매우 얇음
→ 금속 재질

발 받침
→ 원형의 얇은 금속 구조

하단 받침
→ 넓은 금속 원판
```

처럼 색상, 형태, 재질 및 연결 구조가 서로 다르다.

사람은 이러한 부품들이 하나의 의자를 구성한다는 Semantic 정보를 이용하여 모두 chair라고 판단할 수 있다.

반면 이번 Point Prompt는 단순히 다음 정보만 전달하였다.

```text
"이 Point가 포함된 영역을 찾아라."
```

따라서 SAM2는 사용자가 클릭한 빨간 좌석 영역을 매우 정확하게 분리했지만,

```text
"이 좌석과 아래의 금속 기둥,
 발판,
 받침대가 모두 동일한 chair이다."
```

라는 Semantic 관계까지 자동으로 적용하지 못했다.

이것이 이번 실험에서

```text
SAM2 Score = 97.74%
Precision  = 99.54%
```

처럼 선택된 영역 자체의 품질은 높았지만,

```text
Recall = 52.07%
IoU    = 51.94%
```

로 전체 객체 기준 성능은 낮아진 주된 이유로 볼 수 있다.

---

## 10. Pixel Accuracy가 98.24%인 이유

이번 실험에서는 Pixel Accuracy도 매우 높은 값이 나왔다.

```text
Pixel Accuracy = 98.24%
```

하지만 이 값을 Segmentation 성능의 대표값으로 사용하기는 어렵다.

이번 이미지의 전체 픽셀 대부분은 chair가 아닌 배경이다.

실제 True Negative는 다음과 같다.

```text
TN = 11,745,170 pixel
```

즉 수많은 배경 Pixel이 정확하게 Background로 판단되면서 전체 Pixel Accuracy가 매우 높아졌다.

따라서 Pixel Accuracy만 보면 SAM2가 거의 완벽하게 동작한 것처럼 보일 수 있지만, 실제 객체 자체를 기준으로 보면 다음과 같다.

```text
IoU    = 51.94%
Recall = 52.07%
```

본 프로젝트에서는 이러한 Class Imbalance 문제 때문에 Pixel Accuracy보다 IoU를 대표 지표로 사용하는 것이 더 적절하다.

---

## 11. 프로젝트 목적과 SAM2의 차이

본 프로젝트에서 궁극적으로 원하는 것은 다음과 같다.

```text
Frame
  ↓
전체 Scene 분석
  ↓
각 Pixel의 Semantic Class 결정
  ↓
chair / floor / wall / ceiling
  ↓
3D Point에 Class 전달
```

즉 중요한 것은 단순히 특정 영역의 경계를 정확하게 잘라내는 것이 아니다.

각 Pixel에 대해 다음과 같은 Semantic 정보가 자동으로 결정되어야 한다.

```text
"이 영역은 chair이다."
"이 영역은 floor이다."
"이 영역은 wall이다."
"이 영역은 ceiling이다."
```

하지만 SAM2의 기본적인 Point Prompt 방식은 다음과 같다.

```text
사용자 Prompt
      ↓
특정 영역 선택
      ↓
Mask
```

따라서 프로젝트의 요구사항과 SAM2의 기본 목적에는 차이가 존재한다.

---

## 12. SAM2를 실제 시스템에 적용할 경우의 문제

### 12.1 사람이 직접 Point를 입력해야 하는 문제

현재 테스트에서는 사람이 의자의 특정 위치를 클릭하였다.

```text
Positive Point
→ (1535.7, 2012.2)
```

하지만 최종 시스템에서는 영상의 Keyframe이 여러 장 생성될 예정이다.

예를 들어 수백 개의 Keyframe이 생성된다면 사람이 각 Frame마다 다음 작업을 하는 것은 현실적으로 어렵다.

```text
Frame 1 → chair 클릭
Frame 2 → chair 클릭
Frame 3 → chair 클릭
...
```

따라서 SAM2를 사용하려면 자동 Prompt 생성 시스템이 별도로 필요하다.

예를 들어,

```text
Object Detector
      ↓
chair 탐지
      ↓
Point 또는 Bounding Box 생성
      ↓
SAM2
      ↓
Mask 생성
```

과 같은 추가 파이프라인이 필요하다.

이 경우 시스템 구조와 연산량이 더욱 복잡해진다.

---

### 12.2 Class 정보를 직접 제공하지 않는 문제

본 프로젝트에서는 최종적으로 3D Point에 다음과 같은 정보가 저장되어야 한다.

```text
Point 1 → chair
Point 2 → chair
Point 3 → wall
Point 4 → floor
Point 5 → ceiling
```

하지만 SAM2의 기본 결과는 Mask 중심이다.

```text
Mask 1
Mask 2
Mask 3
...
```

따라서 각각의 Mask가 어떤 Class에 해당하는지 결정하는 과정이 추가로 필요하다.

즉 SAM2를 사용할 경우 다음 과정이 필요하다.

```text
SAM2
 ↓
Mask 생성
 ↓
별도 Class 판단
 ↓
Semantic Label
```

반면 Semantic Segmentation 모델이라면 이상적인 경우 다음과 같이 바로 결과를 얻을 수 있다.

```text
Image
 ↓
Semantic Segmentation
 ↓
chair / floor / wall / ceiling
```

프로젝트 구조상 후자의 방식이 더욱 단순하다.

---

## 13. floor / wall / ceiling에서의 추가 문제

현재 실험에서는 chair를 대상으로 테스트하였다.

하지만 프로젝트에서 필요한 나머지 Class는 다음과 같다.

```text
floor
wall
ceiling
```

이들은 chair와 같은 개별 객체와 성격이 다르다.

chair는 비교적 독립적인 객체이지만 wall, floor, ceiling은 이미지 전체에 넓게 펼쳐지는 Scene 영역이다.

예를 들어 하나의 이미지에서도 벽은 다음처럼 여러 영역으로 분리되어 보일 수 있다.

```text
왼쪽 벽
문 옆 벽
물체에 가려진 벽
```

하지만 Semantic 관점에서는 모두 동일한 wall Class이다.

본 프로젝트에서는 이러한 영역 전체에 동일한 Semantic Label을 부여해야 한다.

따라서 특정 Prompt를 기준으로 하나의 Mask를 생성하는 SAM2 방식보다, 이미지 전체 Pixel을 Class별로 분류하는 Semantic Segmentation 방식이 프로젝트 목적에 더 직접적으로 적합할 가능성이 있다.

---

## 14. SAM2의 장점도 존재함

이번 결과가 SAM2 자체의 성능이 낮다는 의미는 아니다.

오히려 선택된 영역만 보면 결과가 매우 좋았다.

```text
Precision = 99.54%
FP        = 1,086 pixel
```

이는 SAM2가 사용자의 Prompt에 대응하는 특정 영역의 경계를 정밀하게 분리하는 능력이 뛰어나다는 것을 보여준다.

따라서 다음과 같은 용도에는 SAM2가 유용할 수 있다.

- Interactive Segmentation
- 사용자가 선택한 객체 Mask 생성
- Object Detector가 제공한 Box를 정밀 Mask로 변환
- 수동 Annotation 보조

즉 SAM2가 좋지 않은 모델이라기보다는,

**SAM2가 잘하는 문제와 현재 프로젝트에서 해결해야 하는 문제가 완전히 동일하지 않다는 것이 핵심이다.**

---

## 15. 현재 SAM2에 대한 프로젝트 평가

### 장점

- Point 하나만으로 Mask 생성 가능
- 선택한 영역 자체는 매우 정밀하게 분할 가능
- 실제 테스트에서 Precision 99.54% 확인
- CUDA 환경에서 Prompt 추론 시간 약 147.72 ms 확인
- 다양한 객체에 대해 별도의 Class 학습 없이 Mask 생성 가능

### 한계

- Semantic Class를 직접 출력하지 않음
- Prompt가 필요함
- Prompt 위치에 따라 결과가 달라질 수 있음
- 동일한 Semantic 객체의 여러 부품을 모두 포함하지 못할 수 있음
- 이번 chair 테스트에서 Recall 52.07%
- 전체 chair Ground Truth 대비 IoU 51.94%
- Frame마다 자동 처리를 위해 추가적인 Prompt 생성 방법이 필요함
- floor / wall / ceiling과 같은 Scene 영역 처리에는 추가 검증 필요

---

## 16. 실험 결과를 통한 핵심 결론

이번 실험에서 가장 중요한 결과는 다음 값들이다.

```text
SAM2 Score = 97.74%
Precision  = 99.54%
Recall     = 52.07%
IoU        = 51.94%
```

이 결과는 다음과 같이 해석할 수 있다.

```text
SAM2가 선택한 영역 자체는 매우 정확하다.
           ↓
Precision 99.54%

하지만 우리가 Semantic하게 정의한
"의자 전체" 중 약 절반만 포함하였다.
           ↓
Recall 52.07%

따라서 전체 Ground Truth와 비교하면
           ↓
IoU 51.94%
```

즉,

> SAM2는 사용자가 지정한 특정 영역을 정밀하게 분할하는 데에는 강점을 보였지만, 의미적으로 하나의 객체에 해당하는 모든 구성 요소를 자동으로 하나의 Class 영역으로 통합하는 데에는 한계가 확인되었다.

---

## 17. 본 프로젝트에서 SAM2를 사용하기 어려운 이유

본 프로젝트의 최종 목적은 단순한 Object Mask 생성이 아니다.

```text
2D 영상
 ↓
Semantic Class 판단
 ↓
chair / floor / wall / ceiling
 ↓
각 Pixel에 Class 정보 부여
 ↓
3D Point로 Semantic Label 전달
```

이 과정은 사람의 Prompt 없이 자동으로 동작해야 한다.

그러나 SAM2 단독 사용 시 다음의 추가 과정이 필요하다.

```text
Object / Region 탐지
        ↓
Prompt 자동 생성
        ↓
SAM2 Segmentation
        ↓
Mask Class 판단
        ↓
Semantic Label
        ↓
3D Mapping
```

따라서 전체 시스템이 복잡해진다.

반면 프로젝트의 4개 Class를 직접 판단할 수 있는 Semantic Segmentation 모델을 사용한다면 다음과 같이 단순화할 수 있다.

```text
Image
  ↓
Semantic Segmentation
  ↓
chair / floor / wall / ceiling
  ↓
3D Mapping
```

따라서 현재 단계에서는 SAM2를 최종 모델로 확정하기보다는 비교용 Baseline 모델로 사용하는 것이 더 적절하다.

---

## 18. 향후 모델 선정 방향

향후 모델은 단순한 Mask 생성 능력보다 다음 조건을 우선적으로 확인해야 한다.

1. chair / floor / wall / ceiling을 직접 구분할 수 있는가?
2. 사람의 Point 또는 Box Prompt 없이 자동으로 처리할 수 있는가?
3. 영상의 모든 Pixel에 Semantic Label을 부여할 수 있는가?
4. IoU가 충분한가?
5. 연속된 Keyframe에 자동 적용 가능한가?
6. GPU 연산량 및 처리시간은 적절한가?
7. 출력 결과를 3D Point에 전달하기 쉬운가?

이 기준을 이용하여 SAM3, Semantic Segmentation 모델, Open-Vocabulary Segmentation 모델 등을 비교할 필요가 있다.

---

## 19. 결론

SAM2.1 Hiera Tiny를 실제 실내 이미지의 chair 객체에 적용한 결과, 단일 Positive Point에 대해 SAM2 내부 Score는 97.74%로 매우 높게 나타났다.

하지만 사람이 직접 제작한 chair Ground Truth와 비교한 결과는 다음과 같았다.

```text
IoU             : 51.94%
Dice            : 68.37%
Precision       : 99.54%
Recall          : 52.07%
Pixel Accuracy  : 98.24%
SAM2 Score      : 97.74%
Inference Time  : 147.72 ms
```

특히 Precision이 99.54%인 반면 Recall이 52.07%로 나타난 것은, SAM2가 선택한 의자 좌석 영역 자체는 매우 정확하게 분할했지만 의자의 지지대, 발판, 하단 받침대 등 전체 Semantic 객체를 포함하지 못했기 때문이다.

따라서 SAM2의 낮은 IoU 결과는 단순히 경계 분할 능력이 부족해서 발생한 것이 아니라, SAM2의 Prompt 기반 Segmentation 특성과 본 프로젝트에서 요구하는 Semantic Segmentation 목적의 차이에서 발생한 것으로 판단할 수 있다.

본 프로젝트에서는 각 Pixel을 chair, floor, wall, ceiling 중 하나로 자동 분류하고 이를 3D Point의 Semantic Label로 전달해야 한다.

이에 따라 SAM2는 Segmentation Baseline 및 Mask 생성 모델로서는 의미가 있지만, 별도의 Class 판단 및 Prompt 생성 과정 없이 본 프로젝트의 최종 Semantic Segmentation 모델로 단독 사용하기에는 한계가 존재한다.

따라서 향후에는 동일한 Ground Truth와 IoU 평가 파이프라인을 이용하여 Semantic 정보를 직접 처리할 수 있는 다른 모델들과 비교한 뒤 최종 모델을 선정하는 방향으로 진행한다.
