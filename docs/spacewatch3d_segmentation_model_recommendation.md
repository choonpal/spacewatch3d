# SpaceWatch3D Segmentation Model 선정 정리

## 1. 프로젝트 목적

SpaceWatch3D에서 Segmentation 모델을 사용하는 목적은 단순히 2D 이미지에서 객체의 Mask를 얻는 것이 아니다.

최종 목표는 영상에서 대표 프레임을 추출하고, 각 프레임에서 Semantic Segmentation을 수행한 뒤, 그 결과를 3D 포인트와 대응시켜 **각 3D Point에 의미 정보를 부여하는 Semantic 3D Map**을 만드는 것이다.

전체 흐름은 다음과 같다.

```text
Video
  ↓
Keyframe Selection
  ↓
2D Semantic Segmentation
  ↓
각 Pixel의 Class 추출
  ↓
2D Pixel ↔ 3D Point 대응
  ↓
3D Point에 Semantic Label 부여
  ↓
Multi-view Voting / Confidence 통합
  ↓
Semantic 3D Map 생성
```

예를 들어 최종 3D 포인트는 다음과 같은 정보를 갖게 된다.

```text
Point #10233
XYZ   = (1.25, 3.41, 0.82)
Class = chair
Confidence = 0.91
```

현재 우선적으로 구분하려는 클래스는 다음 4개이다.

```text
chair
floor
wall
ceiling
```

---

## 2. 현재 SAM2 사용 시 확인된 문제

SAM2는 객체의 정밀한 Mask를 생성하는 능력은 우수하지만, 현재 프로젝트의 핵심 목적과는 몇 가지 차이가 있다.

### 2.1 Prompt 의존성

SAM2는 기본적으로 사용자가 Point나 Box 등의 Prompt를 제공해야 한다.

```text
Image
  ↓
사람이 Point 클릭
  ↓
SAM2
  ↓
Mask
```

따라서 영상 전체를 자동 처리해야 하는 프로젝트에서는 사람이 프레임마다 직접 클릭하는 방식은 사용할 수 없다.

### 2.2 Semantic Class 자동 판단 불가

SAM2는 Mask는 생성하지만 해당 Mask가 무엇인지 직접 판단하지 않는다.

예를 들어 SAM2가 특정 영역을 잘 분리했다고 하더라도 다음 정보는 별도로 필요하다.

```text
이 Mask가 chair인지?
wall인지?
floor인지?
```

결국 SAM2 단독으로는 다음 구조가 된다.

```text
Image
  ↓
Prompt
  ↓
SAM2
  ↓
Mask
  ↓
별도의 Class 판단
```

반면 본 프로젝트에서는 다음과 같은 구조가 더 적합하다.

```text
Image
  ↓
Semantic Segmentation
  ↓
모든 Pixel에 Class ID 할당
```

---

# 3. Segmentation 모델 후보 비교

## 3.1 SegFormer-B0 / B1 + ADE20K

### 역할

현재 프로젝트의 **1순위 메인 후보**이다.

SegFormer는 Semantic Segmentation을 목적으로 만들어진 모델이며, 이미지 전체를 입력받아 각 픽셀에 Class를 부여한다.

```text
Image
  ↓
SegFormer
  ↓
Pixel별 Semantic Class
```

예를 들면 다음과 같은 결과를 얻을 수 있다.

```text
pixel (100, 500)  → wall
pixel (600, 800)  → chair
pixel (800, 2000) → floor
pixel (200, 100)  → ceiling
```

이 결과는 바로 3D Point에 전달하기 좋다.

```text
2D Pixel
  ↓
Semantic Class
  ↓
2D ↔ 3D Mapping
  ↓
3D Point Semantic Label
```

### ADE20K를 사용하는 이유

ADE20K에는 프로젝트에서 필요한 실내 객체 및 배경 클래스가 이미 포함되어 있다.

예:

```text
wall
floor
ceiling
chair
armchair
seat
swivel chair
stool
```

따라서 별도의 학습 없이도 초기 성능을 바로 확인할 수 있다.

### Chair 계열 Class 통합 필요

프로젝트에서는 `chair`, `stool`, `armchair` 등을 모두 하나의 `chair`로 취급할 수 있다.

따라서 다음과 같은 Label Mapping을 적용한다.

```text
ADE20K Label        Project Label

chair          ┐
armchair       │
seat           ├──→ chair
swivel chair  │
stool          ┘

floor          ───→ floor
wall           ───→ wall
ceiling        ───→ ceiling
```

예시 코드:

```python
PROJECT_CHAIR = {
    "chair",
    "armchair",
    "seat",
    "swivel chair",
    "stool",
}
```

이 과정은 평가 시 매우 중요하다.

예를 들어 실제 이미지 속 객체가 Stool인데 모델이 `stool`이라고 정확하게 예측했음에도, GT가 `chair`라는 이유로 오답 처리하면 평가가 왜곡될 수 있다.

### B0와 B1 선택

현재 HP 노트북의 GPU가 MX250이므로 우선적으로 B0를 사용하는 것이 현실적이다.

```text
SegFormer-B0
  ↓
성능 충분?
  ├─ YES → 그대로 사용
  └─ NO  → SegFormer-B1 추가 테스트
```

B0는 상대적으로 가벼워 전체 파이프라인 개발에 유리하다.

---

## 3.2 Mask2Former + ADE20K

### 역할

SegFormer와 비교하기 위한 **고성능 비교 모델**로 적합하다.

Mask2Former는 다음 세 가지 작업을 하나의 구조에서 처리할 수 있다.

```text
Semantic Segmentation
Instance Segmentation
Panoptic Segmentation
```

현재 프로젝트에서는 Semantic Segmentation만 필요하지만, 이후 공간 변화 탐지나 개별 객체 추적을 고려하면 확장성이 있다.

예를 들어 미래에는 단순히:

```text
chair가 있다 / 없다
```

수준이 아니라

```text
Chair #1 위치 변경
Chair #2 그대로 유지
새로운 Chair #3 등장
```

과 같은 Instance 단위 변화 분석이 필요할 수 있다.

이 경우 Mask2Former 계열이 유리할 수 있다.

### 단점

SegFormer-B0보다 구조가 복잡하고 연산량이 크기 때문에 MX250 환경에서는 속도 차이가 클 수 있다.

따라서 다음과 같은 비교가 적절하다.

```text
SegFormer-B0
→ 경량 Semantic Segmentation

Mask2Former
→ 상대적으로 높은 성능의 Semantic Segmentation
```

---

## 3.3 Grounding DINO + SAM2 (Grounded SAM)

Grounded SAM은 기존 SAM2의 Prompt 의존성을 줄일 수 있는 방법이다.

구조는 다음과 같다.

```text
Text Prompt
"chair"

  ↓

Grounding DINO

  ↓

Bounding Box

  ↓

SAM2

  ↓

Object Mask
```

즉 사람이 직접 Point를 찍지 않아도 된다.

### 장점

다음과 같은 객체에는 매우 유용하다.

```text
chair
monitor
table
printer
fire extinguisher
box
robot
```

또한 새로운 객체 이름을 Text로 입력할 수 있기 때문에 Open-Vocabulary 확장에 유리하다.

### 현재 프로젝트의 메인 모델로 애매한 이유

프로젝트의 핵심 클래스는 다음과 같다.

```text
chair      → Thing
wall       → Stuff
floor      → Stuff
ceiling    → Stuff
```

Grounding DINO는 객체 탐지 중심이기 때문에 `chair` 같은 객체에는 잘 맞지만,

```text
wall
floor
ceiling
```

과 같이 화면의 넓은 영역을 차지하는 Stuff Class를 처리하는 데는 전형적인 Semantic Segmentation 모델이 더 자연스럽다.

따라서 Grounded SAM은 메인 모델보다는 **향후 추가 객체를 탐지하고 의미를 확장하는 보조 모델**로 사용하는 것이 적절하다.

---

# 4. Open-Vocabulary Segmentation

향후 프로젝트가 고정된 4개 Class를 넘어 새로운 객체를 자동으로 추가하려면 Open-Vocabulary Segmentation을 고려할 수 있다.

후보:

```text
CAT-Seg
FC-CLIP
ODISE
OVSeg
OpenSeeD
```

이 모델들은 Text / Vision-Language Feature를 활용하여 학습 당시 정의되지 않았던 Class도 Text Prompt를 통해 구분할 수 있다.

예를 들어 향후 VLM이 Keyframe을 보고 다음과 같은 Class를 추출했다고 가정한다.

```text
chair
monitor
printer
fire extinguisher
robot
```

이 결과를 Open-Vocabulary Segmentation이나 Grounded SAM에 넘겨 Mask를 생성한 뒤 3D 공간으로 투영할 수 있다.

예:

```text
Keyframe
  ↓
VLM
  ↓
Object Class 후보 추출
  ↓
Open-Vocabulary Segmentation
  ↓
Mask
  ↓
2D ↔ 3D Mapping
  ↓
Semantic 3D Map 확장
```

다만 현재 단계에서 바로 Open-Vocabulary 모델부터 적용하면 전체 프로젝트 복잡도가 지나치게 증가할 수 있다.

따라서 먼저 고정 Class 기반 Semantic Segmentation을 완성한 뒤 확장하는 것이 좋다.

---

# 5. 추천 파이프라인

현재 단계에서는 다음과 같은 구조가 가장 적합하다.

```text
                  Video
                    │
                    ↓
             Keyframe Selection
                    │
                    ↓
          ┌─────────────────────┐
          │ SegFormer-B0        │
          │ ADE20K Semantic     │
          └─────────────────────┘
                    │
       ┌────────────┼─────────────┐
       ↓            ↓             ↓
      wall         floor        ceiling
                    │
                  chair
                    │
        chair / stool / seat /
        armchair / swivel chair
                    │
                    ↓
             Label Remapping
                    │
                    ↓
             Semantic Mask
                    │
                    ↓
            2D Pixel ↔ 3D Point
                    │
                    ↓
            Multi-view Voting
                    │
                    ↓
        ┌─────────────────────────┐
        │ Semantic 3D Map         │
        │ XYZ + Class + Conf      │
        └─────────────────────────┘
```

향후에는 별도의 Open-Vocabulary Branch를 추가할 수 있다.

```text
                   Keyframe
                      │
                      ↓
                  VLM / Text
                      │
                      ↓
           Grounding DINO + SAM2
                      │
                      ↓
              Open Object Mask
                      │
                      ↓
               3D Projection
```

역할을 나누면 다음과 같다.

```text
SegFormer
→ 기본 공간 구조에 대한 Semantic Label 생성

Grounded SAM / Open-Vocabulary Model
→ 추가 객체 탐지 및 Semantic Class 확장
```

---

# 6. 모델 선정 실험 구성

현재 모델 선정 단계에서는 모델을 너무 많이 비교할 필요가 없다.

우선 다음 세 개로 비교하는 것이 적절하다.

| 모델 | 역할 | Prompt 필요 | Semantic 자동 분류 | 계산량 | 프로젝트 적합성 |
|---|---|---:|---:|---:|---:|
| SAM2 | 기존 Baseline | O | X | 중~높음 | 낮음 |
| SegFormer-B0 + ADE20K | 메인 후보 | X | O | 낮음 | 매우 높음 |
| Mask2Former + ADE20K | 성능 비교 | X | O | 높음 | 높음 |

추가 실험 후보:

```text
Grounding DINO + SAM2
```

---

# 7. 평가 지표

현재 SAM2 평가에서 사용했던 지표를 동일하게 사용할 수 있다.

```text
IoU
Dice Score
Precision
Recall
Inference Time
```

추가로 전체 Scene Segmentation 단계에서는 다음 항목도 고려할 수 있다.

```text
mIoU
Pixel Accuracy
Class별 IoU
GPU Memory Usage
```

초기 `sample3.png` 실험에서는 Chair GT와 비교하고, 이후 Wall / Floor / Ceiling GT도 추가한다.

---

# 8. 테스트 이미지 구성

`sample3.png` 하나만으로 모델을 선정하는 것은 어렵다.

각 Class별 다양한 환경을 포함한 이미지를 준비할 필요가 있다.

예:

```text
Chair
 ├─ 일반 의자
 ├─ 회전 의자
 ├─ Stool
 ├─ 책상 아래 가려진 의자
 └─ 멀리 있는 작은 의자

Wall
 ├─ 일반 벽
 ├─ 창문이 포함된 벽
 └─ 가구에 많이 가려진 벽

Floor
 ├─ 빈 바닥
 ├─ 가구가 많은 바닥
 └─ 반사가 있는 바닥

Ceiling
 ├─ 일반 천장
 ├─ 조명이 포함된 천장
 └─ 구조물이 포함된 천장
```

이를 이용해야 모델의 실제 프로젝트 적합성을 판단할 수 있다.

---

# 9. 최종 추천 순서

## 1단계

```text
SegFormer-B0 + ADE20K
```

부터 테스트한다.

목표:

```text
sample3.png
  ↓
SegFormer-B0
  ↓
전체 Semantic Segmentation
  ↓
chair 계열 Label 통합
  ↓
sample3_chair_gt.png와 비교
```

측정:

```text
IoU
Dice
Precision
Recall
Inference Time
```

---

## 2단계

SegFormer-B0 결과가 충분하지 않다면 다음을 추가한다.

```text
SegFormer-B1
Mask2Former
```

그리고 동일한 이미지와 GT를 사용하여 비교한다.

---

## 3단계

최종 Semantic Segmentation 모델을 선정한 뒤 3D Mapping 단계로 넘어간다.

```text
Segmentation
  ↓
2D Pixel ↔ 3D Point
  ↓
Semantic Label Projection
  ↓
Multi-view Voting
  ↓
Semantic 3D Map
```

---

## 4단계

기본 4개 Class의 3D Semantic Mapping이 안정화되면 Open-Vocabulary 확장을 진행한다.

```text
VLM
  ↓
Object Candidate 추출
  ↓
Grounding DINO + SAM2
또는
Open-Vocabulary Segmentation
  ↓
새로운 Semantic Class
  ↓
3D Map에 추가
```

---

# 10. 현재 결론

현재 SpaceWatch3D의 목적을 고려하면 가장 적합한 선택은 다음과 같다.

### 메인 모델

```text
SegFormer-B0 + ADE20K
```

이유:

- 사람의 Prompt가 필요하지 않음
- Pixel 단위 Semantic Class를 직접 출력
- `chair / floor / wall / ceiling` Class를 ADE20K에서 바로 사용할 수 있음
- 상대적으로 가벼워 MX250 환경에서 테스트하기 적합
- 2D → 3D Semantic Mapping 구조와 직접적으로 연결 가능

### 비교 모델

```text
Mask2Former + ADE20K
```

SegFormer보다 무겁지만 Semantic Segmentation 성능 비교 모델로 적합하다.

### Baseline

```text
SAM2
```

Mask 생성 능력은 우수하지만 Prompt 의존성과 Semantic Class 자동 판단 부재를 보여주는 비교 기준으로 사용한다.

### 향후 확장

```text
Grounding DINO + SAM2
또는
Open-Vocabulary Segmentation
```

새로운 객체 Class 추가와 VLM 기반 Semantic 확장을 위해 사용하는 것이 적절하다.

따라서 현재 모델 선정 실험은 다음 구성이 가장 현실적이다.

```text
SAM2
vs
SegFormer-B0
vs
Mask2Former
```

이후 프로젝트가 Open-Vocabulary 단계로 확장되면:

```text
Grounding DINO + SAM2
CAT-Seg
FC-CLIP
```

등을 별도 후보로 검토한다.
