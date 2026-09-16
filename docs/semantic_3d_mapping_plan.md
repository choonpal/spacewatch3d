# Semantic 3D Mapping 계획

## 목표

영상으로 생성한 3D Point Cloud의 각 Point에 객체 정보를 추가하여
Semantic 3D Map을 구축한다.

우선 분류할 객체는 다음 4가지이다.

- Chair
- Floor
- Wall
- Ceiling

## 진행 계획

### 1. Segmentation 모델 조사

우선 SAM2를 테스트한다.

추가적으로 실내 환경의 Chair, Floor, Wall, Ceiling을
잘 분할할 수 있는 Segmentation 모델을 조사하고 비교한다.

초기 테스트는 영상 전체가 아니라 이미지 단위로 진행한다.

### 2. 2D Segmentation

입력 이미지에서 Segmentation을 수행하고
각 Pixel에 객체 정보를 부여한다.

예:

- chair
- floor
- wall
- ceiling

### 3. 2D to 3D Mapping

2D 이미지의 Pixel과 대응되는 3D Point를 찾고,
Segmentation 결과를 해당 Point에 저장한다.

최종적으로 Point는 다음과 같은 정보를 갖도록 한다.

(x, y, z, RGB, semantic_label)

### 4. Multi-view Semantic Fusion

같은 3D Point가 여러 Frame에서 관측되는 경우
각 Frame의 예측 결과를 누적한다.

Voting 또는 Confidence 기반 방법을 사용하여
최종 Semantic Label을 결정한다.

### 5. Keyframe

영상의 모든 Frame을 사용하는 대신
공간을 대표하는 Frame만 선택한다.

초기에는 다음 기준을 고려한다.

- 카메라 이동 거리
- 카메라 회전량
