# Camera and panorama coordinate system

## World coordinates

- 모든 위치 단위는 GLB 장면 좌표계 기준 미터입니다.
- 세계의 위쪽은 `+Y`입니다.
- `camera_position_glb`와 `look_at_glb`는 길이 3의 `[x, y, z]` 배열입니다.

## Camera JSON minimum schema

```json
{
  "title": "custom viewpoint",
  "sample_index": 0,
  "camera_position_glb": [5.46, 1.73, 10.65],
  "look_at_glb": [6.46, 1.73, 10.66]
}
```

`sample_index`는 참조 선명화를 사용할 때만 원본 프레임을 찾는 키로 필요합니다. 순수 3DGS 가상 위치 렌더링에서는 생략할 수 있습니다.

## ERP orientation

- 출력은 폭:높이 `2:1` equirectangular panorama입니다.
- `look_at_glb` 방향의 수평 방위가 이미지 중앙(`yaw=0`)입니다.
- 뒤쪽 방향은 이미지 왼쪽과 오른쪽 경계에 나뉩니다.
- 위쪽은 세계 `+Y`, 아래쪽은 `-Y`입니다.
- `look_at_glb`의 높이 차이는 ERP 자체를 기울이지 않습니다. 수평 방위만 ERP 중심을 정하고, 고도는 뷰어 초기 pitch 메타데이터로 기록됩니다.

## Matrix convention

`cameras.npz`와 `cameras_refined.npz`의 `camera_to_world_glb`는 4x4 CV 카메라 행렬입니다.

- 카메라 `+X`: 오른쪽
- 카메라 `+Y`: 영상 아래쪽
- 카메라 `+Z`: 전방
- 마지막 열: GLB 세계 좌표의 카메라 중심

Nerfstudio용 OpenGL 변환은 데이터 준비 단계에서 Y/Z 축 부호를 변환해 처리합니다. 외부 시스템에서 행렬을 직접 사용할 경우 CV/OpenGL convention을 혼합하지 마십시오.

## Cubemap and ERP rendering

한 위치에서 100도 FOV의 6개 투시 면을 렌더링한 뒤 중첩 영역을 가중 블렌딩합니다. 기본 `face-size=1024`는 최종 `4096x2048` ERP에 대응합니다. 출력 폭만 8K로 높여도 3DGS 모델의 실제 공간 디테일이 증가하지는 않습니다.
