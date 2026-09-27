# 합성 입출력 예제

이 디렉터리의 **모든 자료는 scaffold 검증용 합성 자료**입니다. LingBot-Map, MV3DIS, 정합/변화탐지 또는 키프레임 알고리즘으로 얻은 결과가 아닙니다.

- `before/`: 16×16의 작은 합성 영상, 프레임 한 장, 8개 점으로 된 큐브, 라벨 CSV, OBJ, 각 단계 manifest.
- `after/`: x축으로 0.3만큼 이동시킨 큐브. 같은 물체여도 회차별 instance_id가 다르다는 예제.
- `panorama/`: 32×16의 합성 2:1 영상과 perspective view 형식 예제. 실제 구면 투영 정확도를 검증한 결과가 아닙니다.
- `changes.json`: 좌표계가 같다고 정한 합성 정답 변환과 이동 사례. 실제 검출 성능 수치가 아닙니다.

MP4·PPM·PLY·OBJ·CSV는 실제로 읽을 수 있는 작은 파일입니다. JSON만 존재하고 파일은 없는 placeholder가 아닙니다.
일반 영상은 2fps, 1초이며 manifest에는 첫 프레임만 추출한 상태를 표현합니다.
PPM은 텍스트 이미지 형식이라 diff가 가능하며, 실제 task에서는 PNG/JPEG 등 backend가 지원하는 형식을 사용하면 됩니다.
fixture의 카메라와 이미지는 형식 예제이며 큐브의 정확한 렌더링/재투영 정답을 보장하지 않습니다.

```bash
python -m spacewatch3d check-fixtures
python -m spacewatch3d check-fixtures --task 4
```

Task 3 등의 실제 GPU 모델에 8개 점 fixture를 넣는 것은 성능 실험이 아닙니다. 데이터 로딩·경로·ID 변환 코드를 개발하고, 모델은 별도의 적절한 실데이터로 확인합니다.
