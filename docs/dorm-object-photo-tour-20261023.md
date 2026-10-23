# Dorm 원본 프레임 중심 변화 투어

기본 화면은 전후 정지 파노라마다. 사진 위 박스를 누르면 같은 시점의 화면에 이전·이후 객체 원본을 반투명하게 겹친다. 박스를 다시 누르면 원본으로 돌아간다. 투명도 조절과 이전/다음 변화, 확인 시점 이동을 제공한다. 메시 지도는 기본적으로 접고 `3D 메시 확인`에서 연다.

## 표시 구분

- 인스턴스 비교: 기존 자동 결과의 후보 2묶음. 이동·추가 가능 대안을 보존하고 확정 이동으로 바꾸지 않는다.
- 변화 의심 구역: 기존 전체 메시 비교 중 객체 표면만 남은 8구역. 같은 삼각형이 인스턴스 비교에 포함되어 있으면 제거한다. 현재 ID는 P0023, P0025, P0027, P0028, P0035, P0038, P0039, P0045다. 물체 8개의 실제 변화나 의자 개수를 뜻하지 않는다.
- 객체 표시 필터는 원래 의미 면의 가구·물품 클래스와 기존 영상 트랙 근거를 이용한다. 미분류 면은 별도 객체 트랙 근거가 있고 낮은 바닥 높이를 벗어났을 때만 남긴다. 알려진 벽·바닥·천장·창·문을 영상 박스만으로 다시 객체에 넣지 않는다. 기존 탐지 자체는 재계산하지 않으며, 객체 분할에 독립적인 기하 탐지와 이번 객체 표시 필터를 구별한다. 원본 의미 라벨이 틀리면 표시 필터도 영향을 받는다.
- 잘린 뒤 남은 표면의 거리·면적·사진 박스는 다시 계산한다. 원래 전체 패치 수치를 잘린 객체 부분의 측정값으로 쓰지 않는다.

## 객체 그림자

그림자는 새 모양을 생성하거나 사진 사각형 전체를 반투명하게 붙인 것이 아니다. 해당 촬영의 객체 삼각형을 대상 원본 카메라에 투영하고, 원래 촬영의 여러 RGB 사진에서 같은 표면을 관측한 픽셀만 가져온다. 전체 원본 메시를 이용해 가려진 원본 표면을 제외하고, 캐시된 YOLO 객체 마스크 안쪽 픽셀로 한 번 더 제한한다. 촬영자·배경이 객체에 잘려 들어오는 것을 줄이기 위한 조건이다. 관측·마스크가 부족한 부분은 투명하게 둔다. 주황/파랑 외곽선은 각각 이전/이후 촬영을 구별한다. 이전 그림자는 현재 배경 위에 겹치므로 현재 가림 뒤의 위치도 확인할 수 있다.

사진 간 3D 재투영은 현재의 고정 구역 정합·카메라 자세를 따른다. 따라서 그림자의 어긋남 자체를 실제 객체 이동의 정답으로 읽으면 안 된다.

## 실행

```powershell
python -m spacewatch3d.tasks.task_5a_change_detection.object_photo_tour --source D:/ChatGPT/Workspaces/Projects/jonghap_project/outputs/registration-benchmark-20261019/new-meshfirst-20261022 --surfaces outputs/dorm-surface-change-20261023/run-v10/published --instances outputs/dorm-change-tour-20261023/published --output outputs/dorm-object-photo-tour-20261023/run-v3/published
python apps/autotour360/server.py --port 8768 --dorm-tour-dir outputs/dorm-object-photo-tour-20261023/run-v3/published
```

출력은 새 경로를 지정한다. 원래 메시·사진·분할·탐지 결과는 읽기 전용이다. 캐시된 영상 마스크는 준비된 소스 내부 또는 소스 상위의 `image-lifecycle-v4/masks`에서 읽는다. 이들 원본과 산출물은 Git 제외 데이터다. 원본 표면 ID와 배경·중복 필터 기록은 비공개 `object-presentation-audit.json`에 보존하며, 뷰어는 정해진 객체 PNG와 사진·메시만 공개한다.

계산은 `src/spacewatch3d/tasks/task_5a_change_detection/object_photo_tour.py`, 화면은 `apps/autotour360/dorm-change-tour.js`에 있다. 객체/배경 필터, 가림·영상 마스크 밖 픽셀 투명 처리, 두 목록 분리 및 PNG 공개 경로를 확인했다. 이는 정답 대비 변화 탐지 성능 평가가 아니다.
