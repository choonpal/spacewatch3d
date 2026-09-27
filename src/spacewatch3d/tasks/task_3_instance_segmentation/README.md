# Task 3 · 3D 인스턴스 분할

MV3DIS adapter와 객체 경계 정제를 구현한다. 의미 클래스가 확정되지 않았으면 class_name=null을 사용한다. 인스턴스 분할과 의미 분류를 구분한다.

상태: **개발 scaffold**. 실제 backend는 아직 연결되지 않았으며 `run`은 명시적으로 미구현 오류를 반환한다. 담당자 배정은 팀에서 정한다.

## 작업 범위

- 이 디렉터리의 `pipeline.py`, backend 모듈, 설정, 의존성 설명을 수정한다.
- task 전용 테스트는 `tests/task_3_instance_segmentation/`에 추가한다.
- 공통 규약 변경은 별도 PR에서 소비 task와 합의한다.
- 브랜치 예: `task/3-instance_segmentation/my-feature`.

## 입력과 출력

| 입력 이름 | artifact kind | 독립 개발용 입력 |
|---|---|---|
| `reconstruction` | `reconstruction` | `examples/fixtures/before/reconstruction.json` |

출력 kind: `instances`. 완성된 형태의 **합성 예제**: [`before/instances.json`](../../../../examples/fixtures/before/instances.json).
공통 규약: [docs/contracts.md](../../../../docs/contracts.md).

## 바로 확인

저장소 루트에서 가상환경을 활성화하고 `python -m pip install -e .`를 먼저 실행한다.

```bash
python -m spacewatch3d check-fixtures --task 3
```

위 명령은 파일·ID·규약만 검증한다. 모델 추론이나 task 알고리즘을 실행하지 않는다.

구현을 연결한 뒤 실행할 명령:

```bash
python -m spacewatch3d run --task 3 --input reconstruction=examples/fixtures/before/reconstruction.json --output outputs/task-3-trial --config src/spacewatch3d/tasks/task_3_instance_segmentation/config.example.json
```

`run(request: TaskRequest) -> Path`를 구현하고, 출력 디렉터리를 생성한 뒤 출력 manifest 경로를 반환한다.
현재는 exit code 2의 미구현 오류가 정상이며, 결과 파일을 성공한 것처럼 만들지 않는다.

## 첫 구현 완료 기준

- [ ] 원본 PLY의 모든 점에 대해 instance_id 또는 빈 라벨 저장
- [ ] 다운샘플링 결과를 원본 점 순서로 역매핑
- [ ] 객체별 신뢰도·근거 frame_id 기록
- [ ] adapter에 필요한 RGB·깊이·카메라가 없으면 명시적으로 오류 보고; PLY만으로 가능하다고 가정하지 않기
- [ ] 자신의 결과 JSON을 `python -m spacewatch3d validate 결과.json --check-files`로 검증
- [ ] 실행 환경·모델 revision·입력·재현 명령·한계를 README에 기록
- [ ] 합성 fixture 통과와 실제 데이터 성능 평가를 구분해서 보고

## 의존성

공통 환경은 Python 3.10+와 jsonschema만 사용한다. `requirements.txt`는 task 담당자가 검증한 의존성을 기록하는 자리다.
모델별 Python/CUDA 충돌이 있으면 task별 가상환경을 사용해 manifest와 파일로 연결한다.
외부 코드는 `third_party/`, 가중치는 `weights/`, 실제 입력은 `data/`에 두며 Git에 포함하지 않는다.
