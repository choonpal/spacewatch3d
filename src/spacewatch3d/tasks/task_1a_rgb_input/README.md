# Task 1a · RGB 영상 입력

일반 영상에서 프레임을 추출하고 원본 frame index·timestamp를 보존한다. 왜곡 보정을 수행했다면 그 결과와 내부 파라미터를 함께 기록한다.

상태: **개발 scaffold**. 실제 backend는 아직 연결되지 않았으며 `run`은 명시적으로 미구현 오류를 반환한다. 담당자 배정은 팀에서 정한다.

## 작업 범위

- 이 디렉터리의 `pipeline.py`, backend 모듈, 설정, 의존성 설명을 수정한다.
- task 전용 테스트는 `tests/task_1a_rgb_input/`에 추가한다.
- 공통 규약 변경은 별도 PR에서 소비 task와 합의한다.
- 브랜치 예: `task/1a-rgb_input/my-feature`.

## 입력과 출력

#다음 작업시 확인 후 삭제 요망
{
링봇 기본 입력할 때 단순히 n프레임 단위로 보는게 아니라
flow_threshold: 움직임에 따라 키프레임 선정
max_non_keyframe_gap: 너무 오래 선택되지 않으면 강제 선정
keyframes_only_points: 키프레임의 점만 포인트클라우드에 포함
save_predictions: 선정 결과와 프레임별 예측 저장
와 같이 불필요한 프레임 제외하고 3d맵을 만드는데 중요한 프레임들만 뽑는 세팅들이 있음 이거 잘 조절하면 영상 속도와 관련없이 3d 맵 잘 만들 수 있을 것 같은데
적용하고 확인요망
}


| 입력 이름 | artifact kind | 독립 개발용 입력 |
|---|---|---|
| `video` | `video` | `examples/fixtures/before/video.json` |

출력 kind: `frames`. 완성된 형태의 **합성 예제**: [`before/frames.json`](../../../../examples/fixtures/before/frames.json).
공통 규약: [docs/contracts.md](../../../../docs/contracts.md).

## 바로 확인

저장소 루트에서 가상환경을 활성화하고 `python -m pip install -e .`를 먼저 실행한다.

```bash
python -m spacewatch3d check-fixtures --task 1a
```

위 명령은 파일·ID·규약만 검증한다. 모델 추론이나 task 알고리즘을 실행하지 않는다.

구현을 연결한 뒤 실행할 명령:

```bash
python -m spacewatch3d run --task 1a --input video=examples/fixtures/before/video.json --output outputs/task-1a-trial --config src/spacewatch3d/tasks/task_1a_rgb_input/config.example.json
```

`run(request: TaskRequest) -> Path`를 구현하고, 출력 디렉터리를 생성한 뒤 출력 manifest 경로를 반환한다.
현재는 exit code 2의 미구현 오류가 정상이며, 결과 파일을 성공한 것처럼 만들지 않는다.

## 첫 구현 완료 기준

- [ ] 일반 RGB 영상의 읽기와 일정 간격 프레임 추출
- [ ] 중복되지 않는 frame_id 및 원본 시각 기록
- [ ] 해상도·선택 간격을 바꿔도 출처를 추적할 수 있는지 확인
- [ ] 자신의 결과 JSON을 `python -m spacewatch3d validate 결과.json --check-files`로 검증
- [ ] 실행 환경·모델 revision·입력·재현 명령·한계를 README에 기록
- [ ] 합성 fixture 통과와 실제 데이터 성능 평가를 구분해서 보고

## 의존성

공통 환경은 Python 3.10+와 jsonschema만 사용한다. `requirements.txt`는 task 담당자가 검증한 의존성을 기록하는 자리다.
모델별 Python/CUDA 충돌이 있으면 task별 가상환경을 사용해 manifest와 파일로 연결한다.
외부 코드는 `third_party/`, 가중치는 `weights/`, 실제 입력은 `data/`에 두며 Git에 포함하지 않는다.
