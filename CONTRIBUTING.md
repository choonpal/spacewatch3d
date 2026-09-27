# 병렬 개발 방법

## 작업을 나누는 기준

`main`은 공통 규약과 통합 코드의 기준입니다. 각 작업은 최신 main에서 만든 짧은 task 브랜치에서 진행하고 PR로 통합합니다.
모든 task를 하나의 개인 브랜치에 누적하거나 한 PR에서 여러 task의 구현을 동시에 바꾸는 방식은 피합니다.
작업 브랜치는 실제 구현을 시작할 때 생성합니다. 담당자를 추측해 CODEOWNERS나 reviewer를 지정하지 않습니다.

| 영역 | 편집 원칙 |
|---|---|
| 내 task 디렉터리·내 task 테스트 | 독립적으로 수정 |
| `task_api.py`, `registry.py`, `contracts.py`, `schemas/` | 영향받는 생산·소비 task와 사전 합의 후 별도 PR |
| 공용 fixture | 기존 fixture를 덮어쓰기보다 별도 사례 추가; 규약 변경 시 함께 갱신 |
| README·협업 문서·CI | 관련 변경만 작게 통합 |
| 다른 task의 내부 코드 | 직접 import하지 않고 manifest와 파일로 연결 |

## 새 작업 시작

```bash
git switch main
git pull --ff-only origin main
git switch -c task/5a-change-detection/registration-baseline
python -m spacewatch3d check-fixtures --task 5a
```

로컬 수정이 있는 상태라면 먼저 해당 작업 브랜치에서 커밋합니다. 기존 브랜치에서 진행 중인 내용은 [이관 가이드](docs/migration.md)를 참고합니다.

## 환경과 실행

공통 환경: `python -m pip install -e .`.
모델 Python/CUDA 버전이 서로 충돌하면 task별 가상환경을 별도로 만듭니다. 별도 환경에서도 이 패키지와 해당 task의 의존성을 설치한 후 같은 manifest 규약을 사용합니다.
task backend는 무거운 모델을 모듈 import 시점에 로딩하지 않습니다. `run`이나 해당 backend 함수 안에서 로딩합니다.

모든 task는 `TaskRequest(inputs, output_dir, config)`를 받아 자기 출력 manifest의 `Path`를 반환합니다.
입력 파일은 읽기 전용으로 다루며, 출력 경로는 실행마다 새로 지정합니다.
공통 CLI에는 자동 실행 스케줄러가 없습니다. task별 CLI를 별도 프로세스로 실행할 수 있으며 GPU 자원 배분은 실행 환경에서 조정합니다.

## main 변화 가져오기

작업 내용을 커밋한 뒤 자신의 task 브랜치에서 실행합니다.

```bash
git fetch origin main
git merge origin/main
```

충돌은 변경 이유를 확인해 해결합니다. 다른 사람의 수정이나 원격 이력을 force push로 덮어쓰지 않습니다.

## PR에 포함할 내용

- 변경 task와 목적, 출력 동작
- 입력 데이터 또는 재현 가능한 작은 fixture
- 모델 repository·commit/revision, Python/CUDA, 설치 명령
- 실행 명령 및 실제 검증 결과
- 공통 규약 변경 여부, 관련 task에 미치는 영향
- 아직 구현하지 못한 범위와 관측된 한계

```bash
python -m spacewatch3d check-fixtures
python -m unittest discover -s tests -v
```

CI는 CPU 환경에서 규약과 scaffold를 검사합니다. 실제 모델 추론 정확도나 GPU 메모리 적합성을 보증하지 않습니다.
PR template은 기록 양식이며 branch protection을 설정하거나 사람에게 리뷰를 요청하지 않습니다.

## 실제 데이터 저장

영상·점군·GLB·모델 가중치는 기본적으로 Git에 올리지 않습니다. `data/`, `outputs/`, `weights/`, `third_party/`를 사용합니다.
공유 저장소에 산출물을 올릴 경우 README에 접근 방법·원본 출처·버전·필요시 체크섬을 기록합니다.
외부 모델의 코드와 가중치 사용 조건은 해당 task README에서 출처와 함께 관리합니다.
