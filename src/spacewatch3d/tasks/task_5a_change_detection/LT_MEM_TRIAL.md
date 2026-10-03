# LT-Mem 입력 준비 실험 · scene0462_00

2026-10-03 기준 **공식 LT-Mem 실행이 아니라 입력 준비 실험**이다. [공식 프로젝트](https://lt-mem.github.io/)의 코드 링크는 `Code (TBD)` 상태다. Task 4와 동일한 공개 SpaCeFormer 예측 JSON에서 객체 관측 정보를 실제로 생성했다. 기존 Task 5a `run` backend와 공통 schema는 변경하지 않았다.

## 실제 실행 결과

원본 60,000점, 50개 예측 마스크 모두 읽어 **50개 관측 레코드**를 생성했다. 마스크에 속한 고유 점은 38,868개이며 그중 716점은 여러 마스크에 중복 포함된다. 따라서 50개 마스크를 정답 객체 50개로 해석하지 않는다. 마스크 번호와 클래스는 예측값 그대로 보존한다.

각 레코드에는 `observation_id`, 원본 `source_mask_id`, 예측 클래스·점수, 점 개수, XYZ 평균 중심, bbox, bbox 부피, RGB 채널별 8-bin 히스토그램을 저장한다. 히스토그램은 **색상 특징 대용값**이며 LT-Mem 논문의 visual embedding이 아니다. bbox 부피는 실제 물체 부피와 다를 수 있다. 원본 `scores`는 보정된 확률로 해석하지 않고 변환 없이 기록한다.

Task 4에서 메시를 생성한 마스크 3·4·13도 포함한다. 큰 영상·점군·가중치는 커밋하지 않고, 실제 생성된 작은 [관측 JSON](assets/lt-mem-scene0462-00/observations.json)과 [실행 보고서](assets/lt-mem-scene0462-00/run-report.json)를 남겼다.

## 현재 확인하지 못한 부분

- scene0462_00은 **단일 촬영**이다. 실제 세션 간 Re-ID, 이동·등장·소실 정확도를 평가할 수 없다.
- segmentation을 다시 추론하지 않고 공개된 예측 결과를 사용했다.
- 공식 LT-Mem의 reasoning, LLM judge, volatility update, Tri-Memory 및 VQA를 실행하지 않았다.
- 현재 생성 파일은 **관측 입력 데이터**이며 Live/Delta/Meta 데이터베이스가 아니다. `mask_XX`는 관측 ID이며 persistent track ID가 아니다.
- 공간은 입력 JSON의 중심 이동된 Z-up 좌표, m 단위를 보존했다. 다른 촬영 회차와 정합된 좌표라고 주장하지 않는다. Task 4의 ScanNet 원본 좌표로 옮길 때는 해당 문서의 translation을 적용해야 한다.
- 관측 JSON은 Task 5a 내부 실험 형식이다. 공통 `objects`/`changes` manifest를 대체하지 않는다.

메모리 생성은 보통 `관측 입력 → 세션 정합·객체 대응 → 변화 판단·메모리 갱신 → Live/Delta/Meta 저장` 순서이고, VQA는 저장된 메모리를 읽는다. LT-VQA는 논문의 평가용 데이터셋 이름이다. [논문 III-B–III-D](https://arxiv.org/html/2608.19059v1)

## 재현

Python 3.10+와 저장소 기본 의존성만 필요하다. 실측 실행 환경은 Windows, Python 3.12.14, jsonschema 4.26.0이다. NVIDIA/CUDA가 필요하지 않다.

```bash
python -m pip install -e .
python -c "from pathlib import Path; from urllib.request import urlretrieve; p=Path('data/lt_mem_trial'); p.mkdir(parents=True,exist_ok=True); urlretrieve('https://nvlabs.github.io/SpaCeFormer/assets/pointclouds/scene0462_00.json',p/'scene0462_00.json')"
python -m spacewatch3d.tasks.task_5a_change_detection.prepare_observations \
  --source data/lt_mem_trial/scene0462_00.json \
  --output outputs/lt_mem_input_trial/observations.json
python -m spacewatch3d check-fixtures
python -m unittest discover -s tests -v
```

재실행은 새로운 출력 파일 경로를 지정한다. 기존 파일을 덮어쓰지 않는다. 원본 SHA-256은 `5a363f8a444d13827e4113f6f1dad1676b90c9cbbca7af4490d1cdfad21f9abb`이며 Task 4 source manifest와 일치했다.

[Colab 입력 준비 노트북](lt_mem_input_trial.ipynb)은 위 다운로드·입력 생성 명령을 실행하고 결과를 ZIP으로 내려받는 경로다. **Colab에서 실행했다고 주장하지 않는다.** 메모리 추론 구현이나 GPU backend가 없으므로 GPU를 선택해도 공식 LT-Mem 추론이 추가되지 않는다. 실제 GPU 모델 단계는 공식 코드 공개와 별도의 영상·세션 입력 연결 후 검증해야 한다.

## 공식 구현 연결에 필요한 데이터

같은 공간의 두 회차 이상, 공통 좌표 정합 결과와 스케일, 객체별 원본 이미지/마스크와 visual embedding, 가림·관측 범위 정보가 필요하다. 논문의 비공개 세부 구현을 가정해 생성한 변화 이력을 공식 결과로 올리지 않는다.
