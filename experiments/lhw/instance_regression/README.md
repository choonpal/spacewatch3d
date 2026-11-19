# Dorm2 의자 후보 소실 회귀 분석

## 원인

2026년 10월 26일의 로컬 비교에서 legacy 의자 후보 14개가 evidence_v1의 크기 적합 후보 1개로 감소했다. 의자 Semantic 1,305면은 보존됐으며, 큰 연결 영역 두 개의 1,102면이 클래스 크기 상한을 초과했다. 작은 면적 조건에서 47면, 두 축 크기 하한 조건에서 39면이 후보에서 제외됐고 117면만 적합했다.

Seed 부재에 따른 최종 ID 보류는 별도의 조건이다. legacy 전체 최종 ID는 87개였으며 evidence_v1 및 evidence_v2의 실제 입력은 검증된 Seed가 없어 최종 ID 0개였다.

## 구현

evidence_v2는 크기 상한을 지키는 topology proposal을 만들고, 충분한 면적이 있으나 부분 관측으로 작게 보이는 표면을 partial_surface 상태로 기록한다. proposal ID와 최종 Instance ID를 별도로 저장한다. 불안정한 좁은 연결 절단은 진단으로 제한하고, 관측 근거가 없는 Gap 자동 병합은 적용하지 않는다.

## 결과

| 항목 | legacy | evidence_v1 | evidence_v2 |
| --- | ---: | ---: | ---: |
| chair 후보 또는 잠정 proposal | 14 | 1 | 19 |
| 해당 chair 면 | 1,260 | 117 | 1,257 |
| chair 최종 Instance | 14 | 0 | 0 |
| 전체 후보 또는 proposal | 87 | 68 | 179 |
| 전체 최종 Instance | 87 | 0 | 0 |

v2 chair proposal은 object-scale 표면 14개와 partial 표면 5개로 구성된다. 기존 chair 후보 14개가 모두 새 proposal과 겹쳤으며, 기존 후보 1,260면 중 1,247면이 proposal에 포함됐다. 나머지는 잔여 표면으로 보존됐다. 전체 147,020면과 RGB·Semantic·원본 관측 배열을 유지했다.

후보 정의가 달라 개수 증가는 실물 검출 정확도의 지표로 사용할 수 없다. 물리적 의자 개수, 실제 과잉 병합·분할 수, 실제 Seed 적용 성능은 GT 검증이 남아 있다.

## 패치와 재현

`instance_regression.patch`는 당시 v5.1 로컬 스냅샷용이다. 기존 소스 5개 변경, 신규 instance_proposals 모듈 및 회귀 테스트 12개를 포함한다. 신규 파일의 패치 헤더에는 git apply의 기본 경로 제거 규칙에 맞는 after/ 접두사를 추가했다.

당시 수정 전 스냅샷의 별도 작업 폴더에서 다음 명령을 실행한다. PATCH_FILE은 보관 패치의 실제 경로다. 테스트에는 기존 test_instance_evidence.py와 NumPy·SciPy가 필요하다.

```sh
git apply --check --ignore-space-change PATCH_FILE
git apply --ignore-space-change PATCH_FILE
python -B -m unittest discover -s tests -p test_instance_proposals.py -v
```

당시 실제 Dorm2 legacy ID 배열은 기존 결과와 정확히 일치했다. 업로드 준비 중 원래 로컬 코드의 신규 회귀 테스트 12개를 다시 실행했다. 실제 메시 exporter·후속 planner·변화 탐지 전체 실행 검증은 당시 환경 의존성이 남아 있다.

`chair_cause_summary.json`은 evidence_v1에서 의자 면이 처음 탈락한 조건별 집계다. 공간 좌표의 미터 스케일과 실물 크기는 독립 검증이 필요하다.
