# Dorm2 점군 정제 및 크기 기준 실험

2026-10-26에 검증한 spacewatch_pipeline_v4_ooops 스냅샷용 패치입니다. 분리된 점군 성분을 보존하고 관측 보존과 크기 기준 채택을 따로 판단합니다. 프레임별 중앙값을 대표 치수에 사용하고, 채택 관측을 병합한 치수는 진단용으로 기록합니다.

패치에는 소스 3개의 변경과 합성 회귀 테스트 12개가 포함됩니다. 당시 수정 전 v4 OOOPS 스냅샷의 별도 작업 폴더에서 아래 명령을 실행합니다. 기존 테스트 환경은 Python 3.14, NumPy 2.5.2, SciPy 1.18.0입니다. PATCH_FILE을 내려받은 패치 경로로 바꿉니다.

```powershell
git apply --check --ignore-space-change PATCH_FILE
git apply --ignore-space-change PATCH_FILE
py -3.14 -B -m unittest discover -s tests -p test_measurement_phase1.py -v
```

동일 Dorm2 입력 비교에서 정제 입력 관측 102개의 XYZ 해시가 일치했습니다. 보존 관측은 96→102개, 크기 기준 채택 후보는 1→2개였습니다. 최종 ID는 양쪽 모두 5개였으며 ID 할당 면은 1,227→1,320개였습니다. 실측 크기 정확도는 평가하지 않았습니다. 실제 Dorm2 결과를 재현하려면 당시 입력과 실행 환경이 필요합니다.
