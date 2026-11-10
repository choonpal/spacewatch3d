Dorm3 키프레임 중요도 적용 배포본

관측점 21개. 클래스별 0/1/2 단계, score=level/2 적용.
201개 의미영역 점수 산출 / 기존 제외108개 유지 / 관측 대상93개에 점수 적용.
바닥·경로·카메라·노이즈 제외·관측품질·안전거리·개수 상한 해제는 이전21개 조건 유지.
관측 목표는 개수(count_v1), 중요도는 효용에만 사용. 새 필수 관측 조건 없음.

중요: 실행 완료와 관측 목표 완료는 다르다.
planning_status=incomplete, input_coverage_status=targets_unmet.
result/와 verification.json의 미달 대상 및 분모를 확인할 것.
실물 안전/분할 정확도/LLM 역할 판단은 UNVERIFIED이며 완전한 물리 객체 인스턴스가 아니다.
원본 매시 삭제0. 세그멘테이션의 문/승강기문 분류 충돌은 이번 작업에서 수정하지 않았다.

기존 spacewatch_pipeline_v5_1, 다른 작업자의 코드/설정/결과는 변경하지 않았다.
이 폴더는 필요한 코드·관측 메시·이동 프록시·설정·점수·결과를 포함하는 독립 실행 프로필이다.
최종 통합 실행 시 이 프로필을 선택해 사용하며, 기존 파이프라인의 기본값으로 자동 활성화하지 않았다.

재실행(출력 경로는 매번 새로 지정):
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu /workspace/task5b_dorm3_keyframes_20261017/venv/bin/python /workspace/spacewatch_pipeline_releases/dorm3_task5b_scored_20261026_PhVMbj/profile_runner.py --output /workspace/spacewatch_runs/dorm3_scored_NEW

재실행은 저장된 importance_cache를 사용하고 API를 호출하지 않는다.
새 입력 장면용 중요도 산정에는 별도의 클래스 분류/매핑 절차가 필요하다.
result/는 검증한 실행 결과의 원래 출처/경로를 보존한 사본이다.
배포 폴더의 scene.json은 자산 경로만 상대 경로로 치환했고, 기하/객체/표본 분할 동일성을 검사했다.
