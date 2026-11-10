SpaceWatch v5.1.9 — 공통20cm 키포인트 생성 기준

현재 릴리즈 v5.1.9: 모든 새 키포인트 생성의 기본 clearance_m=0.20 m. 기존 Dorm3 결과는 보존한다.
검증 결과와 제한 사항: docs/KEYPOINTS_5_1_9.txt. 과거 검증 수치는 아래에 별도로 보존한다.

이전 릴리즈 v5.1.7의 검증 기록
  Florence-2에서 같은 뷰의 이미지 전처리·GPU 전송·시각 특징 계산을 한 번만 수행합니다.
  객체별 텍스트 처리와 생성은 기존 설정 그대로 실행하며 캐시는 해당 뷰에서만 유지합니다.
  기본 72뷰 ABBA: 추론(모델 로딩 포함) 167.37초 -> 137.78초, 17.68% 감소.
  보완 72뷰 AB: 216.32초 -> 170.01초, 21.41% 감소(각 코드 1회).
  원래 재시도 처리와 프로세스 실행까지 포함하면 기본 226.27초 -> 197.77초(12.60%),
  보완 291.52초 -> 244.92초(15.99%)입니다. 모든 토큰·마스크·품질 결과가 일치합니다.
  로컬 CPU 71개 및 서버 격리 CPU 86개 테스트, 캐시 무효화·재시작 검사를 통과했습니다.
  고정된 전체 뷰 입력의 구성 요소 검증이며 원본 영상부터의 전체 시간은 아닙니다.
  기본 설정은 이전 결과를 가져오므로 별도 전체 회귀 실행은 호환성 확인용입니다.
  서버 배포·추가 호환성 검증 완료: 1,057.25초(17분37초), 메시·계획·렌더26장 동일.
  이 재사용 실행의 시간 차이는 Florence 최적화 효과로 해석하지 않습니다.
  앞으로 Florence/Qwen은 추가 최적화 대상에서 제외합니다(사용자 지시2026-10-26).
  진행 상태: docs/OPTIMIZATION_STATE.json.
  근거: docs/RELEASE_5_1_7_FLORENCE_ADOPTION.json.

이전 릴리즈 v5.1.6의 검증 기록
  BVH 가시성 계산에서 순서가 명확한 두 경계값은 직접 비교하여 함수 호출 비용을 줄였습니다.
  같은 값·부호 있는 0·NaN은 기존 fmin/fmax를 사용해 원래 수치 동작을 유지합니다.
  200,182개 수치 쌍, 합성 광선 10,827개 및 실제 광선 49,152개의 결과가 정확히 같았습니다.
  동일 입력 전체 계획 6회에서 모든 계획 필드와 렌더 요청이 정확히 일치했습니다.
  dorm2 각 코드 2회 교차 비교 평균: 520.51초 -> 460.33초(11.56% 감소).
  dorm3 각 코드 1회 비교: 427.01초 -> 370.45초(13.25% 감소, 추가 반복 측정 없음).
  최대 RSS 차이는 dorm2 -0.90MiB, dorm3 +0.19MiB입니다.
  이 수치는 계획 단계 시간이며, 5.1.6 전체 파이프라인 회귀 검증도 완료했습니다.
  기본 재사용 실행1,062.53초(17분43초), 직전5.1.5 대비8.30% 감소했습니다.
  메시·계획·렌더26장이5.1.5/r2/v5와 정확히 일치하며 기존 관측 목표 미달 상태도 같습니다.
  직전 5.1.5 기본 재사용 파이프라인은 1,158.68초(19분19초)였습니다.
  이전 5.1.4 새 추론/재구성은 2,196.08초(36분36초)였습니다.
  이전 재구성 수치는 동일 조건248프레임 단일 비교이며 분할·GS 학습은 제외합니다.
  캐시·복셀 저장 최적화와 관측 목표·표본 수·학습 스텝 등 모든 품질 조건을 유지합니다.
  근거: docs/NATIVE_BOUNDS_BENCHMARK_ITER19.json, docs/NATIVE_BOUNDS_FULL_PARITY_ITER21.json.

서버 실행
  cd /workspace/spacewatch_pipeline_v5_1
  ./start_v5_1.sh configs/dorm2_dorm3.json /workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027
기본 결과: /workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027

변경한 여섯 영역
1. 플래너: 읽기 전용 배열 스냅샷, 선택 집합별 평가 캐시,
   삭제 후보 검사에서 현재 집합의 반복 평가 제거. 반환 보고서는 별도 복사합니다.
2. 실행기: 의존 관계에 따라 CPU 작업 1개와 GPU 작업 1개를 병행합니다.
   gs_prepare를 먼저 준비하여 계획 계산과 새 3DGS 학습을 겹칩니다.
   GPU 작업은 이 대화의 단일 GPU 잠금을 공유하며, 오류/중단 시 작업을 정리합니다.
3. 점군/의미 투영: 점군 저장 시 voxel key와 pixel_to_point 대응표를 생성합니다.
   후속 인덱스·필터링·투영에서 재사용하며 체크섬과 정확한 PLY 행 순서를 검증합니다.
   이전 재구성에 대응표가 없으면 기존의 정확한 검색 경로를 사용합니다.
4. 3DGS 입력: 기본 2 GiB 디코딩 캐시, CPU 미리 읽기, pinned staging buffer 2개,
   CUDA 복사 스트림, 고정 카메라 역행렬 재사용, 중복 체크포인트 쓰기 제거.
5. PanoVGGT: 겹치는 프레임의 DINO 특징을 최대 512 MiB까지 보관합니다.
   여러 프레임을 함께 처리하는 후속 네트워크는 매 윈도마다 실행합니다.
6. 집계/등록: 의미 투표의 Python 반복문을 NumPy 집계로 교체하고,
   CUDA 정수 집계와 등록 최적화의 해석적 희소 Jacobian을 추가합니다.
   단계 receipt에 계산·GPU 대기·해시 검사 시간과 시작/종료 시각을 기록합니다.

실행 옵션
  runtime.overlap_cpu_gpu: true (기본), false는 순차 실행
  gaussian.decoded_cache_mib: 2048 (기본)
  semantic.aggregation_backend: "numpy" (기본) 또는 "cuda"
    CUDA 선택 시 의미 투영 단계도 GPU 잠금을 사용합니다.
    작은 입력은 torch 시작/파일 전달 비용이 있어 NumPy를 기본으로 유지합니다.
  재구성 설정 execution.dino_feature_cache: true (기본)
  재구성 설정 execution.dino_cache_mib: 512 (기본)
  재구성 설정 optimization.analytic_jacobian: true (기본)
  재구성 설정 execution.observation_pair_cache: true (기본), false는 매 단계 재계산

기본 configs/dorm2_dorm3.json은 기존 v5처럼 재구성·의미 메시와 학습된
3DGS를 재사용합니다. 기본 실행만으로는 새 PanoVGGT 추론이나 새 3DGS 학습
속도를 측정할 수 없으며, 해당 경로는 별도 서버 검증을 수행했습니다.

이전 Gaussian 체크포인트는 configs/gaussian_reuse_compatibility.json의
생성 코드/현재 코드 SHA-256 쌍에 한해 가져옵니다. 데이터·좌표·학습 설정과
파일 체크섬 검사를 유지하고 결과에 reused 및 원본 생성 코드 해시를 기록합니다.

검증 자료
  docs/V5_1_VALIDATION.txt: v5.1 검증 범위, 수치, 제한 사항
  docs/UNIT_TESTS.txt, docs/SERVER_UNIT_TESTS.txt: 회귀 테스트
  docs/inputs.json: GPU 입력 전송 및 CPU/CUDA 의미 집계 비교
  docs/panovggt.json: 실제 겹치는 두 윈도의 추론 비교
  docs/export.json: 실제 6프레임 점군·픽셀 대응표 비교
  docs/registration.json: 실제 20프레임 등록 비교
  docs/gs.json, docs/gs_reference.json: 100스텝 3DGS 비교
  docs/v5_baseline/: 이전 v5 기록 (v5.1 실행 기록이 아님)

명령
  ./run.sh --version
  ./run.sh doctor --config configs/dorm2_dorm3.json
  ./run.sh plan --config configs/dorm2_dorm3.json
  ./run.sh status --output /workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027
  ./run.sh stop --output /workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027
  ./start_v5_1.sh configs/dorm2_dorm3.json /workspace/spacewatch_runs/my_v51_run

코드나 설정을 변경한 뒤에는 새 결과 디렉터리를 사용합니다.
같은 코드·설정·출력 경로는 검증된 단계부터 재개합니다.
완료된 출력 경로에서는 start_v5_1.sh가 뷰어를 연결합니다.
원본 v5 디렉터리와 실행 결과는 별도로 보존되어 있습니다.

배포 상태와 실행 PID는 docs/OPTIMIZATION_STATE.json을 확인합니다.
버전별 ZIP과 SHA-256 파일을 보존하며, spacewatch_pipeline_v5_1.zip은 최신 배포본입니다.
과거 완료 결과와 검증 기록의 버전은 그대로 유지합니다.


5.1.8 Dorm3 키프레임 업데이트
Dorm3 기본은 통계 필터93개 대상 + 캐시된 클래스 중요도 + 평지 경로 보정 + 상한 해제된 선정기이며 require_all_objects=false다.
이번 고정 입력 재선정은13개였다.13개를 강제하지 않는다. 입력이 바뀌면 프로필 재사용을 거부한다.
선정 완료와 전체입력 목표 충족은 다르다.83/93품질충족,전체입력90%및기존지역별목표는미달이다.
Dorm2 크기 방식과5a변화탐지는 그대로다. Dorm3에서 크기 템플릿으로 관측 대상을 다시 만들지 않는다.
문서: docs/DORM3_PROFILE_5_1_8.txt. 복구: 서버 배포 영수증의 backup 경로. 새 출력 폴더로 실행한다.

검증 완료: Dorm3 통합 재실행에서 동일13개 시점·경로 재현, 실제3DGS 파노라마13장 생성, 검증 및 뷰어 연결 통과.
PC에서 이번 결과 연결: python3 connect_dorm3_viewer.py
선정 complete와 전체입력 목표 미달(83/93)을 구분한다.
