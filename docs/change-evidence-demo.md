# 실제 전후 점군으로 만드는 변화 근거 데모

2026-10-12. 기존 AUTO TOUR 360 서버의 http://127.0.0.1:8766/changes.html 또는 상단 **3D 변화 근거**를 연다.

## 목적과 범위

### 사용한 기존 방법과 변화 종류 표시

구현은 SciPy의 26방향 연결 영역(label)과 KD-tree 최근접 검색(cKDTree.query)을 사용한다. 특정 논문의 변화탐지 모델을 재현한 구현은 아니다. MultiChange3D의 공식 변화 라벨이 앞 단계의 변화 결과 역할을 한다.

점군 위와 영역 목록에 제거 후보·추가 후보를 표시한다. 이전 회차의 변화 표면은 제거 후보, 이후 회차의 변화 표면은 추가 후보다. 이동한 동일 객체의 이전·이후 위치가 각각 후보로 나타날 수 있으므로 이 둘을 실제 객체 제거·추가로 확정하지 않는다.

영역의 change.type은 removed / added / moved / unknown, change.status는 candidate / confirmed를 표시한다. 이동이 입력된 경우 change.displacement_m에 명시적으로 제공된 미터 단위 거리만 '이동 · 0.75 m' 형식으로 표시한다. 전후 이동 위치는 change.before_bounds / change.after_bounds로 받을 수 있다. 이는 데모의 표시 형식이며 공통 파이프라인 변위 단위를 자동으로 미터로 바꾸는 어댑터가 아니다.

현재 데이터에는 동일 객체 대응과 미터 단위 변위가 없으므로 change.displacement_m은 null이며 '객체 이동 거리 · 산정 불가'로 표시한다. 최근접 표면 거리나 영역 중심 간 차이를 객체 이동 거리로 쓰지 않는다.

이번 라벨 변경은 저장된 점군과 거리 결과를 재사용하며 요청에 따라 측정·테스트·재검증을 실행하지 않았다.

앞 단계의 변화 결과를 받아, 사용자가 전후 3D 차이와 원본 포인트 출처를 검토하게 만든다. 이번 입력은 [MultiChange3D](https://github.com/3DOM-FBK/multichange3d)의 실제 Open Space RGB-D 재촬영 pair_01, Epoch 0–1과 저자가 제공한 cd_type 변화 라벨이다. 실제 공간에서 유도한 변화다. 취득·정합 출처는 [조사 문서](change-evidence-data-sources.md)에 기록했다.

공식 변화 라벨을 탐지 결과 대신 입력하므로 **자체 변화탐지 모델의 예측·정확도 평가가 아니다**. 인스턴스 ID·동일 객체 대응·RGB 프레임·깊이·카메라 포즈는 이 배포 쌍에 없다. 객체별 이동·삭제·추가 확정, 가림·빈 공간 검증, 전후 키프레임은 아직 수행할 수 없다.

## 데이터와 계산

| 항목 | Epoch 0 | Epoch 1 |
|---|---:|---:|
| 원본 점 수 | 3,845,314 | 3,020,989 |
| 공식 변화 라벨 점 수 | 1,270,148 | 792,550 |
| 선택 영역의 전체 점 수 | 1,270,141 | 791,749 |
| 화면용 원본 대표점 수 | 249,369 | 178,724 |
| 원본 파일 크기 | 73,061,170 bytes | 57,398,995 bytes |

저자가 제공한 공통 Epoch 0 좌표를 보존한다. 화면 맞춤용 원점 이동은 두 회차에 같은 값을 적용하고 display_origin으로 기록한다.

1. 공식 변화 라벨 점들을 0.04 원본 좌표 단위의 복셀에서 26방향 연결 영역으로 묶는다. 1,000점 미만의 영역은 제외한다. 제외된 변화점은 총 808점이며 회차별 수를 기록한다. 연결 영역은 객체 인스턴스가 아니다.
2. 각 영역에서 0.02 단위 복셀의 첫 원본 포인트를 선택한다. 4,096점을 넘으면 정렬된 복셀 순서에서 균등 간격으로 선택한다. 표본은 모두 실제 원본 레코드다.
3. 측정 표본 각각에서 반대 회차 **전체 원본 점군**까지 정확한 최근접 거리를 계산한다. 표본 통계의 대상은 총 14,340점이다. 전체 영역점의 통계나 탐지 정확도로 해석하지 않는다.
4. 전체 영역의 원본 점 번호, 표본의 원본 점 번호·반대 회차 최근접 점 번호·거리, 파일 SHA-256과 설정을 저장한다.
5. 화면은 0.01 단위 복셀의 원본 대표점으로 표시한다. 거리 계산은 화면용 점군을 참조하지 않는다.

거리 단위는 PLY 헤더에 선언되어 있지 않아 **원본 좌표 단위**로 표시한다. 임계값 0.05는 이 데모의 검토 기준이며 데이터셋 공식 평가 기준이나 일반화된 판정값이 아니다.

| 영역 | 라벨 출처 회차 | 전체 점 수 | 측정 표본 | 표본 거리 중앙값 | 0.05 초과 표본 |
|---|---|---:|---:|---:|---:|
| 1 | 이전 | 783,246 | 4,096 | 0.470729 | 92.7% |
| 2 | 이전 | 486,895 | 4,096 | 0.085324 | 65.6% |
| 3 | 이후 | 563,515 | 3,960 | 0.096873 | 71.1% |
| 4 | 이후 | 212,440 | 1,536 | 0.108718 | 83.4% |
| 5 | 이후 | 15,794 | 652 | 0.016878 | 19.6% |

표본의 절반 이상이 임계값을 넘으면 ‘기하 차이 근거 있음’, 그렇지 않으면 ‘겹치는 표면 있음’으로 표시한다. 모두 최종 상태는 **추가 검증 필요**다. 점이 없다는 사실만으로 실제 제거를 증명하지 않는다. 최근접 점 번호는 동일 객체의 대응점이 아니다.

## 실행 및 재현

저장소 루트 PowerShell에서 실행한다. Python 3.10 이상과 NumPy·SciPy가 필요하며 기존 trajectory 가상환경에는 설치되어 있다. 새 Python 환경에서는 아래 requirements 파일을 설치하고 명령의 가상환경 Python 경로를 해당 환경의 python으로 바꾼다. 비교 페이지는 영상이나 FFmpeg 없이도 실행된다.

~~~powershell
# 새 환경에서만 설치
python -m pip install -r tools/requirements-change-evidence.txt

# 공개 공유 링크 익명 세션으로 실제 라벨 포함 PLY 2개 다운로드
./tools/download_multichange3d.ps1

# 근거 생성 및 원본 대조
./apps/autotour360/.venv/Scripts/python.exe tools/build_change_evidence_demo.py
./apps/autotour360/.venv/Scripts/python.exe tools/build_change_evidence_demo.py --verify

# 기존 뷰어 서버에서 제공
./apps/autotour360/.venv/Scripts/python.exe apps/autotour360/server.py
~~~

브라우저에서 http://127.0.0.1:8766/changes.html 을 연다. 생성 도구의 --data-dir, --output-dir, --threshold, --region-voxel, --render-voxel, --min-points로 조건을 바꿀 수 있다. 별도 출력 폴더는 서버의 --evidence-dir에 전달한다. 결과표는 기본 설정이다.

공식 공개 공유 링크가 유효한 동안 다운로드를 재현할 수 있다. 직접 다운로드 URL에는 익명 공유 세션 쿠키가 필요하다. 데이터는 Git에서 제외된 data/change_evidence/, 결과는 outputs/change_evidence_demo/에 보관한다.

## 근거 출력 구조

공통 파이프라인 artifact 규약과 분리된 데모 형식 kind=change_evidence_demo, version=1이다.

- evidence.json: 출처, 해시, 좌표계, 설정, 전체 점 수·제외점 수, 영역, 표본 통계, 검증 불가 항목.
- before.bin / after.bin: 화면용 원본 대표점. 레코드당 20 bytes: float32 XYZ(공통 원점 기준), uint8 RGB, uint8 영역 번호, uint32 원본 점 번호.
- region-NNN.indices.bin: 영역 **전체** 원본 점 번호. little-endian uint32, 0부터 시작.
- region-NNN.bin: **측정 표본**의 원본 점 번호(uint32), 반대 회차 최근접 점 번호(uint32), 거리(float32). little-endian 12 bytes.

원본 PLY는 element params, XYZRGB, float32 cd_type을 갖고 한 레코드는 19 bytes다. 원본 점 번호는 회차별 PLY 레코드 순서다.

## 파이프라인에 연결할 다음 입력

실제 시스템에서는 공식 라벨 대신 Task 5a의 changes.json과 전후 객체 라벨·공통 좌표 변환을 받는다. 인스턴스 대응으로 이동을 검증하고 RGB-D·카메라 포즈로 관측 범위·가림을 확인한 뒤 전후 키프레임을 연결한다. [근거 생성 설계](change-evidence-design.md)를 기준으로 어댑터를 추가해야 한다. 이 데모는 공통 스키마나 Task 5a/5b 구현을 변경하지 않는다.

데이터 출처: MultiChange3D authors, 3DOM / Fondazione Bruno Kessler. 데이터 라이선스 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

검증: 근거 표본 14,340건의 원본 좌표·저장 거리 일치, 회차별 원본 파일 해시, 전체 영역 점 번호, 공통 원점의 표시 좌표를 확인했다. 영역마다 32개 표본은 원본 전체 점군에서 최근접 검색을 다시 실행했다. 별도 도구 테스트 5개와 기존 앱 Python 테스트 14개·Node 테스트 30개가 통과했다. 도구 테스트는 공통 스키마 테스트의 의존성을 늘리지 않도록 tools/tests에 둔다.
