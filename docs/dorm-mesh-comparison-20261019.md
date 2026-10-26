# Dorm2 → Dorm3 메시 자체 비교 · 2026-10-19

사용자가 같은 공간의 전후 촬영으로 확인한 실제 입력을 사용했다. 원본은 수정하지 않았다.

| 입력 | 서버 경로 | 정점 / 삼각형 | 원본 파노라마·추정 자세 |
|---|---|---|---|
| Dorm2 | `/workspace/dorm2_2d3d_v1_object_mesh_20261017/meshes/scene.ply` | 1,953,934 / 3,303,609 | 233개 |
| Dorm3 | `/workspace/dorm3_mesh_generator_v1_20261017/meshes/scene.ply` | 2,006,783 / 3,447,635 | 248개 |

사진은 각각 `/workspace/dorm2_pointcloud_v1_20261017/input/frames/`, `/workspace/dorm3_repro_validation_20261016/input/frames/`의 JPEG를 그대로 복사했다. 원본 사진 SHA256를 확인해 연결하며, 사진은 1036×518 equirectangular 파노라마다. 추정 카메라는 각 메시 폴더의 `cameras/poses.json`을 사용한다.

## 처리

1. 메시는 XYZ·삼각형으로 읽고 면 법선을 계산한다. RGB는 표시와 바닥·벽 사진 대응에 사용한다. 객체·클래스 ID와 `objects.json`, 변화 정답은 정합·탐지에 읽지 않는다. 입력 메시가 이전 semantic 분할을 거쳐 만들어졌다는 사실까지 없어지는 것은 아니다.
2. 넓은 수평면과 높이가 있는 수직면을 기하로 찾는다. 낮은 수평면을 바닥, 높은 공통 수평면을 천장 후보로 삼고 큰 벽과 함께 상대 Sim(3) 초기화를 구한다. 모든 바닥·벽이 실제로 불변이라는 정답은 주어져 있지 않다.
3. 복도의 거의 평행한 벽만으로는 복도 방향 대응이 모호했다. 기하 기준 표면에 원본 사진의 광선이 닿는 SIFT 특징만 사용해 다른 초기 변환을 구했다. 객체 대응은 사용하지 않았다.
4. 별도로 남긴 사진에서 두 고정 가설을 비교했다. 기하 초기화는 0개, 사진 유도 가설은 9개의 3D 대응이 확인됐다. 사진 유도 초기 대응은 20개다. 미리 정한 별도 확인 최소 15개를 넘지 못했으므로 어느 가설도 검증된 정합으로 채택하지 않았다. 사진 지지가 더 있는 가설을 **임시 미리보기**로만 사용한다.
5. 원본 전체 삼각형에 대한 양방향 최근접 표면 거리를 계산한다. 세 반경에서 국소 거리 중앙값·법선 방향 요철을 구하고, 공통 구조면의 공간 셀별 잔차 P95로 구한 오차 기준과 비교한다. 최소 차이·여러 크기·연결 면적 조건을 함께 적용한다. 정식 M3C2 실행 결과나 보정된 변화 확률이 아니다.
6. 후보점에서 원래 촬영 위치의 광선을 검사한다. 자기 촬영에서 표면이 보이고 상대 촬영에서는 더 먼 표면이 확인되는 경우를 최소 두 시점에서 요구한다. 상대 메시 광선이 아무 면도 만나지 않으면 빈 공간 정답으로 간주하지 않고 미확인으로 둔다. 이 검사는 재구성 메시와 추정 카메라에 의존하며 독립적인 실제 깊이 검증은 아니다.
7. 지도는 경량화한 표시용 메시를 사용하고, 거리 계산은 원본 전체 메시를 사용한다. 카메라 중심에는 상대 축척을 적용하지만 카메라 회전축은 단위 길이를 유지한다. PanoVGGT의 OpenCV 축·equirectangular 투영을 사용하며 경도 경계에 걸린 표시 박스는 나눠 그린다.

## 현재 미리보기

- Dorm3 → Dorm2 상대 축척: `0.9714285860`배. 실측 미터 축척은 아니다.
- 구조면 거리 기준 내 대응·겹침 비율: 약 `43.49%`. 전체 지도 정확도나 변화 탐지 정확도가 아니다.
- 구조면 법선 잔차 중앙값: `0.02806 u`. 공간 셀 잔차의 보수적 P95 기준: `0.114724 u`.
- 공통 기준: 10,968대응·1,293공간 셀. 샘플 수가 많다고 오차를 `sqrt(n)`으로 줄이지 않는다.
- 후보 18구역·8,591코어, 전후 사진 연결 18구역·서로 다른 원본 사진 103장.
- `registration_status=provisional_corridor_ambiguity`, `independent_photo_check_passed=false`를 저장하고 뷰어 상단에 표시한다. 원래 기하 초기화와 실패한 사진 확인 기록도 보존한다.

18개는 객체 개수·정답 변화 개수·추가/소실/이동 자동 분류가 아니다. 긴 복도에서 서로 다른 위치를 대응했을 가능성과 국소적인 비강체 재구성 왜곡이 남아 있으므로, 주황 구역을 모두 실제 변화라고 해석하면 안 된다. 임시 정합으로 인해 의미가 불확실한 결과를 성공률로 보고하지 않는다.

## 코드·실행·산출물

계산 코드는 `tools/dorm_mesh_geometry.py`, `tools/run_dorm_mesh_comparison.py`, `tools/refine_dorm_structure_photos.py`, `tools/prepare_dorm_alignment_preview.py`다. numpy·scipy·Open3D·Pillow가 필요하며 사진 특징 계산에는 OpenCV를 사용한다. 실제 실행은 서버에서 CPU 4스레드로 수행했으며 GPU를 점유하지 않았다. 원본 계산 환경은 변경하지 않고 OpenCV 4.11.0.86만 별도 작업 폴더에 설치했다.

서버 산출물 루트는 `/workspace/dorm2_dorm3_mesh_changes_20261019/`다. `run-v1`은 기하 초기화, `run-v2`는 관측 범위 선별을 추가한 비교, `run-v3/4`는 사진 초기화 시도, `photo-check-v5`는 별도 사진 확인, `run-v6`은 명시적인 임시 정합 미리보기다. 원본 두 메시 폴더에는 쓰지 않았다.

주요 실행 순서는 다음과 같다. 매번 새 출력 경로를 사용한다.

```bash
python tools/run_dorm_mesh_comparison.py --output <geometry-run>
python tools/refine_dorm_structure_photos.py --previous <geometry-run> --output <photo-run> --deps <opencv-dir>
python tools/refine_dorm_structure_photos.py --previous <geometry-run> --output <check-run> --deps <opencv-dir> --frame-offset 2 --check-seed <photo-run>/feature-alignment-audit.json
python tools/prepare_dorm_alignment_preview.py --geometry <geometry-run> --photo-audit <photo-run>/feature-alignment-audit.json --photo-check <check-run>/independent-photo-check.json --output <preview-run> --allow-provisional-preview
python tools/run_dorm_mesh_comparison.py --stage publish --output <preview-run>
```

사진 초기화가 최소 대응 수를 통과하지 못해도 진단용 `feature-alignment-audit.json`을 남긴다. 그 파일을 사용했다고 정합이 확인된 것으로 바꾸지 않는다. 공개 보고서는 `published/comparison.json`, 비공개 배열은 `detection.npz`, 변환·원본 SHA256는 `alignment.json`에 저장한다.

로컬 뷰어 산출물: `outputs/dorm2-dorm3-20261019/run-v6/published`. 실제 메시·사진·실행 결과는 Git에서 제외된다.

```bash
python apps/autotour360/server.py --port 8768 --rescan-dir outputs/rescan-three-changes --dorm-dir outputs/dorm2-dorm3-20261019/run-v6/published
```

`http://127.0.0.1:8768/residual-comparison.html?dataset=dorm2-dorm3`에서 기존 전후 메시 화면의 촬영 쌍을 전환한다. Auto Tour의 원래 영상·Rescan 이동 9/추가 2/소실 1은 유지한다.

검사: 상대 축척 복원·파노라마 투영·카메라 축 처리·가림/무관측 차단 Python 4개, 앱 Node 71개, 서버 경로·자료 분리 Python 10개 통과. 이는 구현 검사이며 실제 변화 정답 대비 성능 평가는 아니다.


## 2026-10-19 정합 전용 겹침 보기 (alignment-v8 / v9)

사용자가 변화 탐지 전에 원본 두 메시의 정합을 다시 요구하여, Dorm 메뉴의 기본 화면을 **정합 겹침 보기**로 변경했다. `residual-comparison.html?dataset=dorm2-dorm3`에서 Dorm2 주황 / Dorm3 청록을 같은 카메라로 합성한다. 각 메시의 자체 가림은 각각의 깊이 버퍼에서 계산한다. 보정 전 / 이전 임시 정합 / 새 구조 정합, 바닥·벽 분리 표시, 레이어 토글, 겹침 비율, 전체·국소 위치와 원본 사진을 제공한다. 새 변화 탐지는 실행하지 않았다. 기존 비교 산출물은 보존되어 `view=changes`로만 접근한다.

### 원본 연결 점검에서 확인된 중요한 미해결 사항

사용자는 같은 공간의 전후라고 재확인했지만, 원본 영상에서 직접 디코딩한 약 6초 장면에 **Dorm2 5F / Dorm3 3F**가 표시된다. 단순 뷰어 사진 연결 오류는 확인되지 않았다.

- 메시 provenance → 분할 `source_run` → 재구성 config → input bundle → MP4/프레임 manifest 경로가 각각 일치한다.
- 서버의 두 MP4 전체 SHA-256이 각각 입력 번들 및 재구성 config의 해시와 일치한다.
- 각 촬영의 첫·12번·마지막 표본 JPEG가 manifest와 입력 번들의 해시와 일치한다.
- 원본 재구성 카메라를 각 메시의 `source_to_world`로 변환한 값과 저장된 메시 카메라의 최대 원소 오차는 Dorm2 약 6.66e-16, Dorm3 약 9.43e-7이다.
- 실제 MP4의 180번 프레임(약 6초)을 디코딩해 원본 JPEG와 비교한 평균 픽셀 차이는 약 2.50 / 2.39 (0–255), 안내판의 5F/3F 표기는 직접 확인했다.

따라서 **기하 형상을 맞추는 결과**와 **동일한 물리적 표면을 확인한 결과**를 구별한다. 다른 층이라고 확정하지 않았고, 사용자의 설명과 영상 표기의 불일치를 해소하지 못했다. 이 상태에서 실제 추가·소실·이동으로 해석하지 않는다. 공개 정합 화면에도 이 사실을 표시한다.

### 재현한 정합 보정 결함과 수정

실제 표면을 `.12 u` 간격으로 고정 샘플링한 회귀 명령:

```powershell
python tools/check_dorm_alignment_refinement.py outputs/dorm2-dorm3-20261019
```

수정 전: 기존 임시 정합의 구조면 겹침 **0.41427 → 0.01137**, `Refinement destroys existing floor/wall overlap` assertion 실패. 평행한 복도에서 약한 방향의 Sim3 최소제곱 보정을 목적함수 확인 없이 적용했다.

수정 후: 같은 명령 **0.41427 → 0.41997**, 통과. 높이로 정규화한 설계행렬의 작은 특이값 방향을 제한하고, 보정 폭을 제한한 뒤, 대응 탈락도 비용에 포함하는 고정 표본 양방향 거리 목적함수에 대해 backtracking으로 악화 단계를 거절한다. 각 채택 단계의 비용 감소도 assertion으로 확인한다. 이 수정만으로 반복 복도의 물리적 대응이 해결되는 것은 아니다.

### 새 전체 구조 정합

전체 평면 배치의 raster 상관으로 180도 대칭을 포함한 회전·축척·이동 후보를 만들고, 국소 PCA 법선으로 얻은 바닥과 높이 방향으로 충분히 이어지는 벽 기하 표본으로 보정했다. 낮은 가구의 수평면은 바닥 높이 범위에서 제외하고, 벽은 약 0.43×기준 높이 이상의 수직 지지를 요구한다. 클래스를 읽어 가구를 제외하지 않으므로 높은 가구가 벽처럼 평평할 때 완전한 분리를 보장하지 않는다. 수동 사진 대응·객체/변화 라벨을 사용하지 않았다. 표면을 휘거나 바닥을 평탄화하지 않고 원본에 전역 Sim3만 적용한다.

동일한 구조 표본과 조건으로 비교한 수치:

| 상태 | 전체 겹침 | 바닥 겹침 | 벽 겹침 | 바닥 3D 거리 중앙값 | 벽 3D 거리 중앙값 |
|---|---:|---:|---:|---:|---:|
| 좌표 보정 전 | 7.1% | 15.2% | 3.3% | 0.675 u | 0.712 u |
| 이전 임시 정합 | 23.4% | 19.2% | 25.4% | 2.123 u | 2.779 u |
| 새 구조 정합 | 45.6% | 54.6% | 41.4% | 0.110 u | 0.137 u |

겹침: 양방향 최근접 **표본** 3D 거리 ≤ 기준 높이 2.485 u의 5% = 0.12425 u, 법선 절댓값 내적 >0.85. 표본 수 전 41,957 / 후 42,111. 새 Dorm3→Dorm2 축척 0.9774415. u는 실측 미터가 아니다. 중앙값/P95는 관측 범위가 다른 부분도 포함하며, 표본 간격의 영향이 있다. 과거 추출된 일부 RANSAC 면의 43.5%와 모집단이 다르므로 그 숫자와 직접 비교하지 않는다. 원본 삼각형 전체에 대한 오차 수치가 아니다.

전체 배치는 개선됐으나 복도의 서로 다른 구간 및 끝부분의 오차가 남았고 동일 장소 확인도 끝나지 않았다. **정합 완료·실제 전후 변화 검증 완료로 보고하지 않는다.**

재현 (각 output 경로는 새로 지정):

```powershell
python tools/align_dorm_mesh_structure.py outputs/dorm2-dorm3-20261019 outputs/dorm2-dorm3-20261019/alignment-v8
python tools/refine_dorm_whole_structure.py outputs/dorm2-dorm3-20261019 outputs/dorm2-dorm3-20261019/alignment-v8 outputs/dorm2-dorm3-20261019/alignment-v9
python tools/publish_dorm_alignment_view.py outputs/dorm2-dorm3-20261019 outputs/dorm2-dorm3-20261019/alignment-v9
```

실제 입력/출력: Git에서 제외한 `outputs/dorm2-dorm3-20261019/inspection/alignment-samples.npz`, `alignment-v7/input-copy`, `alignment-v7/source-link-audit.json`, `alignment-v8`, `alignment-v9/published`. 원본 파일은 변경하지 않았다. 새 계산은 로컬 CPU, GPU 사용 없음. 서버 연결 점검은 원본 읽기와 별도 출력 폴더 작성만 수행했다.


정합 갱신 후 확인: 기하 테스트 5개(낮은 가구가 바닥·높은 벽 기준에 섞이지 않는 합성 사례 포함), 실제 Dorm 보정 회귀 루프, HTTP 경로·범위/격리 테스트 10개, Chrome WebGL 겹침 렌더링·3개 정합 전환·표면 분리·원본 사진 로드·모바일 폭 확인을 통과했다. 기존 원본 영상 2개가 계속 제공되는 것도 확인했다. 이는 구현 동작 확인이며 정합 정확도 또는 동일 장소 확인의 성공을 의미하지 않는다. 스크린샷은 앱의 `.preview/dorm-alignment-overlap.png`, `dorm-alignment-floor.png`, `dorm-alignment-wall.png`, `dorm-alignment-top-final.png`에 있다. 커밋/푸시는 수행하지 않았다.
