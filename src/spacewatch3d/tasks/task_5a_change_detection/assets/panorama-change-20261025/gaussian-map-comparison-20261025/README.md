# 가우시안 맵 입력 · 메쉬 기준선 / SimSaC 비교

`dorm1_3dgs_integration`은 공통 Nerfstudio 코드 패키지다. 실제 비교 대상은 그 코드로 각각 학습한 Dorm2·Dorm3 두 맵이며 원본 영상 SHA-256, 데이터 receipt, checkpoint SHA-256이 모두 다르다. 각각 60,000 step, 386,826 / 369,446 Gaussians다. 같은 맵의 체크포인트 두 개를 전후로 사용하지 않는다.

- 원본 모델: `/workspace/v1_splatfacto_comparison_20261025/models/{dorm2,dorm3}/splatfacto/linked_recipe/nerfstudio_models/step-000059999.ckpt`
- 원본 코드 패키지: `/workspace/v1_splatfacto_comparison_20261025/vendor/dorm1_3dgs_integration`
- 현재 실행 산출물: `/workspace/gaussian_change_20261025/published`

`prepare.py`는 기존 메쉬 정합·오차·관측검증 판정을 그대로 읽는다. 기존 `native_mesh_to_canonical`과 `mesh_to_glb.npy`를 합성하고 233 / 248개 원본 카메라에서 위치·방향을 확인한다. 새 정합을 피팅하지 않는다. 기준선 박스는 `event_groups.entry_ids` 트랙의 원본 면만 사용하며 그룹 문맥 전체를 양성으로 만들지 않는다.

`render_maps.py`는 원래 학습된 모든 Gaussian을 SH3 PLY로 보존한다. 후 맵의 고정 구역 소속에 따라 13개 파일로 나누고 각 구역에 기존 정합·이동 편향을 적용한다. 평균·공분산의 회전·축척을 함께 변환하며 SH는 각 Gaussian의 원래 좌표계에서 평가한다. 기존 구역 경계 단차는 남는다. 두 맵을 각 구역의 **같은 카메라**에서 렌더링한다. 사진 선명화나 원본 RGB 덧씌우기는 없다.

`infer_simsac.py`는 기존 공개 synthetic+ChangeSim 가중치를 strict 로드해 새 Gaussian RGB 렌더에 양방향 추론한다. 전체 ERP를 기존 520/256 branch 전처리로 사용한다. 0.5 임계값은 유지한다. 13개 동일 입력 대조에서 후보 0을 확인했다. 6개 cube 렌더의 YOLO11l-seg 사람 마스크와 각 변 10% 여유 박스를 양쪽 합집합으로 제외한다. 양쪽 alpha ≥ 0.85, 유한한 예상 깊이, 고정 공간 구역만 표시 근거로 사용하며, 제외 픽셀을 미관측·구역 밖·사람으로 따로 기록한다. 기존 양성 메쉬로 SimSaC 후보를 제한하지 않는다. 사람 여유 박스는 주변 표면도 제외할 수 있고, 흐린 Gaussian 렌더에서 사람 검출 오류가 날 수 있다.

3D 박스는 Gaussian 예상 깊이로 연결한 후보점의 0.06h voxel 연결 성분이며 0.02h 여유를 둔다. 모든 보존 후보 픽셀이 한 박스에 포함되는지 검사한다. 박스를 실제 객체 ID·이동·변화 정답으로 해석하지 않는다. 현재 기존 후보 2묶음, 사람 제외 후 SimSaC 5영역이다.

브라우저는 [Spark 2.4.0](https://sparkjs.dev/docs/)와 Three.js 0.180.0으로 **실제 Gaussian PLY**를 표시한다. 원본 Gaussian 386,826 / 369,446개를 모두 유지한다. 처음은 전 맵 하나이고, 박스를 선택하면 해당 3D volume에서 전 가우시안을 흐리게 하고 후 가우시안을 겹친다. 전/후 맵 전환, 시선·위치 이동, 전체 화면, 영역 내 0–100% 겹침을 지원한다. 원본 사진 파노라마를 Gaussian 맵처럼 표시하지 않는다.

이 실험은 서로 다른 층의 공간 차이 파일럿이다. 이전 RGB 실험 20프레임·36쌍을 그대로 재사용한 통제 평가가 아니다. 13구역별 기존 카메라 중 기하 중심/고정 메쉬 후보에 가까운 위치를 미리 선택했다. 후보 개수를 정확도나 성능 향상으로 해석하지 않는다. 재구성 차이·정합 잔차·가림, Gaussian 예상 깊이의 편향이 남는다.

## 확인

```powershell
python tools/gaussian_change/validate.py
python -m unittest discover -s tests -p test_gaussian_map_bridge.py -v
```

저장소 스냅샷에서 대형 원본 없이 메타데이터·입력 이미지·후보 마스크만 확인할 때는 `validate.py --directory <뷰어 폴더> --metadata-only`를 쓴다. 전체 Gaussian PLY와 float32 확률 NPZ는 Git에서 제외하고 SHA-256과 경로를 기록한다. 전체 검증은 이 파일들과 보존된 메쉬 입력을 요구한다.

실행 환경은 기존 서버의 `v1_splatfacto_comparison_20261025/env.sh`(gsplat 1.4.0)와 `sam31_panorama_test_20261023/env/bin/python`(SimSaC)을 분리했다. 사람 검출 코드는 기존 `image-lifecycle-v4/deps`를 읽는다. 원본 모델과 메쉬 파일은 수정하지 않으며 실행 결과는 새 디렉터리에 쓴다.
