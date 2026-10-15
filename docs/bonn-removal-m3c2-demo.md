# Bonn RGB-D 상자 제거: 실제 다른 데이터셋 적용

2026-10-15. 기존 Rescan `scene_e_0 → scene_e_2`와 다른 데이터셋인 **Bonn RGB-D Dynamic Dataset**에서 수행했다. 같은 실제 영상의 상자 제거 전후 두 시간 구간을 각각 복원했다. 별도 방문으로 재촬영한 두 스캔은 아니며, 이 차이를 화면에도 표시한다. 객체 분할·인스턴스·변화 정답 라벨을 탐지에 사용하지 않았다.

## 원본과 카메라

- 공식 [Bonn 배포 페이지](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)의 `rgbd_bonn_removing_nonobstructing_box`.
- [공식 ZIP](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_removing_nonobstructing_box.zip) 271,656,752 bytes. SHA256 `a1d4731b6d0a60d50009c4c401336a23957c5182e3be79fd8e7028e9dde1db8d`.
- 원본 494 RGB + 494 depth, 카메라 GT 포즈 496개. 입력은 `data/bonn/removing_nonobstructing_box/`에 보존했다.
- 전: 시작 후 1–4.5초, 실제 RGB-D 21프레임. 후: 11.5–16.3초, 29프레임. 사람의 상자 조작 중간 구간은 합성하지 않았다.
- 원본 depth / 5000, RGB에 등록된 depth, 공식 intrinsics/distortion. 제공 GT에 Bonn 보정 `S @ GT @ S @ M`을 적용했다. M의 polar rotation과 균일 배율 1.05929045를 분리해 rigid TSDF pose와 depth 배율에 넣었다. 근거와 좌표 검사 수치는 [자료 조사](alternate-dataset-access-20261015.md)에 있다.
- 두 메쉬와 모든 카메라에 **동일한** 강체 표시 변환을 적용했다. 하단 영상 영역의 평면은 표시 방향을 정하는 데만 쓰며, 변화 검출에서 바닥을 제외하는 mask로 쓰지 않는다.
- TSDF voxel 1.5cm / truncation 6cm. 전 177,905 / 후 243,403 삼각형. 메쉬가 정답으로 제공된 것이 아니라 실제 RGB-D에서 새로 복원한 메쉬다.

## 계산과 결과

Rescan에서 수정한 `mesh_change_noise.py`의 기본값을 **변경하지 않고** 사용했다: 8/16/28cm 반경, 각 맵 2cm 정합 오차 가정, 표면 spread에 표본 수 나눗셈 없음, 같은 방향으로 2개 이상 크기에서 6cm 이상의 오차 초과, 패치 55%, 자기 depth 지지 40%. 반대 시점 빈 공간 55%·2회 이상·15cm baseline, 고유 강체 대응 조건도 같다.

카메라·데이터 형식의 차이만 adapter가 처리한다. 원본 depth를 undistort한 내부 깊이 영상에서 관측을 확인한다. 사진 상자는 보정 포즈로 투영한 뒤 RGB distortion을 되돌려 **공간 보정하지 않은 원본 JPEG 변환 프레임**에 표시한다. 원본 PNG 파일명과 SHA256도 프레임 metadata에 기록했다.

- 공유 M3C2 코어 4,576개.
- 명목 M3C2 변화/의심 코어 1,742개, 실제 mesh 거리 조건 후 후보 코어 1,015개.
- 다중 크기 오차 기준을 넘는 코어 315개.
- 요철·관측 지지 조건에서 제외한 패치 10개. 이것을 모두 검증된 오탐이라고 부르지 않는다.
- 최종 **소실 1 / 추가 0 / 이동 0 / 유형 미정 0**. 원본 프레임을 전후 4장씩 연결했다.
- 소실 구역 코어 67개. 자기 depth 지지 **95.5%**, 반대 시점 빈 공간 지지 **100%**, 반대 시점 지지 카메라 29개, baseline 약 **30cm**.
- 원래 후보 패치의 **73.6%**가 다중 크기 기준 통과. robust 표면 거리 중앙값 **28.6cm**, 요철·정합 오차 범위 중앙값 **14.2cm**. 이 거리값은 상자 이동 거리나 부피가 아니며, 두 요약 중앙값의 비율이 전체 score 중앙값과 같을 필요도 없다.

가장 앞에 표시한 원본은 전 프레임 40 (**1.34초**) / 후 프레임 360 (**12.07초**)다. 전 원본에는 바닥 상자가 있고, 후에는 같은 투영 위치에 빈 바닥이 보이는 것을 직접 확인했다. 이 확인은 라벨을 탐지 입력으로 쓴 것이 아니라 산출물을 원본으로 검토한 것이다. 모든 실내 데이터에 대한 정량 정확도를 확정하는 실험은 아니다.

## 실행과 뷰어

```powershell
python tools/run_bonn_demo.py `
  --data data/bonn/removing_nonobstructing_box/rgbd_bonn_removing_nonobstructing_box `
  --output outputs/bonn-removal-new-run
```

과학 계산 환경은 `tools/mesh-comparison-requirements.txt`를 사용한다. 출력 경로는 실행마다 새로 지정한다. 현재 실제 출력은 `outputs/bonn-removal-20261015/`의 `prepared/`, `frames/`, `comparison/`, `detected/`, `published/`다. source depth/카메라 중간 파일은 로컬에 보존하며 웹 라우트는 `published`의 결과 JSON·geometry/mask binary·원본 JPEG만 제공한다.

현재 서버 기본값은 그 `published`를 사용한다. 다른 실행 출력은 `server.py --rgbd-dir <new-run>/published`로 지정한다. 기존 Rescan 경로는 보존한다.

[실제 뷰어](http://127.0.0.1:8768/mesh-comparison.html?dataset=bonn-removal). 상단에서 **Bonn RGB-D · 상자 제거 / Rescan · 방 재촬영**을 전환할 수 있다. Bonn에서는 **소실 1**을 선택해 전후 원본, 이전/이후 메쉬, M3C2와 GLS 참고 지도를 본다. 소실은 이전 메쉬를 먼저 열어 실제 표면을 보여준다. 이후 메쉬에서는 과거 대상 부피의 윤곽과 현재 빈 바닥을 비교한다. GLS의 유형 표시도 M3C2 통합 결과를 참조하며, 별도 GLS 객체 분류를 실행했다는 뜻은 아니다.

데이터 출처: Palazzolo, Behley, Lottes, Giguère, Stachniss, **ReFusion**, IROS 2019 / University of Bonn. 별도 데이터 라이선스가 명시되지 않은 배포물에 코드 라이선스를 임의 적용하지 않는다.
