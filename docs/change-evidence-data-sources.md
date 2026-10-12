# 실제 전후 점군 비교 데모에 사용할 데이터

확인일: 2026년 10월 12일

## 선택: MultiChange3D Open Space (RGB-D), Epoch 0–1

실제 실내 공간을 별도로 재촬영한 전후 점군이며, 공식 GitHub의 작은 미리보기 쌍과 공식 공유 폴더의 변화 라벨이 포함된 전체 쌍을 모두 확인했다. 두 점군의 공간 차이와 원본 점 출처를 보여주는 데 사용할 수 있다. 인스턴스 대응과 원본 촬영 프레임은 이 배포 파일에 없으므로, 객체 동일성·가림·빈 공간 검증은 별도 입력이 필요하다. [공식 프로젝트](https://3dom-fbk.github.io/multichange3d/), [공식 저장소](https://github.com/3DOM-FBK/multichange3d)

### 실제 취득과 좌표계

논문 §3.1에 따르면 Office와 Open Space RGB-D 장면은 Intel RealSense D435i의 1280×720 RGB-D 관측에서 BAD-SLAM으로 복원한 surfel을 점군으로 내보낸 것이다. 물체의 자세 변경·추가·제거를 실제 공간에서 유도했으며 병·노트북·가구 등이 예로 제시된다. 특정 미리보기 쌍의 각 물체가 무엇인지까지 이름 붙인 목록은 확인되지 않았다. [저자 제공 논문, §3.1](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/paper/Wang_et_al_2026.pdf)

논문 §3.2에서는 모든 회차를 Epoch 0의 지역 기준 좌표계로 변환한다고 설명한다. RGB-D 정합은 수동 대응점 초기 정합 후, 변하지 않은 영역만 사용한 ICP로 보정한다. 표와 그림에서는 거리·밀도 단위를 미터로 제시한다. PLY 헤더 자체에는 단위나 카메라 좌표계 선언이 없다. 데모에서는 제공된 공통 좌표를 보존하고, 화면 맞춤용 원점 이동이 필요하면 **두 회차에 같은 변환**을 적용한다. [저자 제공 논문, §3.2](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/paper/Wang_et_al_2026.pdf)

### GitHub 미리보기: 3.23 MB

확인한 공식 저장소 커밋은 `c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48`이다. 아래 링크는 커밋으로 고정했다. [공식 커밋](https://github.com/3DOM-FBK/multichange3d/commit/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48)

| 회차 | 파일 크기 | 점 수 | 다운로드 |
|---|---:|---:|---|
| Epoch 0 | 1,863,835 bytes | 124,235 | [0.ply](https://raw.githubusercontent.com/3DOM-FBK/multichange3d/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/clouds/open_space_rgbd/0.ply) |
| Epoch 1 | 1,369,074 bytes | 91,251 | [1.ply](https://raw.githubusercontent.com/3DOM-FBK/multichange3d/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/clouds/open_space_rgbd/1.ply) |

두 파일의 헤더를 직접 읽어 확인한 구조는 `binary_little_endian`, `element vertex`, `float x/y/z`, `uchar red/green/blue`이다. 인스턴스 ID·변화 라벨·포즈·깊이는 없다. 공식 웹뷰어가 같은 경로의 파일을 Epoch 0과 Epoch 1로 표시한다. 미리보기의 정확한 다운샘플 방식은 배포 문서에 설명되어 있지 않다. [공식 웹뷰어 소스](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/js/pointcloud-viewer.js)

### 전체 변화 라벨 쌍: 130.46 MB

공식 [Open Space 공유 폴더](https://fbk.sharepoint.com/:f:/s/BENCHMARKS/IgCzHpZzgXjDQI1O1DeZniyZAatqJWchrICnsM1KF7eV2og?e=NCmHOp)의 `pair_01`은 로그인 없이 HTTP 세션으로 열람할 수 있었다. 웹 검색 도구에서는 로그인으로 리디렉션되는 경우가 있어, 공개 공유 링크의 익명 쿠키 세션을 먼저 생성해서 확인했다. 계정 로그인·추가 약관 동의는 수행하지 않았다.

| 파일 | 파일 크기 | 점 수 | 공개 공유 세션에서 사용하는 다운로드 경로 |
|---|---:|---:|---|
| `pointCloud0.ply` | 73,061,170 bytes | 3,845,314 | [Epoch 0 다운로드](https://fbk.sharepoint.com/sites/BENCHMARKS/_layouts/15/download.aspx?UniqueId=dabb523a-ca5b-43fb-9ced-b7a0a5e9f90e) |
| `pointCloud1.ply` | 57,398,995 bytes | 3,020,989 | [Epoch 1 다운로드](https://fbk.sharepoint.com/sites/BENCHMARKS/_layouts/15/download.aspx?UniqueId=13d765f5-45dc-484f-9ca6-8f60c8b6ac3b) |

두 파일에 0–1023 bytes의 HTTP Range 요청을 보내 `206 Partial Content`와 전체 크기를 확인했다. 전체 파일은 이 조사에서 다운로드하지 않았다. 헤더는 `binary_little_endian`, **`element params`**, `float x/y/z`, `uchar red/green/blue`, `float cd_type`이다. 점당 19 bytes이며 헤더 이후에는 같은 구조의 레코드가 이어진다. 파서는 요소 이름을 `vertex`로 한정하지 않아야 한다. [공식 전체 데이터 폴더](https://fbk.sharepoint.com/:f:/s/BENCHMARKS/IgCzHpZzgXjDQI1O1DeZniyZAatqJWchrICnsM1KF7eV2og?e=NCmHOp)

공식 준비 스크립트에서는 `cd_type=0`은 변화 없음, `1`은 변화 있음으로 정의한다. 이 값은 객체 ID나 이동 대응 ID가 아니다. 전후 점군의 어느 쪽에 속하는지와 결합해 변화 영역을 비교할 수 있으나, 라벨만으로 같은 물체의 이동이나 실제 빈 공간을 증명하지는 못한다. [공식 라벨 준비 스크립트](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/scripts/prepare_gt_labels.py)

익명 접근 재현 방법:

```powershell
$shareUrl = 'https://fbk.sharepoint.com/:f:/s/BENCHMARKS/IgCzHpZzgXjDQI1O1DeZniyZAatqJWchrICnsM1KF7eV2og?e=NCmHOp'
$shareResponse = Invoke-WebRequest -Uri $shareUrl -SessionVariable changeDataSession
# 이후 위 다운로드 URL에 -WebSession $changeDataSession을 전달한다.
# 세션 없이 다운로드 URL에 바로 접근했을 때는 403 응답을 확인했다.
```

`pair_01`의 공개 목록에는 위 두 PLY만 있다. RGB 이미지·깊이·포즈·원본 키프레임·인스턴스 라벨은 별도로 제공되지 않았다. 프로젝트의 `qualitative_open_space.png`는 **Epoch 0–2** 비교 그림이므로 이 데모의 0–1 원본 키프레임으로 연결하면 안 된다. [공식 프로젝트 비교 그림 설명](https://3dom-fbk.github.io/multichange3d/)

### 사용과 표시 범위

- 미리보기만 사용하는 경우: 거리 임계값과 연결 영역으로 공간 변화 후보를 만든다. 공식 정답 라벨과의 성능 검증을 했다고 표시하지 않는다.
- 전체 라벨 쌍을 사용하는 경우: 공식 `cd_type`으로 지정된 변화 영역을 추출하고, 전후 실제 점의 위치 차이와 출처를 묶는다. 같은 물체의 이동을 확정하는 인스턴스 대응은 추가 입력이 필요하다.
- 제거·추가 확정에는 카메라와 깊이로 관측 범위·가림을 검증해야 한다. 해당 입력이 없으면 근거 부족으로 표시한다.
- 데이터 라이선스는 **CC BY-NC-SA 4.0**, 저장소 코드 라이선스는 **MIT**이다. 파생 데이터나 화면에는 저자·데이터셋·라이선스 출처를 유지한다. [공식 라이선스 안내](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/readme.md#license)

## 대안

**MultiChange3D Classroom:** 실제 Leica RTC360 전후 실내 스캔이며 작은 공식 미리보기는 [0.ply](https://raw.githubusercontent.com/3DOM-FBK/multichange3d/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/clouds/classroom/0.ply) 1,759,677 bytes, [1.ply](https://raw.githubusercontent.com/3DOM-FBK/multichange3d/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/clouds/classroom/1.ply) 1,712,424 bytes이다. 논문은 가구 등 물체 추가·제거를 설명한다. 미리보기의 인스턴스 대응·관측 검증 입력은 역시 없다. [저자 제공 논문, §3.1](https://github.com/3DOM-FBK/multichange3d/blob/c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48/docs/assets/paper/Wang_et_al_2026.pdf)

**3RScan:** 실제 재촬영, 전후에 일관된 인스턴스 ID, 객체별 변화 변환, RGB-D와 카메라 포즈가 있어 후속 객체 단위 검증에 적합하다. 그러나 공식 다운로드는 Terms of Use 양식 제출을 요구하므로 이번 익명 소량 데모에서는 선택하지 않았다. 별도의 양식 제출이나 승인 요청은 수행하지 않았다. [공식 3RScan 데이터 구조](https://github.com/WaldJohannaU/3RScan), [공식 다운로드 안내](https://waldjohannau.github.io/RIO/)
