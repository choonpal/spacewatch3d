# 360° 가상투어

`apps/autotour360/`는 입력한 360도 영상의 촬영 동선을 추정하고, 영상 위 경로나 미니맵을 클릭해 해당 촬영 시각으로 이동하는 웹 앱이다. 설치와 실행 방법은 [앱 README](../apps/autotour360/README.md)에 있다.

## 저장소에서의 역할

기존 `src/spacewatch3d/tasks/`는 프레임 추출·3D 재구성·객체 분할·변화탐지 단계별 분석 코드다. 가상투어 앱은 자체 서버와 브라우저 화면을 가지며 별도의 `.venv`로 실행한다. 공통 Python 패키지의 의존성·task registry·artifact schema를 변경하지 않는다.

가상투어의 `trajectory.json`과 `.tour.json`은 앱 전용 형식이다. 현재 공통 artifact validator에 입력하는 형식으로 변환되어 있지 않다. 기존 task의 카메라 결과를 연결하려면 촬영 시각, 좌표축, camera-to-world 회전, 상대 스케일 및 바닥 평면을 맞추는 adapter가 필요하다.

## 디렉터리

| 위치 | 내용 | Git 관리 |
| --- | --- | --- |
| `apps/autotour360/` | 서버·뷰어·경로 추정·실행 스크립트·테스트 | 예 |
| `apps/autotour360/.venv/` | 앱 전용 Python 패키지 | 아니요; 의존성 목록으로 재설치 |
| `data/virtual_tour/<sha256>/` | 입력 영상 복사본, `source.json` | 아니요 |
| `outputs/virtual_tour/analysis/<sha256>/` | `trajectory.json`, 상태·로그, `work/` | 아니요 |
| `outputs/virtual_tour/tours/` | 사용자가 저장한 투어 JSON | 아니요 |
| `outputs/virtual_tour/migration-*.json` | 이관 파일 목록과 SHA-256 검증 기록 | 아니요 |

영상은 내용의 SHA-256으로 식별한다. 같은 영상을 다시 입력하면 저장된 원본과 경로를 재사용한다. `server.py --data-dir ... --output-dir ...`로 위치를 바꿀 수 있고 상대 경로는 앱 폴더 기준이다. 저장소 루트의 `data/README.md`, `outputs/README.md`만 Git에 포함한다.

## 분석과 표시

1. 스티칭된 2:1 equirectangular 영상에서 약 2fps로 프레임을 선택한다.
2. CPU PyCOLMAP의 EQUIRECTANGULAR 카메라 모델, SIFT 특징점과 순차 매칭을 사용해 촬영 위치·방향을 복원한다.
3. 관측한 3D 특징점에서 RANSAC으로 바닥 평면을 추정한다.
4. 촬영 경로를 바닥에 투영해 현재 시야에 맞춰 표시한다. 파란색은 현재 시각 이후, 주황색은 이전 구간이다.

영상 위 표시 범위는 앞뒤 각각 6·12·24·60초로 선택한다. 새 영상의 기본값은 12초이며, 변경한 값은 영상별로 저장한다. 미니맵은 복원된 전체 경로를 표시한다. 측면 경로 지점 목록과 자동 지점 생성·장면 추가 기능은 제거된 상태다.

연결되지 않은 구간은 비워 둔다. 바닥 추정이 불충분하면 미니맵만 표시한다. 벽·물체 가림, 계단·다층 바닥, 절대 거리 보정은 지원하지 않는다. 화면의 연결 비율은 시간 범위의 비율이며 위치 정확도가 아니다. 녹화된 카메라 경로를 따라 시간을 이동하는 방식이며, 촬영하지 않은 길을 생성하지 않는다.

## 실행과 보관

저장소 루트에서 `./apps/autotour360/start.sh`를 실행하고 `http://127.0.0.1:8766/`에 접속한다. 서버가 실행 중이어야 접속할 수 있다. Windows에서는 앱 폴더의 `start.bat`을 사용한다. 새 PC의 최초 설치는 앱 README를 따른다.

영상별 경로 표시 설정과 편집 내용은 브라우저 `localStorage`의 `autotour360.project.v1`에도 저장된다. 같은 접속 주소와 브라우저에서는 기존 설정이 이어진다. 다른 PC·브라우저로 이관할 때는 투어 JSON과 입력 영상을 함께 보관한다. 서버는 `/media/<sha256>`와 투어 다운로드 URL을 통해 등록된 파일을 제공하며 저장 폴더 전체를 정적 공개하지 않는다.

## 2026-10-10 이관

기존 독립 앱의 소스를 이 디렉터리로 통합하고, 입력과 결과 저장 경로를 분리했다. 이 PC의 영상 2개(`normal0.MP4`, `chaos.MP4`)를 포함한 250개 파일을 `migrate_legacy.py`로 복사하고 SHA-256으로 검증했다. 파일별 결과는 Git에서 제외된 migration JSON에 기록했다. 원본 독립 앱 폴더는 복구용으로 보존한다.

새 경로에서 앱 Python 테스트 13개, JavaScript 테스트 15개, 프로젝트 공통 규약 테스트 19개와 task fixture 7개가 통과했다. 12초짜리 실제 영상 구간의 별도 분석은 23개 촬영 위치를 복원했으며 약 7.4초가 걸렸다. 기존 영상 2개의 스트리밍·경로 로딩, 영상별 60초 범위 설정 유지, 새 tours 폴더에 대한 저장·다운로드를 확인했다. 실제 영상 검사는 이 PC의 Linux 환경에서 수행했다. 검증 요약은 `outputs/virtual_tour/integration-verification-20261010.json`에 보관한다.

새로 영상을 분석하는 검사는 앱의 별도 환경에서 수행한다. 기존 task의 fixture 검증 통과와 가상투어의 실제 영상 분석 검증은 서로 다른 검사다.
