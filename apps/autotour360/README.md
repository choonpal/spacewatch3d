# AUTO TOUR 360

360도 영상을 선택하면 **경로 자동 분석 → 경로를 클릭해 이동**으로 이어지는 SpaceWatch3D의 독립 웹 앱입니다. 분석 task와 별도의 Python 환경으로 실행하며, 프로젝트의 `data/virtual_tour/`와 `outputs/virtual_tour/`를 사용합니다.

설계·범위는 [가상투어 문서](../../docs/virtual-tour.md)를 참고하세요.

## 실행

이 PC에는 앱 전용 분석 환경이 설치되어 있습니다. 아래 폴더에서 실행합니다.

```bash
cd "/home/kdj/Desktop/capstone/spacewatch3d/apps/autotour360"
./start.sh
```

브라우저에서 **http://127.0.0.1:8766/** 를 엽니다. Windows에서는 `start.bat`을 실행합니다. 서버 종료는 실행 터미널에서 `Ctrl+C`입니다.

다른 PC에서 처음 실행할 때는 Python 3.10 이상, FFmpeg(`ffmpeg`와 `ffprobe`를 PATH에 등록)를 준비하고 `setup-trajectory.sh`를 한 번 실행하세요. Windows에서는 `setup-trajectory.bat`입니다. 이후 `start.sh` / `start.bat`으로 실행합니다. 최초 패키지 설치에는 인터넷이 필요하며, 설치 후 분석과 재생은 외부 서비스 없이 로컬에서 처리합니다.

### FFmpeg 설치 및 확인

사용하는 운영체제에 해당하는 명령만 실행하세요. FFmpeg 패키지에는 영상 정보를 읽는 `ffprobe`도 포함됩니다. **두 명령의 버전 정보가 모두 출력되면** 준비된 상태이며, 이미 정상 실행된다면 다시 설치할 필요가 없습니다.

#### Ubuntu / Debian

터미널에서 설치하고 실행 경로와 버전을 확인합니다. 패키지 관리자로 설치하면 일반적으로 PATH를 직접 수정할 필요가 없습니다.

```bash
sudo apt update
sudo apt install -y ffmpeg

command -v ffmpeg
command -v ffprobe
ffmpeg -version
ffprobe -version
```

일반적인 실행 경로는 `/usr/bin/ffmpeg`, `/usr/bin/ffprobe`입니다. 배포판별 패키지는 [FFmpeg 공식 다운로드 안내](https://ffmpeg.org/download.html)에서도 확인할 수 있습니다.

#### Windows 10 / 11 — PowerShell

WinGet으로 [Gyan의 FFmpeg Windows 패키지](https://www.gyan.dev/ffmpeg/builds/)를 설치합니다. `--id`, `--exact`, `--source` 옵션은 설치할 패키지를 지정합니다. [Microsoft 설치 명령 안내](https://learn.microsoft.com/en-us/windows/package-manager/winget/install)

```powershell
winget --version
winget install --id Gyan.FFmpeg --exact --source winget
```

설치가 끝나면 **Windows Terminal / PowerShell 창을 완전히 닫고 새로 연 뒤** 확인합니다. 편집기 안의 터미널이라면 편집기도 다시 실행하세요.

```powershell
where.exe ffmpeg
where.exe ffprobe
ffmpeg -version
ffprobe -version
```

WinGet이 인식되지 않으면 [Microsoft의 WinGet 설치 안내](https://learn.microsoft.com/en-us/windows/package-manager/winget/#install-winget)에 따라 **앱 설치 관리자(App Installer)**를 설치하거나 업데이트한 뒤 다시 실행하세요. FFmpeg ZIP을 직접 내려받아 압축을 푼 경우에는 `ffmpeg.exe`와 `ffprobe.exe`가 들어 있는 `bin` 폴더를 사용자 환경 변수 `Path`에 추가하고 터미널을 다시 열어야 합니다.

#### macOS — Homebrew

Homebrew가 설치된 터미널에서 [공식 FFmpeg Formula](https://formulae.brew.sh/formula/ffmpeg)의 설치 명령을 실행합니다.

```bash
brew install ffmpeg

command -v ffmpeg
command -v ffprobe
ffmpeg -version
ffprobe -version
```

`brew`가 인식되지 않으면 [Homebrew 설치 안내](https://brew.sh/)에 따라 설치하고, 설치 마지막에 표시되는 `brew shellenv` 설정을 적용한 뒤 새 터미널에서 실행하세요.

### FFmpeg 준비 후 최초 설정과 실행

Python 3.10 이상이 준비된 상태에서 진행합니다. 아래 앱 폴더 경로는 각 PC에 프로젝트를 저장한 실제 경로로 바꾸세요.

**Ubuntu / Debian / macOS:**

```bash
cd "/path/to/spacewatch3d/apps/autotour360"
python3 --version
./setup-trajectory.sh
./start.sh
```

Ubuntu / Debian에서 가상환경 생성 중 `venv` 또는 `ensurepip` 관련 오류가 나면 `sudo apt install -y python3-venv`로 기본 Python의 가상환경 패키지를 설치한 뒤 `./setup-trajectory.sh`를 다시 실행하세요.

**Windows PowerShell:**

```powershell
Set-Location "C:\path\to\spacewatch3d\apps\autotour360"
py -3 --version
.\setup-trajectory.bat
.\start.bat
```

설정 스크립트가 오류 없이 완료된 뒤 시작 스크립트를 실행하세요. 이후에는 `start.sh` / `start.bat`만 실행하면 됩니다. FFmpeg를 설치하거나 PATH를 바꾼 동안 뷰어 서버가 실행 중이었다면, 기존 서버를 `Ctrl+C`로 종료하고 새 터미널에서 다시 시작하세요.

## 사용 흐름

1. 시작 화면의 **360° 영상 선택**을 누르거나 영상을 끌어다 놓습니다. 여러 영상도 선택할 수 있습니다.
2. 영상이 PC에 보관되면 **분석이 자동으로 시작**됩니다. 별도 분석 시작 버튼을 누를 필요가 없습니다. 동시에 입력한 영상들은 순서대로 분석합니다.
3. 영상이 바로 열리며, 상단에서 프레임 추출·촬영 위치 복원 등의 진행 상태를 확인할 수 있습니다. 분석 취소·재시도가 가능합니다.
4. 완료되면 영상 위와 미니맵에 촬영 경로가 표시됩니다. 측면에 투어 지점을 생성·저장하는 기능은 제공하지 않습니다.
5. 영상 위 또는 미니맵의 **경로를 클릭·탭하면 해당 촬영 시각으로 이동**합니다. 파란색은 이후 경로, 주황색은 지나온 경로입니다. 재생 중에는 재생을 이어가며, 정지 중에는 정지 상태를 유지합니다.
6. 드래그·방향키로 시선을 바꾸고 휠·두 손가락으로 확대합니다. 미니맵에 초점을 두면 방향키로 5초씩 이동합니다.
7. **투어 편집**에서 장면 이름·설명을 바꾸고 설명 지점 또는 사용자 지정 연결을 추가할 수 있습니다. 이전/다음 지점 버튼은 자동 생성하지 않습니다.
8. **투어 저장**으로 경로·장면·설정이 담긴 JSON을 내보냅니다. 브라우저 다운로드와 함께 저장소의 `outputs/virtual_tour/tours/` 폴더에도 파일이 남습니다. **내 영상**에서 보관한 영상을 다시 열 수 있습니다.

**촬영 경로** 버튼에서는 표시 여부, 미니맵, 앞뒤 표시 범위, 바닥 높이를 조절하고 다시 분석할 수 있습니다. 바닥을 충분히 추정하지 못한 경우 미니맵만 표시합니다.

## 파일과 저장

- 입력 영상과 원본 정보: `data/virtual_tour/<영상 SHA-256>/input.mp4`, `source.json`
- 경로 JSON, 분석 상태, 특징점 데이터베이스, 로그: `outputs/virtual_tour/analysis/<영상 SHA-256>/`
- 같은 내용의 영상을 다시 입력하면 같은 원본과 기존 분석 결과를 재사용합니다.
- 서버를 껐다 켜도 영상 목록과 분석 결과를 다시 불러옵니다. 브라우저를 닫아도 실행 중인 서버는 분석을 계속합니다.
- 장면 편집·설명·표시 설정은 브라우저 `localStorage`의 `autotour360.project.v1`에 저장합니다. 다른 브라우저나 PC로 옮기려면 JSON과 원본 영상을 함께 보관하세요. 영상 자체는 JSON에 포함되지 않습니다.
- 브라우저 보관 데이터를 삭제해도 서버의 영상·경로는 남지만, 사용자 편집 내용은 내보낸 JSON이 있어야 복원할 수 있습니다.
- 캐시는 파일시스템에만 저장되며 외부 서비스에 전송하지 않습니다. 다른 기기에서 접속해 입력하면 프로그램이 실행 중인 PC로 전송됩니다.
- `outputs/virtual_tour/tours/`에는 직접 저장한 투어 JSON 파일이 보관됩니다. 브라우저에서 다운로드가 제한되어도 이 파일로 복원할 수 있습니다.
- 위 데이터·결과 경로는 저장소 루트 기준입니다. `data/`, `outputs/`, 앱의 `.venv/`, `.preview/`는 Git에서 제외됩니다.

## 입력과 분석 범위

| 항목 | 지원 |
| --- | --- |
| 입력 | 스티칭이 완료된 단안 2:1 equirectangular 영상 |
| 권장 코덱 | MP4/H.264. MOV·WebM 등은 브라우저와 FFmpeg의 코덱 지원에 따름 |
| 한도 | 파일당 최대 2GB, 3초 이상·10분 이하 |
| 분석 | 약 2fps 추출, CPU PyCOLMAP 구면 SfM, 관측 특징점에서 바닥 평면 추정 |
| 조작 | PC 마우스·키보드, 모바일 터치, 전체 화면 |
| 경로 이동 | 영상 위 경로·미니맵 클릭으로 해당 시각에 이동. 분석 실패 구간은 연결하지 않음 |

경로는 녹화된 영상에서 추정한 **카메라의 촬영 동선**입니다. 촬영하지 않은 미래 길을 예측하거나 공간 안에서 자유롭게 걸어 다니는 3D 메시 뷰어는 아닙니다. 단안 영상의 거리 단위는 상대값이며, 실제 측량값과 비교한 위치 정확도는 제공하지 않습니다. 화면의 ‘촬영 구간 연결 비율’은 분석된 시간 구간의 비율입니다.

흔들림, 반복 무늬, 특징이 적은 벽, 움직이는 사람, 조명 변화 등으로 분석이 실패하거나 경로가 틀어질 수 있습니다. 벽·물체 뒤 경로의 가림 처리, 계단·다층 바닥, 듀얼 어안 스티칭, 입체 360 영상, 실시간 SLAM은 지원하지 않습니다. 자동 분석에 실패해도 영상을 둘러보고 재생 막대로 이동할 수 있습니다.

## 다른 실행 옵션

원본을 복사하지 않고 지정 경로에서 직접 분석·재생할 수도 있습니다.

```bash
python3 server.py --media "/path/to/video.mp4" --open
```

이 경우 다음 실행에도 같은 `--media` 인수를 사용해야 합니다. 파일 선택으로 입력한 영상은 인수 없이 복원됩니다.

같은 Wi-Fi의 휴대전화에서 열려면 `python3 server.py --host 0.0.0.0`으로 실행하고 `http://PC의-로컬-IP:8766/`로 접속합니다. 기본은 이 PC에서만 접속 가능한 `127.0.0.1`입니다.

입력·출력 폴더는 옵션으로 지정할 수 있습니다. 상대 경로는 실행한 터미널 위치와 관계없이 **앱 폴더 기준**입니다.

```bash
python3 server.py --data-dir "/path/to/videos" --output-dir "/path/to/results" --open
```

`--output-dir` 아래에는 `analysis/`와 `tours/`가 생성됩니다. 서버 시작 메시지에 실제 저장 위치가 표시됩니다. 기본 데이터 경로를 사용하려면 저장소의 `apps/autotour360/` 위치를 유지하세요.

## 이전 독립 앱의 데이터 가져오기

기존 앱에서 실행 중인 분석을 완료하거나 취소하고, 브라우저에서 **투어 저장**을 누릅니다. 이후 아래 명령으로 영상·경로·투어 파일을 복사합니다. 이미 다른 내용의 파일이 있는 대상은 덮어쓰지 않으며, 모든 복사 파일의 SHA-256을 검증합니다. 원본 폴더는 유지됩니다.

```bash
python3 migrate_legacy.py "/path/to/old/autotour360"
```

이관 기록은 `outputs/virtual_tour/migration-*.json`에 남습니다. 같은 호스트·포트·브라우저로 접속하면 기존 브라우저 설정을 계속 사용합니다. 다른 주소나 브라우저로 열 때는 저장한 투어 JSON을 불러오세요.

## 구성 및 검사

- `app.js`, `index.html`, `styles.css`, `autotour.css`: 시작 화면, 영상 보관함, 투어 편집·재생
- `autotour.js`: 입력 영상 검증, 진행률을 포함한 파일 전달
- `panorama.js`: WebGL 360도 영상 렌더링
- `trajectory.js`, `trajectory-math.js`: 경로 투영·미니맵·클릭으로 시간 이동
- `server.py`, `trajectory_jobs.py`: 로컬 영상 보관·스트리밍·자동 분석 작업 API
- `estimate_trajectory.py`: 실제 영상에서 카메라 위치와 바닥 추정
- `storage.py`: 코드·입력·출력 경로 분리
- `migrate_legacy.py`: 기존 독립 앱의 영상·분석 결과·투어 JSON 이관

```bash
node --test tests/*.test.mjs
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

외부 JavaScript 라이브러리나 CDN 없이 실행됩니다. 분석 의존성은 `requirements-trajectory.txt`에 고정되어 있습니다.
