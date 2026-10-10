@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements-trajectory.txt
if errorlevel 1 exit /b 1
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg is required. Install FFmpeg and add it to PATH.
  exit /b 1
)
where ffprobe >nul 2>nul
if errorlevel 1 (
  echo FFprobe is required. Install the complete FFmpeg package and add it to PATH.
  exit /b 1
)
echo Restart the viewer server after setup.
pause
