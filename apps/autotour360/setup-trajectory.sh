#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-trajectory.txt
command -v ffmpeg >/dev/null || { echo 'FFmpeg is required. Install it using your OS package manager.'; exit 1; }
command -v ffprobe >/dev/null || { echo 'FFprobe is required. Install the complete FFmpeg package.'; exit 1; }
echo 'Trajectory analysis is ready. Restart the viewer server.'
