#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$task_root/run_env.sh" "$task_root/pipeline.py" "$@"
