#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
config_path="${1:-configs/dorm2_dorm3.json}"
output_path="${2:-/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027}"
start_mode=$(python3 - "$config_path" "$output_path" <<'PY'
import json, sys
from pathlib import Path
config_path, output_path = map(Path, sys.argv[1:])
state = output_path / 'status.json'
if state.exists() and json.loads(state.read_text()).get('status') == 'complete':
    if json.loads(config_path.read_text()) != json.loads((output_path / 'config.json').read_text()):
        raise SystemExit('Completed output uses different settings. Choose a new output directory.')
    print('viewer')
else:
    print('pipeline')
PY
)
if [[ "$start_mode" == viewer ]]; then
  # Attach the new UI to receipted results without rewriting the completed run.
  exec ./run.sh viewer --output "$output_path"
fi
./run.sh doctor --config "$config_path"
exec ./run.sh start --config "$config_path" --output "$output_path"
