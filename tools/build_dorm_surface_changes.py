"""Compatibility entry point; implementation lives in task 5a under src/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spacewatch3d.tasks.task_5a_change_detection.dorm_surface import *  # noqa: F403

if __name__ == '__main__':
    from spacewatch3d.tasks.task_5a_change_detection.__main__ import main as cli_main
    raise SystemExit(cli_main())
