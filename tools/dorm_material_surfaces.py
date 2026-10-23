"""Compatibility entry point; implementation lives in task 5a under src/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spacewatch3d.tasks.task_5a_change_detection.surface_materials import *  # noqa: F403
