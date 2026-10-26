"""Compatibility entry point; implementation lives in task 5a under src/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spacewatch3d.tasks.task_5a_change_detection.surface_review import *  # noqa: F403

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for key in ['before', 'after', 'notes', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    main(args.before, args.after, args.notes, args.output)
