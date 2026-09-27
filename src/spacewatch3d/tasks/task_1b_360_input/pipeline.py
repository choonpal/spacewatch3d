"""Task 1b: 360° 영상 입력. Implement the backend here."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a frames manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    raise BackendNotImplemented(
        'Task 1b (360° 영상 입력) backend is not implemented. '
        'Implement task_1b_360_input/pipeline.py; '
        'use `spacewatch3d check-fixtures --task 1b` to check the contract.'
    )
