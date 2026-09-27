"""Task 5b: 객체 대표 키프레임 선별. Implement the backend here."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a keyframes manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    raise BackendNotImplemented(
        'Task 5b (객체 대표 키프레임 선별) backend is not implemented. '
        'Implement task_5b_keyframes/pipeline.py; '
        'use `spacewatch3d check-fixtures --task 5b` to check the contract.'
    )
