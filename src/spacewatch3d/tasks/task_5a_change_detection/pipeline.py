"""Task 5a: 공간 정합 및 변화탐지. Implement the backend here."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a changes manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    raise BackendNotImplemented(
        'Task 5a (공간 정합 및 변화탐지) backend is not implemented. '
        'Implement task_5a_change_detection/pipeline.py; '
        'use `spacewatch3d check-fixtures --task 5a` to check the contract.'
    )
