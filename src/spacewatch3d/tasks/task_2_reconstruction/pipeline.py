"""Task 2: 포인트클라우드 생성. Implement the backend here."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a reconstruction manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    raise BackendNotImplemented(
        'Task 2 (포인트클라우드 생성) backend is not implemented. '
        'Implement task_2_reconstruction/pipeline.py; '
        'use `spacewatch3d check-fixtures --task 2` to check the contract.'
    )
