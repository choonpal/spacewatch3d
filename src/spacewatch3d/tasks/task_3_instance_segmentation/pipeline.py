"""Task 3: 3D 인스턴스 분할. Implement the backend here."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a instances manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    raise BackendNotImplemented(
        'Task 3 (3D 인스턴스 분할) backend is not implemented. '
        'Implement task_3_instance_segmentation/pipeline.py; '
        'use `spacewatch3d check-fixtures --task 3` to check the contract.'
    )
