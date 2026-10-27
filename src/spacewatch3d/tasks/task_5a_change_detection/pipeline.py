"""Task 5a entry point; optional geometry dependencies load only on execution."""

from pathlib import Path

from spacewatch3d.task_api import BackendNotImplemented, TaskRequest


def run(request: TaskRequest) -> Path:
    """Read named manifests, write a changes manifest, return its path.

    Keep heavy model imports inside this function or a backend module.
    Never modify upstream inputs. Paths in the output manifest are relative
    to that manifest. See README.md in this directory and docs/contracts.md.
    """
    backend = request.config.get('backend', 'dorm_geometry_first')
    if backend == 'mesh_rgb_simsac':
        from .mesh_backend import run as run_mesh
        return run_mesh(request)
    if backend == 'dorm_geometry_first':
        from .dorm_backend import run as run_dorm
        return run_dorm(request)
    raise BackendNotImplemented(
        f'Task 5a backend {backend!r} is not implemented. '
        'Use mesh_rgb_simsac with segmented meshes and original RGB, or dorm_geometry_first with a prepared whole-mesh pair.'
    )
