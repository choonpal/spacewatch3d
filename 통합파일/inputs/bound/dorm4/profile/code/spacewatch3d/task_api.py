"""The small shared API every task implements."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskRequest:
    inputs: dict[str, Path]
    output_dir: Path
    config: dict[str, Any] = field(default_factory=dict)


class BackendNotImplemented(NotImplementedError):
    """A task scaffold exists, but its model/processing backend is not wired yet."""
