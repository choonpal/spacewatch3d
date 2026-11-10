"""Resolve app code, uploaded videos, and generated outputs independently."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoragePaths:
    data: Path
    output: Path

    @classmethod
    def for_app(cls, app_root, data_dir=None, output_dir=None):
        app_root = Path(app_root).resolve()
        project_root = app_root.parent.parent if app_root.parent.name == "apps" else app_root

        def resolve(value, default):
            path = Path(value).expanduser() if value is not None else default
            return (path if path.is_absolute() else app_root / path).resolve()

        return cls(resolve(data_dir, project_root / "data" / "virtual_tour"),
                   resolve(output_dir, project_root / "outputs" / "virtual_tour"))

    @property
    def analysis(self):
        return self.output / "analysis"

    @property
    def tours(self):
        return self.output / "tours"
