"""Task-local contracts. No changes to the shared objects -> keyframes API."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import json
import math


class InputError(ValueError):
    """Invalid or unsupported planning input (not a successful empty tour)."""


def read_json(path: Path) -> dict:
    def constant(value):
        raise InputError(f"Non-finite JSON value: {value}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read {path}: {exc}") from exc
    def visit(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError("Non-finite JSON number")
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(data)
    if not isinstance(data, dict):
        raise InputError("Expected a JSON object")
    return data


@dataclass(frozen=True)
class Config:
    grid_m: float = 0.15
    clearance_m: float = 0.3
    camera_height_m: float | None = None
    body_min_m: float = 0.08
    body_max_m: float = 1.9
    candidate_spacing_m: float = 0.6
    adaptive_candidates: bool = True
    candidate_min_spacing_m: float = 0.15
    candidate_narrow_clearance_m: float = 0.6
    candidate_visibility_change_threshold: float = 0.15
    route_aware_selection: bool = True
    max_link_m: float = 2.0
    max_view_distance_m: float = 15.0
    tile_m: float = 3.0
    surface_samples: int = 600
    min_object_samples: int = 12
    min_region_samples: int = 12
    space_weight: float = 0.6
    object_weight: float = 0.4
    space_target: float = 0.9
    region_target: float = 0.6
    object_target: float = 0.9
    object_quality_threshold: float = 0.35
    solid_angle_target_sr: float = 0.02
    default_importance: float = 0.5
    seed: int = 0
    max_grid_cells: int = 250000
    max_route_nodes: int = 20000
    max_candidates: int = 800
    max_visibility_probes: int = 2000
    max_triangles: int = 200000
    width: int = 4096
    height: int = 2048

    @classmethod
    def from_dict(cls, values: dict | None = None):
        values = values or {}
        if not isinstance(values, dict):
            raise InputError("Config must be an object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise InputError(f"Unknown config fields: {sorted(unknown)}")
        result = cls(**values)
        defaults = asdict(cls())
        for name, value in asdict(result).items():
            if isinstance(defaults[name], bool):
                if not isinstance(value, bool):
                    raise InputError(f"Config {name} must be boolean")
                continue
            if name == "camera_height_m" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InputError(f"Config {name} must be finite numeric")
            if isinstance(defaults[name], int) and not isinstance(value, int):
                raise InputError(f"Config {name} must be an integer")
            if name != "seed" and value < 0:
                raise InputError(f"Config {name} must be nonnegative")
        positive = ("grid_m", "candidate_spacing_m", "candidate_min_spacing_m", "max_link_m",
                    "max_view_distance_m", "tile_m", "surface_samples", "min_object_samples", "min_region_samples",
                    "solid_angle_target_sr", "max_grid_cells", "max_route_nodes",
                    "max_candidates", "max_visibility_probes", "max_triangles", "width", "height")
        if any(getattr(result, name) <= 0 for name in positive):
            raise InputError("Lengths, sample counts and resource limits must be positive")
        if result.camera_height_m is not None and result.camera_height_m <= 0:
            raise InputError("camera_height_m must be positive or null")
        if result.candidate_min_spacing_m > result.candidate_spacing_m:
            raise InputError("candidate_min_spacing_m must not exceed candidate_spacing_m")
        if not 0 < result.candidate_visibility_change_threshold <= 1:
            raise InputError("candidate_visibility_change_threshold must be in (0,1]")
        for name in ("space_target", "region_target", "object_target", "default_importance",
                     "object_quality_threshold"):
            if not 0 <= getattr(result, name) <= 1:
                raise InputError(f"Config {name} must be in [0, 1]")
        if result.space_weight + result.object_weight <= 0:
            raise InputError("At least one objective weight must be positive")
        if result.object_quality_threshold == 0:
            raise InputError("object_quality_threshold must be positive; zero would mark invisible objects observed")
        if result.seed < 0 or result.body_min_m >= result.body_max_m:
            raise InputError("Invalid seed or body height interval")
        if result.camera_height_m is not None and not result.body_min_m < result.camera_height_m <= result.body_max_m:
            raise InputError("Body interval must include camera height")
        if result.width != 2 * result.height:
            raise InputError("Equirectangular resolution must be 2:1")
        if result.max_link_m < math.sqrt(2) * result.grid_m:
            raise InputError("max_link_m must allow at least one diagonal grid step")
        return result


@dataclass
class Geometry:
    geometry_id: str
    triangles: Any
    object_id: str | None
    obstacle: bool = True
    solid: bool = False


@dataclass
class Scene:
    scene_id: str
    capture_id: str
    synthetic: bool
    reference_frame_id: str
    floor: Any
    floor_height: float
    geometries: list[Geometry]
    objects: list[dict]
    regions: list[tuple[str, Any]]
    start: Any
    provenance: dict
    camera_height_m: float = 1.5
