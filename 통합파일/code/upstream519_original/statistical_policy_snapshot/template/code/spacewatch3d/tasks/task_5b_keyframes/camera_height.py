"""Initialize a constant viewing height from existing metric capture positions.

This estimates the capture height, not an optimal or image-validated viewpoint.
The supplied single-level floor and coordinate normalization remain prerequisites.
"""

import hashlib
from pathlib import Path

import numpy as np
import shapely

from .models import InputError, read_json


def _positions(value):
    if (not isinstance(value, list)
            or any(not isinstance(row, list) or len(row) != 3
                   or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not np.isfinite(v) for v in row) for row in value)):
        raise InputError("camera positions_m must be an array of finite XYZ positions in the planning frame")
    return np.asarray(value, dtype=float).reshape((-1, 3))


def resolve_camera_height(camera, config, floor, floor_height, scene_path, frame_id,
                          warnings, upstream=None):
    """Return effective height evidence; a reference may supersede it afterward."""
    if not isinstance(camera, dict):
        raise InputError("camera must be an object")
    measured = camera.get("height_above_floor_m")
    if "height_above_floor_m" in camera and (isinstance(measured, bool)
            or not isinstance(measured, (int, float)) or not np.isfinite(measured) or measured <= 0):
        raise InputError("camera.height_above_floor_m must be finite and positive")
    if config.camera_height_m is not None:
        return {"height_above_floor_m": float(config.camera_height_m), "source": "config"}
    if measured is not None:
        return {"height_above_floor_m": float(measured), "source": "scene.camera.height_above_floor_m"}

    evidence, invalid = {}, 0
    if "positions_m" in camera and "trajectory_uri" in camera:
        raise InputError("Supply camera.positions_m or camera.trajectory_uri, not both")
    if "positions_m" in camera:
        positions, source = _positions(camera["positions_m"]), "scene.camera.positions_m"
    elif "trajectory_uri" in camera:
        uri = camera["trajectory_uri"]
        if not isinstance(uri, str) or not uri:
            raise InputError("camera.trajectory_uri must name a local JSON trajectory")
        path = (Path(scene_path).parent / uri).resolve()
        trajectory = read_json(path)
        for key, expected in (("frame_id", frame_id), ("units", "m"), ("up_axis", "+Z")):
            if key in trajectory and trajectory[key] != expected:
                raise InputError("Camera trajectory " + key + " differs from planning coordinates")
        positions, source = _positions(trajectory.get("positions_m")), "scene.camera.trajectory_uri"
        evidence = {"trajectory_manifest": str(path),
                    "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    elif upstream is not None:
        positions, invalid, path = upstream
        source = "upstream.reconstruction.cameras"
        evidence = {"reconstruction_manifest": str(path),
                    "reconstruction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        return {"height_above_floor_m": 1.5, "source": "fallback_unverified"}

    # A path crossing unknown floor or a hole cannot establish height above
    # this floor. Do not invent a floor there or clip estimates to body limits.
    over_floor = shapely.covers(floor, shapely.points(positions[:, :2]))
    heights = positions[:, 2] - floor_height
    usable = over_floor & (heights > 0)
    values = heights[usable]
    evidence.update(source=source, estimator="median_height_above_supplied_floor",
                    input_position_count=len(positions) + invalid, used_position_count=len(values),
                    excluded_position_count=int((~usable).sum()) + invalid,
                    outside_floor_count=int((~over_floor).sum()),
                    nonpositive_height_count=int((over_floor & (heights <= 0)).sum()),
                    invalid_pose_count=invalid, measurement_validated=False,
                    image_quality_validated=False)
    if not len(values):
        warnings.append({"code": "camera_height_estimation_unavailable",
                         "message": "No valid capture positions above the supplied floor; retaining unverified height fallback",
                         "estimation": evidence})
        return {"height_above_floor_m": 1.5, "source": "fallback_unverified",
                "attempted_estimation": evidence}
    height = float(np.median(values))
    evidence.update(height_above_floor_m=height,
                    height_range_m=[float(values.min()), float(values.max())],
                    height_mad_m=float(np.median(np.abs(values - height))))
    warnings.append({"code": "camera_height_estimated",
                     "message": "Capture height estimated from supplied floor and trajectory; use provenance.camera_height for the effective canonical height",
                     "height_above_floor_m": height, "source": source})
    return evidence
