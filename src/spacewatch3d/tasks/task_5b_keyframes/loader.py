"""Read explicit metric Z-up scenes and adapt Task4/3 manifests read-only."""

from pathlib import Path
import hashlib
import numpy as np
import shapely
from shapely.geometry import Polygon

from .models import Config, Geometry, InputError, Scene, read_json


COVERAGE_ONLY_CLASSES = {"wall", "walls", "ceiling", "ceilings", "벽", "천장"}


def matrix(value, name="transform"):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(f"Invalid {name}") from exc
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise InputError(f"{name} must be a finite 4x4 matrix")
    linear = result[:3, :3]
    magnitude = float(np.max(np.abs(linear)))
    if not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-8) or magnitude == 0:
        raise InputError(f"{name} must be an invertible affine transform")
    # Determinant magnitude depends on the source unit cubed. Validate rank
    # after normalization so a small, well-conditioned unit conversion is valid
    # while a large but almost singular affine transform is not.
    singular = np.linalg.svd(linear / magnitude, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-12:
        raise InputError(f"{name} must be a numerically stable invertible affine transform")
    return result


def polygon(data):
    try:
        exterior = np.asarray(data["polygon"], dtype=float)
        holes = [np.asarray(ring, dtype=float) for ring in data.get("holes", [])]
        if exterior.ndim != 2 or exterior.shape[1] != 2 or not np.isfinite(exterior).all():
            raise ValueError("Polygon coordinates must be finite XY")
        if any(ring.ndim != 2 or ring.shape[1] != 2 or not np.isfinite(ring).all() for ring in holes):
            raise ValueError("Polygon holes must be finite XY")
        result = Polygon(data["polygon"], data.get("holes", []))
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError("Invalid floor/region polygon") from exc
    if not result.is_valid or result.is_empty or result.area <= 1e-10:
        raise InputError("Polygon must be nonempty, simple and have positive area")
    if not np.isfinite(result.bounds).all():
        raise InputError("Non-finite polygon")
    return result


def box_triangles(bounds):
    import trimesh
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise InputError("box must be [min_xyz, max_xyz] with positive extents")
    mesh = trimesh.creation.box(extents=bounds[1] - bounds[0])
    mesh.apply_translation(bounds.mean(axis=0))
    return mesh.triangles.copy()


def load_triangles(item, base: Path, config: Config):
    import trimesh
    sources = [name for name in ("mesh_uri", "box", "triangles") if name in item]
    if len(sources) != 1:
        raise InputError("Each geometry needs exactly one mesh_uri, box or triangles")
    if "mesh_uri" in item:
        uri = item["mesh_uri"]
        if not isinstance(uri, str) or "://" in uri:
            raise InputError("mesh_uri must reference a local OBJ or GLB file")
        path = (base / uri).resolve()
        if not path.is_file() or path.suffix.lower() not in (".obj", ".glb"):
            raise InputError(f"Missing or unsupported mesh: {path}")
        try:
            loaded = trimesh.load_scene(path, process=False)
            # Bake GLB node transforms once; then apply world_from_object below.
            triangles = np.asarray(loaded.to_mesh().triangles)
        except Exception as exc:
            raise InputError(f"Cannot load mesh {path}: {exc}") from exc
    elif "box" in item:
        triangles = box_triangles(item["box"])
    else:
        triangles = np.asarray(item["triangles"], dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise InputError("Mesh must contain triangular faces")
    if len(triangles) > config.max_triangles or not np.isfinite(triangles).all():
        raise InputError("Non-finite mesh or max_triangles exceeded")
    transform = matrix(item.get("world_from_object", np.eye(4)))
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = triangles @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(transformed).all():
        raise InputError("Non-finite transformed mesh")
    return transformed


def load_scene(path: Path, config: Config) -> Scene:
    path = Path(path).resolve()
    data = read_json(path)
    if data.get("schema_name") != "spacewatch3d.planning_scene" or data.get("schema_version") != "0.1.0":
        raise InputError("Expected spacewatch3d.planning_scene version 0.1.0")
    for key in ("scene_id", "capture_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise InputError(f"Missing {key}")
    if not isinstance(data.get("synthetic"), bool):
        raise InputError("synthetic must be explicit boolean")
    coords = data.get("coordinates", {})
    if (coords.get("units") != "m" or coords.get("up_axis") != "+Z"
            or coords.get("gravity_aligned") is not True
            or not isinstance(coords.get("frame_id"), str) or not coords["frame_id"]):
        raise InputError("Planning coordinates require known meters, +Z and gravity_aligned=true")
    if data.get("geometry_complete") is not True:
        raise InputError("geometry_complete=true must explicitly attest structural/occluder input")
    if data.get("reference_route") is not None or data.get("registration") is not None:
        raise InputError("Inline reference_route/registration is unsupported; use --reference-route and --registration")
    for field in ("objects", "structures", "regions"):
        if not isinstance(data.get(field, []), list) or any(not isinstance(item, dict) for item in data.get(field, [])):
            raise InputError(f"{field} must be an array of objects")
    floor_data = data.get("floor")
    if not isinstance(floor_data, dict) or floor_data.get("confirmed") is not True:
        raise InputError("A confirmed single-level floor is required")
    floor = polygon(floor_data)
    height = floor_data.get("height_m")
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height):
        raise InputError("floor.height_m must be finite")
    if data.get("levels", 1) != 1:
        raise InputError("Multiple levels and stairs are unsupported")
    provenance = {"scene_manifest": str(path), "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    camera = data.get("camera", {})
    if not isinstance(camera, dict):
        raise InputError("camera must be an object")
    captured_height = camera.get("height_above_floor_m")
    if "height_above_floor_m" in camera and (isinstance(captured_height, bool)
            or not isinstance(captured_height, (int, float)) or not np.isfinite(captured_height)
            or captured_height <= 0):
        raise InputError("camera.height_above_floor_m must be finite and positive")
    if config.camera_height_m is not None:
        camera_height, height_source = config.camera_height_m, "config"
    elif captured_height is not None:
        camera_height, height_source = captured_height, "scene.camera.height_above_floor_m"
    else:
        camera_height, height_source = 1.5, "fallback_unverified"
    if not config.body_min_m < camera_height <= config.body_max_m:
        raise InputError("Body height interval must include the resolved camera height; adjust body_max_m if needed")
    provenance["camera_height"] = {"height_above_floor_m": float(camera_height), "source": height_source}
    items = [(item, path.parent) for item in data.get("objects", [])]
    if data.get("objects_manifest"):
        object_path = (path.parent / data["objects_manifest"]).resolve()
        exported = read_json(object_path)
        instance_path = object_path.parent / exported["instances_manifest"]
        instances = read_json(instance_path)
        reconstruction = read_json(instance_path.parent / instances["reconstruction_manifest"])
        for manifest, kind in ((exported, "objects"), (instances, "instances"), (reconstruction, "reconstruction")):
            if manifest.get("kind") != kind:
                raise InputError(f"Expected {kind} manifest")
            for key in ("scene_id", "capture_id", "synthetic"):
                if manifest.get(key) != data[key]:
                    raise InputError(f"Upstream {key} mismatch")
        source_coords = reconstruction["coordinates"]
        aligned = (source_coords.get("frame_id") == coords["frame_id"] and source_coords.get("units") == "m" and source_coords.get("up_axis") == "+Z")
        if "planning_from_world" in data:
            planning_from_world = matrix(data["planning_from_world"], "planning_from_world")
            linear = planning_from_world[:3, :3]
            magnitude = float(np.max(np.abs(linear)))
            normalized = linear / magnitude
            column_norm = np.linalg.norm(normalized[:, 0])
            rotation = normalized / column_norm
            with np.errstate(over="ignore"):
                scale = magnitude * column_norm
            if (not np.isfinite(scale) or scale <= 0 or np.linalg.det(rotation) <= 0
                    or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)):
                raise InputError("planning_from_world must be a proper uniform-scale similarity")
            known_scale = source_coords.get("scale_to_meters")
            if known_scale is not None:
                if (isinstance(known_scale, bool) or not isinstance(known_scale, (int, float))
                        or not np.isfinite(known_scale) or known_scale <= 0):
                    raise InputError("source scale_to_meters must be finite and positive")
                if not np.isclose(scale, known_scale, rtol=1e-8, atol=0):
                    raise InputError("planning_from_world scale disagrees with source scale_to_meters")
            if known_scale is None and data.get("source_scale_confirmed") is not True:
                raise InputError("Unknown source scale requires source_scale_confirmed=true and measured planning_from_world")
            provenance["planning_from_world"] = planning_from_world.tolist()
        elif aligned:
            planning_from_world = np.eye(4)
        else:
            raise InputError("Upstream coordinates differ; provide measured planning_from_world into metric +Z frame")
        names = {obj["instance_id"]: obj for obj in instances["instances"]}
        if len(names) != len(instances["instances"]):
            raise InputError("Duplicate upstream instance_id")
        for obj in exported["objects"]:
            if obj["instance_id"] not in names:
                raise InputError("Object has no matching instance metadata")
            combined = planning_from_world @ matrix(obj["world_from_object"], "world_from_object")
            items.append(({**names[obj["instance_id"]], **obj, "world_from_object": combined.tolist()}, object_path.parent))
        provenance["objects_manifest"] = str(object_path)
        provenance["source_cameras"] = len(reconstruction.get("cameras", []))
    geometries, objects, ids = [], [], set()
    coverage_only_objects = []
    for item, base in items:
        oid = item.get("instance_id")
        if not isinstance(oid, str) or not oid or oid in ids:
            raise InputError("Object instance_id must be unique nonempty string")
        ids.add(oid)
        name = item.get("class_name")
        if name is not None and not isinstance(name, str):
            raise InputError("class_name must be string or null")
        if not isinstance(item.get("must_see", False), bool):
            raise InputError("must_see must be boolean")
        if name is not None and name.strip().casefold() in COVERAGE_ONLY_CLASSES:
            if item.get("must_see", False):
                raise InputError("Walls and ceilings are coverage-only; must_see applies to semantic objects")
            solid = item.get("solid", "box" in item)
            if not isinstance(solid, bool):
                raise InputError("structure.solid must be boolean")
            geometries.append(Geometry(f"structure_object:{oid}", load_triangles(item, base, config), None, solid=solid))
            coverage_only_objects.append({"instance_id": oid, "class_name": name})
            continue
        objects.append({"instance_id": oid, "class_name": name,
                        "must_see": item.get("must_see", False), "importance": item.get("importance"),
                        "description": item.get("description", "")})
        geometries.append(Geometry(f"object:{oid}", load_triangles(item, base, config), oid, solid=True))
    for index, item in enumerate(data.get("structures", [])):
        if not isinstance(item.get("solid", "box" in item), bool):
            raise InputError("structure.solid must be boolean")
        geometries.append(Geometry(f"structure:{index}", load_triangles(item, path.parent, config), None,
                                   solid=item.get("solid", "box" in item)))
    overrides = data.get("object_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {obj["instance_id"] for obj in objects}:
        raise InputError("object_overrides must reference semantic object IDs; walls/ceilings are coverage-only")
    for obj in objects:
        override = overrides.get(obj["instance_id"], {})
        if not isinstance(override, dict) or set(override) - {"importance", "must_see", "description"}:
            raise InputError("Only importance, must_see and description can be overridden")
        if "must_see" in override and not isinstance(override["must_see"], bool):
            raise InputError("must_see override must be boolean")
        obj.update(override)
    floor_triangles = []
    for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(floor)):
        xy = np.asarray(triangle.exterior.coords)[:3, :2]
        floor_triangles.append(np.column_stack([xy, np.full(3, height)]))
    geometries.append(Geometry("floor", np.asarray(floor_triangles), None, obstacle=False))
    # One physical face must not multiply coverage when referenced twice.
    seen = {}
    for geom in geometries:
        unique = []
        for triangle in geom.triangles:
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 1e-12:
                continue
            key = tuple(sorted(tuple(np.round(vertex, 9)) for vertex in triangle))
            if key in seen:
                if seen[key] != geom.object_id and seen[key] is not None and geom.object_id is not None:
                    raise InputError("Coincident face belongs to different objects")
                continue
            seen[key] = geom.object_id
            unique.append(triangle)
        geom.triangles = np.asarray(unique).reshape((-1, 3, 3))
    if len(seen) > config.max_triangles:
        raise InputError("Total max_triangles exceeded")
    geometry_hash = hashlib.sha256()
    for geom in geometries:
        geometry_hash.update(geom.geometry_id.encode())
        geometry_hash.update(geom.triangles.astype("<f8").tobytes())
    provenance["geometry_sha256"] = geometry_hash.hexdigest()
    provenance["coverage_only_objects"] = coverage_only_objects
    regions = []
    for item in data.get("regions", []):
        region = polygon(item)
        rid = item.get("region_id")
        if not isinstance(rid, str) or not rid or any(rid == old for old, _ in regions):
            raise InputError("region_id must be unique nonempty string")
        if any(region.intersection(old).area > 1e-8 for _, old in regions):
            raise InputError("Regions must not overlap")
        regions.append((rid, region))
    start = data.get("start_xy_m")
    if start is not None and (np.asarray(start).shape != (2,) or not np.isfinite(start).all()):
        raise InputError("start_xy_m must be finite 2-vector")
    return Scene(data["scene_id"], data["capture_id"], data["synthetic"], coords["frame_id"],
                 floor, float(height), geometries, objects, regions, start, provenance, float(camera_height))
