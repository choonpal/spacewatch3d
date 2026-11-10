"""Read explicit metric Z-up scenes and adapt Task4/3 manifests read-only."""

from pathlib import Path
import hashlib
import numpy as np
import shapely
from shapely.geometry import Polygon

from .models import Config, Geometry, InputError, Scene, read_json
from .camera_height import resolve_camera_height
from .object_identity import group_physical_objects,record_observation_identity


COVERAGE_ONLY_CLASSES = {"wall", "walls", "ceiling", "ceilings", "벽", "천장"}


def geometry_digest(geometries):
    result = hashlib.sha256()
    for geom in geometries:
        result.update(geom.geometry_id.encode())
        result.update(geom.triangles.astype("<f8").tobytes())
        result.update(repr((geom.object_id, geom.object_ids, geom.obstacle, geom.solid)).encode())
        if geom.collision_triangles is not None:
            result.update(geom.collision_triangles.astype("<f8").tobytes())
    return result.hexdigest()


def retain_input_context(data, path, provenance, warnings):
    """Keep producer facts without making optional metadata an input gate."""
    for key in ("input_assumptions", "source_context"):
        if key in data:
            provenance[key] = data[key]
    assumptions = data.get("input_assumptions", {})
    if isinstance(assumptions, dict) and (assumptions.get("height_approximate") is True
            or assumptions.get("floor_estimated") is True or assumptions.get("physical_safety_verified") is False):
        warnings.append({"code": "input_assumptions_unverified", "assumptions": assumptions,
                         "message": "Supplied scale/floor/safety assumptions are retained, not independently verified"})
    context = data.get("source_context", {})
    if isinstance(context, dict) and context.get("unmeshed_source_instance_ids"):
        warnings.append({"code": "source_regions_without_mesh",
                         "instance_ids": context["unmeshed_source_instance_ids"],
                         "message": "Producer reports source regions without mesh; coverage targets concern supplied mesh, not missing regions"})
    if "preprocessing_protocol" not in data:
        return
    declared = data["preprocessing_protocol"]
    record = {"declared": declared, "status": "unavailable"}
    provenance["preprocessing_protocol"] = record
    try:
        if not isinstance(declared, str) or not declared:
            raise InputError("Optional preprocessing_protocol is not a local JSON path")
        manifest = (path.parent / declared).resolve()
        record["manifest"] = str(manifest)
        value = read_json(manifest)
        record.update(status="available", record=value,
                      sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    except (InputError, OSError) as error:
        warnings.append({"code": "preprocessing_protocol_unavailable", "detail": str(error),
                         "message": "Optional preprocessing record unavailable; supplied geometry is still processed"})


def deduplicate_surfaces(geometries, config, warnings):
    """Physical coverage once, all semantic owners, original collision solids.

    This reconciles exact face keys only, not intersecting or retriangulated
    surfaces. Conflicting labels remain aliases rather than guessed identities.
    """
    seen, raw = {}, []
    duplicate_count, metadata_conflicts = 0, 0
    for index, geom in enumerate(geometries):
        nondegenerate = []
        for triangle in geom.triangles:
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 1e-12:
                continue
            nondegenerate.append(triangle)
            key = tuple(sorted(tuple(np.round(vertex, 9)) for vertex in triangle))
            owners = {geom.object_id} if geom.object_id is not None else set()
            if key in seen:
                duplicate_count += 1
                record = seen[key]
                metadata_conflicts += int(record[3] != geom.object_id or record[4] != geom.solid)
                record[2].update(owners)
            else:
                seen[key] = [index, triangle, owners, geom.object_id, geom.solid]
        raw.append(np.asarray(nondegenerate).reshape((-1, 3, 3)))
    if len(seen) > config.max_triangles:
        raise InputError("Total max_triangles exceeded")
    unique, shared = [[] for _ in geometries], {}
    for index, triangle, owners, first_owner, _ in seen.values():
        labels = tuple(sorted(owners))
        if labels == ((first_owner,) if first_owner is not None else ()):
            unique[index].append(triangle)
        else:
            shared.setdefault(labels, []).append(triangle)
    for geom, original, faces in zip(geometries, raw, unique):
        geom.triangles = np.asarray(faces).reshape((-1, 3, 3))
        if not np.array_equal(original, geom.triangles):
            geom.collision_triangles = original
    for labels, faces in sorted(shared.items()):
        geometries.append(Geometry("shared_surface:" + repr(labels), np.asarray(faces), labels[0],
                                   obstacle=False, object_ids=labels))
    if metadata_conflicts:
        warnings.append({"code": "shared_face_metadata", "message":
            "Shared faces accepted: physical coverage counted once, semantic owners retained; collision uses the union of original inputs",
            "conflicting_duplicate_count": metadata_conflicts, "duplicate_count": duplicate_count})


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


def load_scene(path: Path, config: Config, *, defer_camera_height_validation=False) -> Scene:
    path = Path(path).resolve()
    data = read_json(path)
    if data.get("schema_name") != "spacewatch3d.planning_scene" or data.get("schema_version") != "0.1.0":
        raise InputError("Expected spacewatch3d.planning_scene version 0.1.0")
    for key in ("scene_id", "capture_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise InputError(f"Missing {key}")
    warnings = []
    if "synthetic" not in data:
        warnings.append({"code": "synthetic_unspecified", "message": "synthetic omitted; false is an unverified default, not evidence of real-data validation"})
    elif not isinstance(data["synthetic"], bool):
        raise InputError("synthetic must be boolean when supplied")
    coords = data.get("coordinates", {})
    if (coords.get("units") != "m" or coords.get("up_axis") != "+Z"
            or not isinstance(coords.get("frame_id"), str) or not coords["frame_id"]):
        raise InputError("Planning coordinates require known meters, +Z and frame_id")
    for container, field in ((coords, "gravity_aligned"), (data, "geometry_complete")):
        if field not in container:
            warnings.append({"code": field + "_unverified", "message": field + " omitted; input completeness/alignment is not verified"})
        elif container[field] is not True:
            raise InputError(field + " explicitly denies or invalidates the supported planning assumption")
    if data.get("reference_route") is not None or data.get("registration") is not None:
        raise InputError("Inline reference_route/registration is unsupported; use --reference-route and --registration")
    for field in ("objects", "structures", "regions"):
        if not isinstance(data.get(field, []), list) or any(not isinstance(item, dict) for item in data.get(field, [])):
            raise InputError(f"{field} must be an array of objects")
    floor_data = data.get("floor")
    if not isinstance(floor_data, dict):
        raise InputError("A single-level floor polygon and height are required")
    if "confirmed" not in floor_data:
        warnings.append({"code": "floor_unverified", "message": "Floor confirmation omitted; using supplied floor without asserting surveyed safety"})
    elif floor_data["confirmed"] is not True:
        raise InputError("floor.confirmed explicitly denies or invalidates the supplied floor")
    if "components" in floor_data:
        parts = floor_data["components"]
        if ("polygon" in floor_data or "holes" in floor_data
                or not isinstance(parts, list) or not parts
                or any(not isinstance(part, dict) for part in parts)):
            raise InputError("floor.components must be a nonempty polygon array, instead of polygon/holes")
        # Same height for every part; preserve gaps and holes, never bridge
        # disconnected reconstructed floor islands by a convex hull.
        parsed = [polygon(part) for part in parts]
        floor = parsed[0] if len(parsed) == 1 else shapely.union_all(parsed)
    else:
        floor = polygon(floor_data)
    height = floor_data.get("height_m")
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height):
        raise InputError("floor.height_m must be finite")
    if data.get("levels", 1) != 1:
        raise InputError("Multiple levels and stairs are unsupported")
    provenance = {"scene_manifest": str(path), "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "input_warnings": warnings, "synthetic_source": "declared" if "synthetic" in data else "fallback_unverified"}
    retain_input_context(data, path, provenance, warnings)
    camera = data.get("camera", {})
    if not isinstance(camera, dict):
        raise InputError("camera must be an object")
    upstream = None
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
                if key == "synthetic" and key not in data:
                    # Inherit a supplied upstream declaration, never guess it.
                    if not isinstance(manifest.get(key), bool):
                        raise InputError("Upstream synthetic must be boolean")
                    data[key] = manifest[key]
                    provenance["synthetic_source"] = "upstream_manifest"
                if manifest.get(key) != data.get(key):
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
                warnings.append({"code": "source_scale_unverified", "message":
                    "Using explicit planning_from_world without redundant scale-confirmation flag; measurement quality is not verified"})
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
        if (not {"positions_m", "trajectory_uri"}.intersection(camera) and (
                "navigation_completion" in data or
                config.camera_height_m is None and "height_above_floor_m" not in camera)):
            positions, invalid = [], 0
            for record in reconstruction.get("cameras", []):
                try:
                    pose = matrix(record["world_from_camera"], "world_from_camera")
                    position = (planning_from_world @ pose)[:3, 3]
                    if not np.isfinite(position).all():
                        raise InputError("Non-finite transformed camera position")
                except (InputError, KeyError, TypeError):
                    invalid += 1
                    continue
                positions.append(position)
            upstream = (np.asarray(positions).reshape((-1, 3)), invalid,
                        (instance_path.parent / instances["reconstruction_manifest"]).resolve())
    provenance["camera_height"] = resolve_camera_height(
        camera, config, floor, float(height), path, coords["frame_id"], warnings, upstream)
    camera_height = provenance["camera_height"]["height_above_floor_m"]
    # body_min/max describe the body, not allowed camera heights. Collision
    # checks include their union with the resolved lens height. The v29 defer
    # keyword remains accepted for callers; there is no body-limit height gate.
    if not np.isfinite(camera_height) or camera_height <= 0:
        raise InputError("Resolved camera height must be finite and above the floor")
    geometries, objects, ids = [], [], set()
    coverage_only_objects = []
    for item, base in items:
        oid = item.get("instance_id")
        if not isinstance(oid, str) or not oid or oid in ids:
            raise InputError("Object instance_id must be unique nonempty string")
        ids.add(oid)
        source_metadata = {key: item[key] for key in
                           ("source_instance_id", "class_id", "source_points", "source_triangle_count") if key in item}
        if source_metadata:
            provenance.setdefault("source_object_metadata", {})[oid] = source_metadata
        name = item.get("class_name")
        if name is not None and not isinstance(name, str):
            raise InputError("class_name must be string or null")
        if not isinstance(item.get("must_see", False), bool):
            raise InputError("must_see must be boolean")
        if name is not None and name.strip().casefold() in COVERAGE_ONLY_CLASSES:
            if item.get("must_see", False):
                warnings.append({"code": "coverage_only_metadata_ignored", "instance_id": oid,
                                 "message": "Wall/ceiling must_see ignored; evaluated through spatial coverage"})
            solid = item.get("solid", "box" in item)
            if not isinstance(solid, bool):
                raise InputError("structure.solid must be boolean")
            geometries.append(Geometry(f"structure_object:{oid}", load_triangles(item, base, config), None, solid=solid))
            coverage_only_objects.append({"instance_id": oid, "class_name": name})
            continue
        objects.append({"instance_id": oid, "class_name": name,
                        "must_see": item.get("must_see", False), "importance": item.get("importance"),
                        "description": item.get("description", "")})
        if item.get("physical_object_id") is not None:
            objects[-1]["physical_object_id"]=item["physical_object_id"]
        solid = item.get("solid", True)
        if not isinstance(solid, bool):
            raise InputError("object.solid must be boolean")
        geometries.append(Geometry(f"object:{oid}", load_triangles(item, base, config), oid, solid=solid))
    for index, item in enumerate(data.get("structures", [])):
        if not isinstance(item.get("solid", "box" in item), bool):
            raise InputError("structure.solid must be boolean")
        geometries.append(Geometry(f"structure:{index}", load_triangles(item, path.parent, config), None,
                                   solid=item.get("solid", "box" in item)))
    overrides = data.get("object_overrides", {})
    if not isinstance(overrides, dict):
        raise InputError("object_overrides must be an object")
    raw_object_ids = {obj["instance_id"] for obj in objects}
    physical_object_ids = {obj.get("physical_object_id", obj["instance_id"]) for obj in objects
                           if isinstance(obj.get("physical_object_id", obj["instance_id"]), str)}
    for oid, override in overrides.items():
        if not isinstance(override, dict):
            raise InputError("Each object override must be an object")
        if oid not in raw_object_ids | physical_object_ids:
            warnings.append({"code": "unused_object_override", "instance_id": oid,
                             "message": "Override has no semantic target; ignored (including coverage-only walls/ceilings)"})
    def apply_override(obj):
        override = overrides.get(obj["instance_id"], {})
        unused = set(override) - {"importance", "must_see", "description"}
        if unused:
            warnings.append({"code": "unused_override_fields", "instance_id": obj["instance_id"],
                             "fields": sorted(unused), "message": "Unconsumed metadata ignored"})
        if "must_see" in override and not isinstance(override["must_see"], bool):
            raise InputError("must_see override must be boolean")
        obj.update({key: value for key, value in override.items() if key not in unused})
    for obj in objects:
        apply_override(obj)
    objects,geometries=group_physical_objects(objects,geometries,provenance,warnings)
    for obj in objects:
        if "source_member_instance_ids" in obj:
            apply_override(obj)
    record_observation_identity(objects,provenance,warnings)
    include_floor = floor_data.get("include_in_coverage", True)
    if not isinstance(include_floor, bool):
        raise InputError("floor.include_in_coverage must be boolean")
    if include_floor:
        floor_triangles = []
        for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(floor)):
            xy = np.asarray(triangle.exterior.coords)[:3, :2]
            floor_triangles.append(np.column_stack([xy, np.full(3, height)]))
        geometries.append(Geometry("floor", np.asarray(floor_triangles), None, obstacle=False))
    deduplicate_surfaces(geometries, config, warnings)
    # Reconstruction cleanup is explicit and navigation-only. Original mesh
    # surfaces/semantic owners still drive coverage, occlusion and object quality.
    # Append after deduplication so it cannot replace the proxy collision faces.
    if "navigation_geometry" in data:
        proxy = data["navigation_geometry"]
        if not isinstance(proxy, list) or any(not isinstance(item, dict) for item in proxy):
            raise InputError("navigation_geometry must be an array of objects")
        navigation = []
        total = 0
        for index, item in enumerate(proxy):
            solid = item.get("solid", "box" in item)
            if not isinstance(solid, bool):
                raise InputError("navigation_geometry.solid must be boolean")
            triangles = load_triangles(item, path.parent, config)
            total += len(triangles)
            if total > config.max_triangles:
                raise InputError("Navigation geometry max_triangles exceeded")
            navigation.append(Geometry(f"navigation_proxy:{index}", np.empty((0, 3, 3)),
                                       None, solid=solid, collision_triangles=triangles))
        for geom in geometries:
            geom.obstacle = False
        geometries.extend(navigation)
        warnings.append({"code": "navigation_proxy_unverified", "message":
            "Explicit navigation geometry replaces collision inputs only; inferred free space and physical safety are not verified"})
    provenance["geometry_sha256"] = geometry_digest(geometries)
    provenance["coverage_only_objects"] = coverage_only_objects
    regions = []
    for item in data.get("regions", []):
        region = polygon(item)
        rid = item.get("region_id")
        if not isinstance(rid, str) or not rid or any(rid == old for old, _ in regions):
            raise InputError("region_id must be unique nonempty string")
        if any(region.intersection(old).area > 1e-8 for _, old in regions):
            warnings.append({"code": "overlapping_regions", "region_id": rid,
                             "message": "Overlap accepted; earlier declared region owns shared area, all region goals remain reported"})
        regions.append((rid, region))
    start = data.get("start_xy_m")
    if start is not None and (np.asarray(start).shape != (2,) or not np.isfinite(start).all()):
        raise InputError("start_xy_m must be finite 2-vector")
    scene = Scene(data["scene_id"], data["capture_id"], data.get("synthetic", False), coords["frame_id"],
                  floor, float(height), geometries, objects, regions, start, provenance, float(camera_height))
    if "navigation_completion" in data:
        from .navigation_completion import complete_navigation
        from .camera_height import _positions
        capture_positions = []
        if "positions_m" in camera:
            capture_positions = _positions(camera["positions_m"])
        elif "trajectory_uri" in camera:
            trajectory = read_json((path.parent / camera["trajectory_uri"]).resolve())
            for key, expected in (("frame_id",coords["frame_id"]),("units","m"),("up_axis","+Z")):
                if key in trajectory and trajectory[key] != expected:
                    raise InputError("Camera trajectory " + key + " differs from planning coordinates")
            capture_positions = _positions(trajectory.get("positions_m"))
        elif upstream is not None:
            capture_positions = upstream[0]
        # Explicitly unassigned/unknown surfaces are not wall/floor evidence.
        # Unnamed entries in structures retain the existing structural contract.
        structural_names = COVERAGE_ONLY_CLASSES | {"floor", "floors", "바닥"}
        structure_ids = {"structure:"+str(i) for i,item in enumerate(data.get("structures",[]))
                         if item.get("class_name") is None or isinstance(item.get("class_name"),str)
                         and item["class_name"].strip().casefold() in structural_names}
        structure_ids.update("structure_object:"+item["instance_id"] for item in coverage_only_objects)
        scene = complete_navigation(scene,config,data["navigation_completion"],capture_positions,structure_ids)
    return scene
