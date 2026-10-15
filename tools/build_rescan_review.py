"""Real Rescan mesh review; segmentation is context, never the change detector.

Compare surfaces in a shared coordinate frame, not triangle/vertex numbering.
Require original-point support and stability across Poisson resolutions. Same
capture split and reconstruction controls quantify artifacts, not accuracy.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.ndimage import label
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
PACKED = np.dtype([("xyz", "<f4", (3,)), ("normal", "<f4", (3,)),
                   ("rgb", "u1", (3,)), ("region", "u1"), ("instance", "<u4")])
TO_DISPLAY = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_cloud(path):
    data = PlyData.read(path)["vertex"].data
    xyz = np.column_stack([data[n] for n in ("x", "y", "z")]).astype(np.float64)
    normals = np.column_stack([data[n] for n in ("nx", "ny", "nz")]).astype(np.float64)
    rgb = np.column_stack([data[n] for n in ("red", "green", "blue")])
    if not np.isfinite(xyz).all() or not np.isfinite(normals).all():
        raise ValueError("Invalid source coordinates")
    return {"xyz": xyz, "normals": normals, "rgb": rgb,
            "instance": data["instance_idx"].copy(), "class": data["class_idx"].copy()}


def cloud_geometry(cloud, indices=None):
    p = o3d.geometry.PointCloud()
    indices = slice(None) if indices is None else indices
    p.points = o3d.utility.Vector3dVector(cloud["xyz"][indices])
    p.normals = o3d.utility.Vector3dVector(cloud["normals"][indices])
    p.colors = o3d.utility.Vector3dVector(cloud["rgb"][indices] / 255.)
    return p


def reconstruct(cloud, path, depth=8, indices=None):
    if path.exists():
        return o3d.io.read_triangle_mesh(str(path))
    p = cloud_geometry(cloud, indices)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p, depth=depth, scale=1.1, n_threads=4)
    # Poisson extrapolates into unobserved space. Keep only source-supported
    # vertices; no sealed or watertight mesh is assumed after this trimming.
    source = cloud["xyz"] if indices is None else cloud["xyz"][indices]
    support = cKDTree(source).query(np.asarray(mesh.vertices), workers=4)[0]
    mesh.remove_vertices_by_mask((support > .045) | (np.asarray(density) < np.quantile(density, .01)))
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise ValueError("Reconstruction has no supported triangles")
    o3d.io.write_triangle_mesh(str(path), mesh)
    return mesh


def surface_distance(mesh, xyz):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(xyz, dtype=np.float32))).numpy()


def stats(values):
    return {"median_m": float(np.median(values)), "p95_m": float(np.quantile(values, .95)),
            "p99_m": float(np.quantile(values, .99)), "maximum_m": float(np.max(values))}


def register(clouds):
    # The release is already approximately aligned. Refine ONLY close
    # correspondences, with a robust kernel, to avoid following moved objects.
    source, target = [cloud_geometry(c).voxel_down_sample(.025) for c in (clouds[1], clouds[0])]
    estimate = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=.02))
    result = o3d.pipelines.registration.registration_icp(source, target, .045, np.eye(4), estimate,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    transform = result.transformation
    rotation_deg = np.rad2deg(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
    translation = np.linalg.norm(transform[:3, 3])
    if translation > .15 or rotation_deg > 5 or result.fitness < .4:
        raise ValueError("Close-overlap registration failed; inspect pair before change detection")
    clouds[1]["xyz"] = clouds[1]["xyz"] @ transform[:3, :3].T + transform[:3, 3]
    clouds[1]["normals"] = clouds[1]["normals"] @ transform[:3, :3].T
    return {"after_to_before": transform.tolist(), "method": "close-overlap robust point-to-plane ICP",
            "fitness": result.fitness, "inlier_rmse_m": result.inlier_rmse,
            "translation_m": float(translation), "rotation_deg": float(rotation_deg),
            "caveat": "Conditional on released approximate alignment; inlier RMSE is not full-scene ground-truth accuracy"}


def connected_regions(xyz, mask, voxel=.10, minimum=60):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    grid = np.floor(xyz[indices] / voxel).astype(int)
    grid -= grid.min(0)
    shape = grid.max(0) + 1
    if np.prod(shape) > 20_000_000:
        raise ValueError("Region grid too large")
    occupied = np.zeros(shape, bool)
    occupied[tuple(grid.T)] = True
    components, _ = label(occupied, np.ones((3, 3, 3)))
    membership = components[tuple(grid.T)]
    sizes = np.bincount(membership)
    return [indices[membership == k] for k in sorted(range(1, len(sizes)), key=lambda k: -sizes[k])
            if sizes[k] >= minimum]


def pack_geometry(geometry, cloud, ids, origin, path, triangles):
    if triangles:
        geometry = geometry.simplify_quadric_decimation(70000)
        geometry.compute_vertex_normals()
        vertices = np.asarray(geometry.vertices)
        normals = np.asarray(geometry.vertex_normals)
        faces = np.asarray(geometry.triangles)
        support, nearest = cKDTree(cloud["xyz"]).query(vertices)
        # Remove any triangles whose simplification introduced unsupported
        # vertices, preserving holes instead of manufacturing evidence.
        faces = faces[(support[faces] <= .045).all(1)]
        v = faces.ravel()
        positions, normals = vertices[v], normals[v]
        nearest = nearest[v]
    else:
        cells = np.floor(cloud["xyz"] / .02).astype(int)
        _, nearest = np.unique(cells, axis=0, return_index=True)
        positions, normals = cloud["xyz"][nearest], cloud["normals"][nearest]
    packed = np.empty(len(positions), PACKED)
    packed["xyz"] = positions @ TO_DISPLAY.T - origin
    packed["normal"] = normals @ TO_DISPLAY.T
    packed["rgb"] = cloud["rgb"][nearest]
    packed["region"] = ids[nearest]
    packed["instance"] = cloud["instance"][nearest]
    packed.tofile(path)
    return {"url": "/rescan/" + path.name, "count": len(packed), "stride": PACKED.itemsize,
            "primitive": "triangles" if triangles else "points", "sha256": digest(path),
            "triangle_count": len(packed) // 3 if triangles else 0}


def build(data, output, threshold=.06):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    paths = [data / f"scene_i_{epoch}.ply" for epoch in (1, 2)]
    clouds = [read_cloud(p) for p in paths]
    registration = register(clouds)
    print("Registration:", registration, flush=True)
    meshes = []
    for epoch, cloud in enumerate(clouds):
        pair = []
        for depth in (8, 9):
            print(f"Reconstructing epoch {epoch} depth {depth}", flush=True)
            pair.append(reconstruct(cloud, cache / f"{epoch}-depth{depth}.ply", depth))
        meshes.append(pair)
    controls = []
    sample = np.linspace(0, len(clouds[0]["xyz"]) - 1, 18000).astype(int)
    for name, indices in [("resolution", None), ("split_even", np.arange(0, len(clouds[0]["xyz"]), 2)),
                          ("split_odd", np.arange(1, len(clouds[0]["xyz"]), 2))]:
        mesh = meshes[0][1] if indices is None else reconstruct(clouds[0], cache / f"control-{name}.ply", 8, indices)
        distance = surface_distance(mesh, clouds[0]["xyz"][sample])
        controls.append({"name": name, "sample_count": len(sample), **stats(distance),
                         "above_6cm_fraction": float(np.mean(distance > threshold))})
    effective_threshold = max(threshold, max(c["p99_m"] for c in controls))
    all_xyz = np.vstack([c["xyz"] @ TO_DISPLAY.T for c in clouds])
    origin = (all_xyz.min(0) + all_xyz.max(0)) / 2
    regions, displays, object_context = [], [], []
    next_id = 1
    for epoch, cloud in enumerate(clouds):
        opposite = clouds[1 - epoch]
        nn = cKDTree(opposite["xyz"]).query(cloud["xyz"], workers=4)[0]
        distances = [surface_distance(m, cloud["xyz"]) for m in meshes[1 - epoch]]
        candidate8 = (distances[0] > effective_threshold) & (nn > effective_threshold)
        candidate9 = (distances[1] > effective_threshold) & (nn > effective_threshold)
        mask = candidate8 & candidate9
        ids = np.zeros(len(nn), np.uint8)
        print(f"Epoch {epoch}: {mask.sum():,} persistent candidates", flush=True)
        for indices in connected_regions(cloud["xyz"], mask):
            if next_id > 254:
                raise ValueError("Too many regions")
            ids[indices] = next_id
            xyz = cloud["xyz"][indices]
            display_xyz = xyz @ TO_DISPLAY.T
            objects, counts = np.unique(cloud["instance"][indices], return_counts=True)
            classes, class_counts = np.unique(cloud["class"][indices], return_counts=True)
            floor_share = float(np.mean(np.isin(cloud["class"][indices], [1, 2, 22])))
            regions.append({"id": next_id, "epoch": epoch, "status": "attention_candidate",
                            "point_count": len(indices), "bounds": {"min": display_xyz.min(0).tolist(), "max": display_xyz.max(0).tolist()},
                            "source_bounds": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
                            "distances": stats(distances[0][indices]), "raw_distances": stats(nn[indices]),
                            "stability": float(np.mean(candidate9[indices])),
                            "context_instance_ids": objects[np.argsort(-counts)[:4]].tolist(),
                            "context_classes": classes[np.argsort(-class_counts)[:3]].tolist(),
                            "coverage_warning": floor_share > .7,
                            "reason": "Missing opposite surface; coverage/occlusion must be reviewed" if floor_share > .7 else "Surface difference supported by original points and both mesh resolutions",
                            "keyframes": [[], []]})
            np.asarray(indices, dtype="<u4").tofile(output / f"region-{next_id:03d}.indices.bin")
            next_id += 1
        display = {"epoch": epoch, "source_points": len(cloud["xyz"]),
                   "mesh": pack_geometry(meshes[epoch][0], cloud, ids, origin, output / f"{epoch}-mesh.bin", True),
                   "points": pack_geometry(None, cloud, ids, origin, output / f"{epoch}-points.bin", False),
                   "full_mesh_vertices": len(meshes[epoch][0].vertices), "full_mesh_triangles": len(meshes[epoch][0].triangles),
                   "resolution_candidate_fraction": [float(candidate8.mean()), float(candidate9.mean())],
                   "persistent_candidate_fraction": float(mask.mean())}
        displays.append(display)
        for instance in np.unique(cloud["instance"]):
            points = cloud["xyz"][cloud["instance"] == instance]
            object_context.append({"epoch": epoch, "instance": int(instance), "count": len(points),
                                   "centroid": points.mean(0).tolist(), "semantic_class": int(np.median(cloud["class"][cloud["instance"] == instance]))})
    manifest = {"version": 1, "dataset": "Rescan", "scene": "scene_i", "source_epochs": [1, 2], "synthetic": False,
                "source_url": "https://rescan.cs.princeton.edu/", "code_url": "https://github.com/mhalber/Rescan",
                "source_files": [{"path": str(p.resolve()), "sha256": digest(p), "bytes": p.stat().st_size} for p in paths],
                "method": "Segmentation-independent surface attention candidates",
                "unit": "m", "display_origin": origin.tolist(), "display_rotation": TO_DISPLAY.tolist(),
                "bounds": {"min": all_xyz.min(0).tolist(), "max": all_xyz.max(0).tolist()},
                "registration": registration, "threshold_m": effective_threshold, "requested_threshold_m": threshold,
                "mesh_reconstruction": {"method": "Poisson", "depths": [8, 9], "scale": 1.1, "maximum_source_support_distance_m": .045,
                                        "browser_maximum_triangles": 70000, "segmentation_transfer": "nearest original surfel; visualization context only"},
                "controls": controls, "control_scope": "Same capture; reconstruction/sampling artifact diagnostics, not independent change-detection accuracy",
                "displays": displays, "regions": regions, "object_context": object_context,
                "keyframes": [], "camera_status": "pending_recovery",
                "limitations": ["Surface difference is an attention signal, not proof of object addition/removal",
                                "No official per-frame camera poses or calibration in inspected release",
                                "Occluded/unscanned areas cannot establish absence",
                                "One real pair and same-capture controls do not establish general accuracy"]}
    (output / "review.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"regions": len(regions), "threshold_m": effective_threshold, "controls": controls}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.06)
    args = parser.parse_args()
    if args.threshold <= 0:
        parser.error("threshold must be positive")
    build(args.data, args.output, args.threshold)
