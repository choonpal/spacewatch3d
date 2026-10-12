"""Traceable 3D evidence from MultiChange3D's actual labeled epoch 0–1 pair.

Author labels are incoming change results, not predictions. Spatial regions are
not instances. Spatial samples are measured against full original clouds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/3DOM-FBK/multichange3d"
REVISION = "c3af7c9b62ac13d30e4c5bd048d7fe90d391ac48"
SHARE = "https://fbk.sharepoint.com/:f:/s/BENCHMARKS/IgCzHpZzgXjDQI1O1DeZniyZAatqJWchrICnsM1KF7eV2og?e=NCmHOp"
DOWNLOAD_IDS = ["dabb523a-ca5b-43fb-9ced-b7a0a5e9f90e", "13d765f5-45dc-484f-9ca6-8f60c8b6ac3b"]
PLY_DTYPE = np.dtype([(n, "<f4") for n in ("x", "y", "z")] + [(n, "u1") for n in ("red", "green", "blue")] + [("cd_type", "<f4")])
DISPLAY_DTYPE = np.dtype([("position", "<f4", (3,)), ("rgb", "u1", (3,)), ("region_id", "u1"), ("point_index", "<u4")])
REF_DTYPE = np.dtype([("point_index", "<u4"), ("opposite_point_index", "<u4"), ("distance", "<f4")])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labeled_ply(path):
    """Read the official binary layout, preserving original record indices."""
    with Path(path).open("rb") as stream:
        lines = []
        while True:
            raw = stream.readline(4096)
            if not raw or len(lines) > 30:
                raise ValueError("Missing or unexpected PLY header")
            lines.append(raw.decode("ascii").strip())
            if lines[-1] == "end_header":
                break
        offset = stream.tell()
    expected = ["property float x", "property float y", "property float z",
                "property uchar red", "property uchar green", "property uchar blue", "property float cd_type"]
    elements = [s for s in lines if s.startswith("element ")]
    if lines[:2] != ["ply", "format binary_little_endian 1.0"] or len(elements) != 1 or [s for s in lines if s.startswith("property ")] != expected:
        raise ValueError("Expected official XYZRGB/cd_type PLY layout")
    _, element, count = elements[0].split()
    count = int(count)
    if element not in ("params", "vertex") or count < 2 or Path(path).stat().st_size - offset != count * PLY_DTYPE.itemsize:
        raise ValueError("Unexpected PLY element or truncated payload")
    rows = np.memmap(path, mode="r", dtype=PLY_DTYPE, offset=offset, shape=(count,))
    xyz = np.column_stack([rows[n] for n in ("x", "y", "z")])
    rgb = np.column_stack([rows[n] for n in ("red", "green", "blue")])
    if not np.isfinite(xyz).all() or not np.isin(rows["cd_type"], [0, 1]).all():
        raise ValueError("Invalid coordinates or unknown change labels")
    return xyz, rgb, rows["cd_type"] == 1


def spatial_regions(xyz, changed, voxel, min_points):
    """26-connected annotated voxels, returning original record indices."""
    indices = np.flatnonzero(changed)
    if not len(indices):
        return []
    grid = np.floor(xyz[indices].astype(np.float64) / voxel).astype(np.int32)
    grid -= grid.min(0)
    shape = grid.max(0).astype(np.int64) + 1
    if np.prod(shape) > 20_000_000:
        raise ValueError("Region grid too large; increase region voxel")
    occupied = np.zeros(shape, bool)
    occupied[tuple(grid.T)] = True
    components, _ = label(occupied, np.ones((3, 3, 3)))
    membership = components[tuple(grid.T)]
    sizes = np.bincount(membership)
    order = sorted(range(1, len(sizes)), key=lambda k: (-sizes[k], k))
    return [indices[membership == k] for k in order if sizes[k] >= min_points]


def nearest_evidence(points, reference):
    if not len(points) or not len(reference):
        raise ValueError("Both epochs need points")
    return cKDTree(reference, leafsize=128).query(points, k=1, workers=4)


def sample_surface(xyz, indices, voxel=.02, limit=4096):
    """Deterministic spatial sample of ORIGINAL records, never averaged points."""
    cells = np.floor(xyz[indices].astype(np.float64) / voxel).astype(np.int32)
    _, first = np.unique(cells, axis=0, return_index=True)
    if len(first) > limit:
        first = first[np.linspace(0, len(first) - 1, limit).astype(int)]
    return np.sort(indices[first])


def render_indices(xyz, region_ids, voxel):
    # Keep the first original point per spatial/region cell, never a centroid.
    keys = np.column_stack((np.floor(xyz.astype(np.float64) / voxel).astype(np.int32), region_ids))
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def summary(values, threshold):
    return {"median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "maximum": float(values.max()), "above_threshold_count": int(np.count_nonzero(values > threshold)),
            "above_threshold_fraction": float(np.mean(values > threshold))}


def add_change_labels(manifest):
    """Display-only semantics; changed surfaces do not identify moved objects."""
    for region in manifest["regions"]:
        region["change"] = {
            "type": "removed" if region["source_epoch"] == 0 else "added",
            "status": "candidate",
            "source": "dataset_changed_surface_epoch",
            "displacement_m": None,
            "reason": "Changed surface in one epoch; object identity and visibility are not provided",
        }
    manifest["movement_status"] = {
        "status": "unavailable",
        "reason": "No cross-epoch object correspondence; nearest-surface distance is not object displacement",
    }
    return manifest


def build(data_dir, output_dir, *, region_voxel=.04, render_voxel=.01, threshold=.05, min_points=1000):
    if min(region_voxel, render_voxel, threshold) <= 0 or min_points < 1:
        raise ValueError("Positive sizes, threshold and minimum point count required")
    paths = [data_dir / f"{e}.ply" for e in (0, 1)]
    clouds = [read_labeled_ply(p) for p in paths]
    bounds_min = np.minimum(clouds[0][0].min(0), clouds[1][0].min(0))
    bounds_max = np.maximum(clouds[0][0].max(0), clouds[1][0].max(0))
    origin = (bounds_min.astype(np.float64) + bounds_max) / 2
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, regions, displays, controls = [], [], [], []
    next_id = 1
    for epoch, (xyz, rgb, changed) in enumerate(clouds):
        opposite = clouds[1 - epoch][0]
        changed_indices = np.flatnonzero(changed)
        print(f"Epoch {epoch}: extracting regions from {len(changed_indices):,} changed points", flush=True)
        tree = cKDTree(opposite, leafsize=128)
        unchanged = np.flatnonzero(~changed)
        control_indices = unchanged[::max(1, len(unchanged) // 20000)]
        control_d = tree.query(xyz[control_indices], k=1, workers=4)[0]
        controls.append({"epoch": epoch, "sample_count": len(control_indices), "distances": summary(control_d, threshold),
                         "interpretation": "Diagnostic on sampled author-labeled unchanged points; not independent registration validation"})
        region_ids = np.zeros(len(xyz), np.uint8)
        for indices in spatial_regions(xyz, changed, region_voxel, min_points):
            if next_id > 254:
                raise ValueError("Too many browser regions")
            region_ids[indices] = next_id
            sampled = sample_surface(xyz, indices)
            print(f"Region {next_id}: measuring {len(sampled):,} spatial samples against the full opposite cloud", flush=True)
            d, nn = tree.query(xyz[sampled], k=1, workers=4)
            stats = summary(d, threshold)
            refs = np.empty(len(sampled), REF_DTYPE)
            refs["point_index"], refs["opposite_point_index"], refs["distance"] = sampled, nn, d
            name = f"region-{next_id:03d}.bin"
            refs.tofile(output_dir / name)
            indices_name = f"region-{next_id:03d}.indices.bin"
            indices.astype("<u4").tofile(output_dir / indices_name)
            examples = [{"point_index": int(sampled[i]), "xyz": xyz[sampled[i]].tolist(),
                         "opposite_point_index": int(nn[i]), "opposite_xyz": opposite[nn[i]].tolist(), "distance": float(d[i])}
                        for i in np.linspace(0, len(sampled) - 1, min(6, len(sampled))).astype(int)]
            regions.append({"id": next_id, "source_epoch": epoch, "opposite_epoch": 1 - epoch,
                            "type": "before_changed_surface" if epoch == 0 else "after_changed_surface",
                            "type_source": "dataset_cd_type_annotation", "point_count": len(indices),
                            "bounds": {"min": xyz[indices].min(0).tolist(), "max": xyz[indices].max(0).tolist()},
                            "distances": stats, "geometric_check": "supported" if stats["above_threshold_fraction"] >= .5 else "mixed",
                            "measured_point_count": len(sampled), "sampling": {"voxel": .02, "maximum_points": 4096, "method": "first original record per voxel; evenly spaced selection in lexicographic voxel order if above limit"},
                            "all_point_indices": {"url": f"/evidence/{indices_name}", "count": len(indices), "format": "little-endian uint32, original PLY record indices, zero-based", "sha256": sha256(output_dir / indices_name)},
                            "verification_status": "needs_review", "identity": "not_provided", "visibility": "not_assessed_no_depth_or_camera",
                            "point_refs": {"url": f"/evidence/{name}", "count": len(refs), "sha256": sha256(output_dir / name),
                                           "stride_bytes": 12, "format": "little-endian uint32 point_index, uint32 opposite_point_index, float32 distance",
                                           "index_base": 0, "index_space": "original PLY record order within each epoch",
                                           "correspondence": "nearest surface point, not same-object correspondence"},
                            "examples": examples})
            next_id += 1
        selected = render_indices(xyz, region_ids, render_voxel)
        packed = np.empty(len(selected), DISPLAY_DTYPE)
        packed["position"], packed["rgb"] = xyz[selected].astype(np.float64) - origin, rgb[selected]
        packed["region_id"], packed["point_index"] = region_ids[selected], selected
        name = "before.bin" if epoch == 0 else "after.bin"
        packed.tofile(output_dir / name)
        displays.append({"epoch": epoch, "url": f"/evidence/{name}", "count": len(packed), "stride_bytes": 20,
                         "format": "float32 xyz relative to shared display_origin, uint8 RGB, uint8 region_id, uint32 original point_index",
                         "sha256": sha256(output_dir / name)})
        sources.append({"epoch": epoch, "name": f"pointCloud{epoch}.ply",
                        "local_path": paths[epoch].relative_to(PROJECT).as_posix() if paths[epoch].is_relative_to(PROJECT) else str(paths[epoch]),
                        "url": f"https://fbk.sharepoint.com/sites/BENCHMARKS/_layouts/15/download.aspx?UniqueId={DOWNLOAD_IDS[epoch]}",
                        "sha256": sha256(paths[epoch]), "bytes": paths[epoch].stat().st_size,
                        "point_count": len(xyz), "annotated_changed_count": int(changed.sum()),
                        "retained_changed_count": int(np.count_nonzero(region_ids)),
                        "excluded_small_region_count": int(np.count_nonzero(changed & (region_ids == 0)))})
        del tree
    manifest = {"version": 1, "kind": "change_evidence_demo", "dataset": "MultiChange3D", "scene": "Open Space / RGB-D / pair_01",
                "epochs": [0, 1], "synthetic": False, "repository": SOURCE, "revision": REVISION, "dataset_share_url": SHARE,
                "authors": "MultiChange3D authors · 3DOM, Fondazione Bruno Kessler",
                "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "change_input": "Author-provided cd_type labels (0 unchanged, 1 changed); no detector prediction or accuracy claim",
                "registration": {"source": "author-provided common epoch-0 coordinates", "applied_transform": "identity", "independently_verified": False},
                "distance_unit": "original_coordinate_unit", "display_origin": origin.tolist(),
                "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
                "settings": {"region_voxel": region_voxel, "render_voxel": render_voxel, "distance_threshold": threshold, "minimum_region_points": min_points,
                             "distance_reference": "full original opposite epoch, not rendered samples", "region_definition": "26-connected annotated voxels, not object instances"},
                "distance_sampling": {"voxel": .02, "maximum_points_per_region": 4096, "statistics_scope": "spatial samples only; not all region points"},
                "available": {"instance_ids": False, "object_correspondences": False, "keyframes": False, "depth": False, "camera_poses": False},
                "sources": sources, "displays": displays, "unchanged_controls": controls, "regions": regions}
    add_change_labels(manifest)
    (output_dir / "evidence.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "regions": len(regions), "rendered_points": [d["count"] for d in displays],
                      "distance_medians": [r["distances"]["median"] for r in regions]}, indent=2), flush=True)
    return manifest


def verify(manifest_path):
    """Check saved evidence against original source records and full-cloud NN."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clouds = []
    for source in manifest["sources"]:
        path = PROJECT / source["local_path"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Source hash mismatch")
        clouds.append(read_labeled_ply(path))
    checked = 0
    for region in manifest["regions"]:
        path = manifest_path.parent / Path(region["point_refs"]["url"]).name
        if sha256(path) != region["point_refs"]["sha256"]:
            raise ValueError("Evidence hash mismatch")
        refs = np.fromfile(path, dtype=REF_DTYPE)
        if len(refs) != region["measured_point_count"]:
            raise ValueError("Evidence count mismatch")
        xyz, _, changed = clouds[region["source_epoch"]]
        opposite = clouds[region["opposite_epoch"]][0]
        if not changed[refs["point_index"]].all():
            raise ValueError("Evidence includes unlabeled points")
        indices_path = manifest_path.parent / Path(region["all_point_indices"]["url"]).name
        if sha256(indices_path) != region["all_point_indices"]["sha256"]:
            raise ValueError("Region index hash mismatch")
        all_indices = np.fromfile(indices_path, dtype="<u4")
        if len(all_indices) != region["point_count"] or not changed[all_indices].all() or not np.isin(refs["point_index"], all_indices).all():
            raise ValueError("Region membership mismatch")
        actual = np.linalg.norm(xyz[refs["point_index"]].astype(np.float64) - opposite[refs["opposite_point_index"]], axis=1)
        np.testing.assert_allclose(actual, refs["distance"], rtol=2e-6, atol=2e-7)
        sampled = np.linspace(0, len(refs) - 1, min(32, len(refs))).astype(int)
        nearest, _ = nearest_evidence(xyz[refs["point_index"][sampled]], opposite)
        np.testing.assert_allclose(nearest, refs["distance"][sampled], rtol=2e-6, atol=2e-7)
        checked += len(refs)
    for display in manifest["displays"]:
        path = manifest_path.parent / Path(display["url"]).name
        if sha256(path) != display["sha256"]:
            raise ValueError("Display hash mismatch")
        packed = np.fromfile(path, dtype=DISPLAY_DTYPE)
        original, rgb, _ = clouds[display["epoch"]]
        np.testing.assert_allclose(packed["position"].astype(np.float64) + manifest["display_origin"], original[packed["point_index"]], atol=2e-7, rtol=1e-7)
        np.testing.assert_array_equal(packed["rgb"], rgb[packed["point_index"]])
    print(f"Verified {checked:,} evidence records, source hashes, full-cloud nearest distances and shared display coordinates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/change_evidence/multichange3d_open_space/full")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/change_evidence_demo")
    parser.add_argument("--region-voxel", type=float, default=.04)
    parser.add_argument("--render-voxel", type=float, default=.01)
    parser.add_argument("--threshold", type=float, default=.05)
    parser.add_argument("--min-points", type=int, default=1000)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output_dir / "evidence.json")
    else:
        build(args.data_dir.resolve(), args.output_dir.resolve(), region_voxel=args.region_voxel,
              render_voxel=args.render_voxel, threshold=args.threshold, min_points=args.min_points)


if __name__ == "__main__":
    main()
