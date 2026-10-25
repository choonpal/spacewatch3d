"""Project existing mesh instance IDs and original RGB into SceneDiff inputs.

No reconstruction, segmentation, inter-map fitting, or label completion occurs.
The known alignment is applied separately in each frozen registration region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def assign_regions(points, partition):
    q = points[:, [0, 2]] @ partition["axes"].T
    ij = np.floor((q - partition["origin"]) / partition["step"]).astype(int)
    valid = ((ij >= 0) & (ij < partition["labels"].shape)).all(1)
    result = np.zeros(len(points), int)
    result[valid] = partition["labels"][tuple(ij[valid].T)]
    missing = result == 0
    if missing.any():
        pixels = np.argwhere(partition["labels"] > 0)
        centers = partition["origin"] + (pixels + .5) * partition["step"]
        result[missing] = partition["labels"][tuple(pixels[cKDTree(centers).query(q[missing])[1]].T)]
    return result


def view_rotation(yaw, pitch=np.deg2rad(10)):
    s, c, sp, cp = np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)
    return np.array([[c, -s * sp, s * cp], [0, cp, sp], [-s, -c * sp, c * cp]])


def make_grid(size, fov=90):
    focal = (size - 1) / (2 * np.tan(np.deg2rad(fov) / 2))
    K = np.array([[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1]], np.float32)
    y, x = np.mgrid[:size, :size]
    directions = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T
    return K, directions


def rasterize(scene, pose, directions, face_ids, height):
    rays = np.concatenate([np.broadcast_to(pose[:3, 3], directions.shape), directions @ pose[:3, :3].T], -1)
    hit = scene.cast_rays(o3d.core.Tensor(rays.astype("f4")), nthreads=4)
    depth = hit["t_hit"].numpy()
    triangles = hit["primitive_ids"].numpy()
    valid = np.isfinite(depth) & (depth > .04 * height) & (depth < 8 * height) & (triangles < len(face_ids))
    depth = np.where(valid, depth, 0).astype("f4")
    ids = np.zeros(depth.shape, np.int32)
    ids[valid] = face_ids[triangles[valid]]
    return depth, ids, valid


def project_erp(path, view, directions):
    image = np.asarray(Image.open(path).convert("RGB"))
    d = directions @ view.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    h, w = image.shape[:2]
    u = (np.arctan2(d[..., 0], d[..., 2]) / (2 * np.pi) + .5) * w - .5
    v = (np.arcsin(np.clip(d[..., 1], -1, 1)) / np.pi + .5) * h - .5
    # Horizontal ERP seam wraps, vertical poles clamp separately.
    padded = np.pad(image, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    coords = [np.clip(v, 0, h - 1), np.mod(u, w) + 1]
    return np.stack([map_coordinates(padded[..., k], coords, order=1, mode="nearest") for k in range(3)], -1).astype("u1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/scenediff-mesh-rgb-20261024"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/registration-benchmark-20261019/new-meshfirst-20261022"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--views-per-side", type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    out, prior = args.output, args.prior
    (out / "prepared").mkdir(parents=True, exist_ok=True)
    ctx, provenance = read(prior / "alignment-context.json"), read(prior / "input-provenance.json")
    partition = dict(np.load(prior.parent / "regional-analysis-v1/partition.npz"))
    h = ctx["height_u"]
    K, directions = make_grid(args.size)
    objects, sides, snapshots, checks = [], [], {}, []
    for e, scene_name in enumerate(["dorm2", "dorm3"]):
        root = out / "inputs/sam31" / scene_name
        arrays = dict(np.load(root / "mesh_arrays.npz"))
        original = dict(np.load(prior / f"new-inputs/labeled-mesh-{e}.npz"))
        native = np.array(provenance["inputs"][e]["native_mesh_to_canonical"])
        p = transform(arrays["vertices"].astype(float), native)
        assert np.array_equal(arrays["faces"], original["faces"])
        assert np.max(abs(p - original["p"])) < 1e-5
        assert np.array_equal(arrays["rgb"], original["rgb"])
        checks.append({"scene": scene_name, "faces_exact": True, "rgb_exact": True, "canonical_vertex_max_error_u": float(np.max(abs(p - original["p"]))), "mesh_unchanged": True})
        ids = arrays["face_instance_id"].astype("i4")
        report, manifest = read(root / "report.json"), read(root / "input_manifest.json")
        poses = {r["frame"]: np.array(r["camera_to_world"]) for r in read(prior / f"new-inputs/poses-{e}.json")["poses"]}
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(p.astype("f4")), o3d.core.Tensor(arrays["faces"]))
        table = []
        for obj in report["instance_records"]:
            face = np.flatnonzero(ids == obj["instance_id"])
            row = {"id": f"D{e+2}-SAM31-{obj['instance_id']:03d}", "epoch": e, "instance_id": obj["instance_id"], "label": obj["label"], "class_id": obj["class_id"], "faces": len(face), "zone": None}
            if len(face):
                center = p[arrays["faces"][face]].reshape(-1, 3).mean(0)
                member = center[None]
                if e:
                    member = transform(member, np.array(ctx["membership_base_matrix"]))
                    parent = ctx["membership_parent_transform"]
                    t = np.clip((member @ np.array(parent["axis"]) - parent["transition_start_u"]) / parent["transition_width_u"], 0, 1)
                    member += (t*t*(3-2*t))[:, None] * (transform(member, np.array(parent["correction_matrix"])) - member)
                row.update(center_native_canonical=center.tolist(), zone=f"R{assign_regions(member, partition)[0]:02d}")
            table.append(row)
            objects.append(row)
        candidates = []
        for frame in manifest["frames"]:
            if frame["local_frame"] % 3:
                continue
            raw_pose = poses[frame["map_frame"]]
            path = root / frame["file"]
            assert path.exists(), path
            assert sha(path) == frame["sha256"], path
            snapshots[str(path)] = frame["sha256"]
            for angle in [0, 90, 180, 270]:
                rotation = view_rotation(np.deg2rad(angle))
                pose = raw_pose.copy()
                pose[:3, :3] = pose[:3, :3] @ rotation
                depth, instance, valid = rasterize(scene, pose, directions, ids, h)
                counts = dict(zip(*np.unique(instance[instance > 0], return_counts=True)))
                if not counts:
                    continue
                candidates.append({"frame": frame, "path": path, "rotation": rotation, "pose": pose, "depth": depth, "instance": instance, "valid": valid, "counts": counts, "yaw": angle})
        sides.append({"candidates": candidates, "objects": table})
    zone_records = []
    for zone in sorted({o["zone"] for o in objects if o["zone"]}):
        folder = out / "prepared" / zone
        folder.mkdir(exist_ok=True)
        arrays, views, side_counts = {}, [], []
        for e, side in enumerate(sides):
            target = {o["instance_id"] for o in side["objects"] if o["zone"] == zone and o["label"] != "person"}
            # Select using projected mask coverage only, before computing change scores.
            candidates = []
            for item in side["candidates"]:
                score = sum(min(int(item["counts"].get(i, 0)), 1400) for i in target)
                if score >= 40:
                    candidates.append((score, item))
            candidates.sort(key=lambda pair: (-pair[0], pair[1]["frame"]["map_frame"], pair[1]["yaw"]))
            selected = []
            for score, item in candidates:
                if any(abs(item["frame"]["map_frame"] - other["frame"]["map_frame"]) < 3 and item["yaw"] == other["yaw"] for other in selected):
                    continue
                selected.append(item)
                if len(selected) == args.views_per_side:
                    break
            side_counts.append(len(selected))
            for item in selected:
                m = np.eye(4)
                if e:
                    m = np.array(ctx["methods"]["structure"]["region_matrices_after_Geo"])[int(zone[1:])-1] @ np.array(ctx["base_matrix"])
                    m[:3, 3] += np.array(ctx["methods"]["structure"]["errors"]["regions"][zone]["translation_bias_u"])
                scale = float(np.cbrt(np.linalg.det(m[:3, :3])))
                pose = item["pose"].copy()
                pose[:3, :3] = m[:3, :3] @ pose[:3, :3] / scale
                pose[:3, 3] = transform(pose[None, :3, 3], m)[0]
                assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                depth = item["depth"] * scale
                points = (directions * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
                id_map = np.where(np.isin(item["instance"], list(target)), item["instance"], 0)
                local = np.full(id_map.shape, -1, np.int32)
                mapping = {}
                for iid in sorted(np.unique(id_map[id_map > 0])):
                    mask = id_map == iid
                    if mask.sum() >= 16:
                        local[mask] = len(mapping)
                        mapping[str(len(mapping))] = f"D{e+2}-SAM31-{iid:03d}"
                idx = len(views)
                rgb = project_erp(item["path"], item["rotation"], directions)
                arrays[f"rgb_{idx}"] = rgb
                arrays[f"depth_{idx}"] = depth.astype("f4")
                arrays[f"points_{idx}"] = points.astype("f4")
                arrays[f"valid_{idx}"] = item["valid"]
                arrays[f"mask_{idx}"] = local
                arrays[f"pose_{idx}"] = pose.astype("f4")
                arrays[f"K_{idx}"] = K
                Image.fromarray(rgb).save(folder / f"view-{idx:02d}.jpg", quality=94)
                overlay = rgb.copy()
                overlay[local >= 0] = (overlay[local >= 0] * .6 + np.array([20, 200, 230]) * .4).astype("u1")
                Image.fromarray(overlay).save(folder / f"mask-{idx:02d}.jpg", quality=94)
                unprojected = (directions * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
                assert np.max(abs(unprojected - points)) < 1e-5
                views.append({"index": idx, "epoch": e, "map_frame": item["frame"]["map_frame"], "timestamp_s": item["frame"]["timestamp_s"], "yaw_degrees": item["yaw"], "source_rgb_sha256": snapshots[str(item["path"])], "source_rgb": str(item["path"]), "region_to_input_instance": mapping, "valid_fraction": float(item["valid"].mean()), "target_pixels": int((local >= 0).sum())})
        meta = {"zone": zone, "views": views, "side_counts": side_counts, "size": args.size, "height_u": h, "map_error_u": ctx["methods"]["structure"]["errors"]["regions"][zone]["tolerance_u"], "state": "ready" if min(side_counts) >= 2 else "insufficient_counterpart_views"}
        np.savez_compressed(folder / "inputs.npz", **arrays)
        write(folder / "manifest.json", meta)
        zone_records.append(meta)
        print(zone, side_counts, meta["state"], flush=True)
    write(out / "prepared/objects.json", objects)
    write(out / "prepared/manifest.json", {"zones": zone_records, "objects": len(objects), "instances_with_mesh": sum(o["faces"] > 0 for o in objects), "checks": checks, "source_hashes": {**snapshots, str(prior / "alignment-context.json"): sha(prior / "alignment-context.json"), **{str(out / 'inputs/sam31' / s / 'mesh_arrays.npz'): sha(out / 'inputs/sam31' / s / 'mesh_arrays.npz') for s in ['dorm2', 'dorm3']}}, "pi3_inference": False, "sam_inference": False, "new_alignment_fitted": False, "geometry_changed": False, "input_instance_source": "existing SAM3.1 16-second pilot, projected onto unchanged meshes", "scope": "Dorm2 and Dorm3 are different floors; spatial difference candidates, not temporal change ground truth", "seconds": time.time() - started})


if __name__ == "__main__":
    main()
