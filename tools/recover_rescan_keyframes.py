"""Recover estimated cameras for original Rescan RGB keyframes.

Depth-frame FPFH/RANSAC+ICP supplies a seed. RGB-to-render SIFT/PnP is an
independent RGB camera check: unmatched/poor frames retain no geometric link.
Official camera poses/calibration are absent; none are invented as ground truth.
"""
import argparse
import json
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from build_rescan_review import read_cloud, cloud_geometry, digest, PROJECT


def depth_records(path):
    raw = path.read_bytes()
    offsets, pos = [], 0
    while pos + 4 <= len(raw):
        # Actual ScanNet capture stores two double timestamp arrays at tail.
        if offsets and len(raw) - pos in (8 * len(offsets), 16 * len(offsets)):
            break
        size = struct.unpack_from("<I", raw, pos)[0]
        if not 0 < size < 640 * 480 * 4 or pos + 4 + size > len(raw):
            raise ValueError(f"Invalid depth record at {pos}")
        offsets.append((pos + 4, size))
        pos += 4 + size
    if len(raw) - pos != 16 * len(offsets):
        raise ValueError("Expected verified depth and RGB timestamp arrays")
    timestamps = np.frombuffer(raw, dtype="<f8", offset=pos).reshape(2, len(offsets)).copy()
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps, axis=1) >= 0).all():
        raise ValueError("Invalid/non-monotonic timestamps")
    return raw, offsets, timestamps


def prepare_frames(data, depth_data, output, stride=60):
    frames = []
    cache = output / "frames-cache"
    cache.mkdir(exist_ok=True)
    capture_metadata=json.loads((data/"source.json").read_text(encoding="utf-8"))
    scene=capture_metadata["scene"]
    for epoch, source_epoch in enumerate(capture_metadata["epochs"]):
        raw, records, times = depth_records(depth_data / f"{scene}_{source_epoch}.depth")
        selected = set(range(0, len(records), stride))
        capture = cv2.VideoCapture(str(data / f"{scene}_{source_epoch}.h264"))
        index = 0
        while capture.grab():
            if index in selected:
                success, image = capture.retrieve()
                if not success:
                    raise ValueError("RGB frame decode failed")
                name = f"frame-{epoch}-{index:06d}"
                resized = cv2.resize(image, (648, 484))
                cv2.imwrite(str(output / (name + ".jpg")), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                payload, size = records[index]
                compressed, decoded = cache / (name + ".depth-record"), cache / (name + ".raw")
                if not decoded.exists():
                    compressed.write_bytes(raw[payload:payload + size])
                    subprocess.run([str(PROJECT / "third_party/rescan-depth/decode.exe"), str(compressed), str(decoded), "307200"], check=True)
                depth = np.fromfile(decoded, dtype="<u2").reshape(480, 640)
                if not .02 < np.mean((depth > 300) & (depth < 5000)) <= 1:
                    raise ValueError("Decoded depth validity is implausible")
                frames.append({"id": name, "epoch": epoch, "source_frame_index": index,
                               "source_timestamp": float(times[1, index]), "depth_timestamp": float(times[0, index]),
                               "time_s": float(times[1, index] - times[1, 0]),
                               "rgb_depth_dt_ms": float(1000 * abs(times[1, index] - times[0, index])),
                               "url": "/rescan/" + name + ".jpg", "depth_path": str(decoded),
                               "width": 648, "height": 484, "original_width": image.shape[1], "original_height": image.shape[0],
                               "status": "unlocalized"})
            index += 1
        capture.release()
        if index != len(records):
            raise ValueError(f"RGB/depth frame counts differ: {index}/{len(records)}")
        print(f"Epoch {epoch}: {index} RGB/depth frames, {len(selected)} samples; timestamps checked", flush=True)
    (output / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
    return frames


def prepare_features(p, voxel=.045):
    down = p.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, feature


def depth_cloud(depth, focal=570.5):
    y, x = np.mgrid[0:480:3, 0:640:3]
    z = depth[::3, ::3].astype(float) / 1000
    valid = (z > .35) & (z < 4.0)
    xyz = np.column_stack(((x[valid] - 319.5) * z[valid] / focal,
                           (y[valid] - 239.5) * z[valid] / focal, z[valid]))
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.06, max_nn=30))
    p.orient_normals_towards_camera_location(np.zeros(3))
    return p


def localize_depth(source, target, target_feature, initial=None):
    down, feature = prepare_features(source)
    candidates = []
    if initial is not None:
        candidates.append(initial)
    reg = o3d.pipelines.registration
    for _ in range(2):
        result = reg.registration_ransac_based_on_feature_matching(down, target, feature, target_feature,
            True, .08, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(.9), reg.CorrespondenceCheckerBasedOnDistance(.08)],
            reg.RANSACConvergenceCriteria(80000, .999))
        candidates.append(result.transformation)
    refined = []
    for transform in candidates:
        result = reg.registration_icp(source, target, .055, transform,
            reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.035)),
            reg.ICPConvergenceCriteria(max_iteration=60))
        position = result.transformation[:3, 3]
        if .1 < position[1] < 2.5:
            refined.append(result)
    if not refined:
        return None
    return max(refined, key=lambda r: r.fitness - r.inlier_rmse * 2)


def render(mesh, c2w, focal=585.225, width=648, height=484):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    K = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]], dtype=np.float32)
    rays = scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(c2w).astype(np.float32)), width, height)
    hits = scene.cast_rays(rays)
    depth = hits["t_hit"].numpy()
    ids = hits["primitive_ids"].numpy()
    valid = np.isfinite(depth)
    colors = np.zeros((height, width, 3), np.uint8)
    face = np.asarray(mesh.triangles)[ids[valid]]
    uv = hits["primitive_uvs"].numpy()[valid]
    weights = np.column_stack((1 - uv.sum(1), uv))
    vertex_colors = np.asarray(mesh.vertex_colors)
    colors[valid] = np.clip(np.sum(vertex_colors[face] * weights[:, :, None], axis=1) * 255, 0, 255).astype(np.uint8)
    rays_np = rays.numpy()
    xyz = rays_np[:, :, :3] + rays_np[:, :, 3:] * np.where(valid, depth, 0)[:, :, None]
    return cv2.cvtColor(colors, cv2.COLOR_RGB2BGR), xyz, depth


def localize_rgb(mesh, image, seed, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    simulated, xyz, depth = render(mesh, seed)
    sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=.015)
    k1, d1 = sift.detectAndCompute(simulated, (np.isfinite(depth) * 255).astype(np.uint8))
    k2, d2 = sift.detectAndCompute(image, None)
    if d1 is None or d2 is None:
        diagnostics["failure"] = "no_descriptors"
        return None, simulated
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in matches if a.distance < .78 * b.distance]
    pairs = []
    for m in good:
        x, y = map(lambda v: int(round(v)), k1[m.queryIdx].pt)
        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and np.isfinite(depth[y, x]):
            pairs.append((xyz[y, x], k2[m.trainIdx].pt))
    # Surfel colors are too smooth for many SIFT descriptors. Nearby rendered
    # corners can still be checked against the ORIGINAL photograph with
    # bidirectional optical flow. These are estimates, not provided poses.
    a = cv2.GaussianBlur(cv2.cvtColor(simulated, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = cv2.erode((np.isfinite(depth) * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    corners = cv2.goodFeaturesToTrack(a, 800, .015, 10, mask=mask)
    if corners is not None:
        moved, okay, error = cv2.calcOpticalFlowPyrLK(a, b, corners, None, winSize=(31, 31), maxLevel=4)
        back, back_okay, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None, winSize=(31, 31), maxLevel=4)
        for p, q, r, ok, bok, e in zip(corners[:, 0], moved[:, 0], back[:, 0], okay.ravel(), back_okay.ravel(), error.ravel()):
            if not ok or not bok or e > 28 or np.linalg.norm(p-r) > 1.5 or np.linalg.norm(p-q) > 85:
                continue
            x, y = np.rint(p).astype(int)
            u, v = np.rint(q).astype(int)
            if min(x,y,u,v) < 5 or x >= 643 or u >= 643 or y >= 479 or v >= 479:
                continue
            aa = a[y-4:y+5,x-4:x+5].astype(float).ravel()
            bb = b[v-4:v+5,u-4:u+5].astype(float).ravel()
            if np.std(aa) < 3 or np.std(bb) < 3 or np.corrcoef(aa,bb)[0,1] < .55:
                continue
            pairs.append((xyz[y,x], q))
    if len(pairs) < 10:
        diagnostics.update(failure="insufficient_matches", match_count=len(pairs))
        return None, simulated
    obj = np.array([p[0] for p in pairs], np.float64)
    pixels = np.array([p[1] for p in pairs], np.float64)
    best = None
    for focal in (530., 565., 585.225, 610., 645.):
        K = np.array([[focal, 0, 323.5], [0, focal, 241.5], [0, 0, 1.]])
        okay, rv, tv, inliers = cv2.solvePnPRansac(obj, pixels, K, None, iterationsCount=2000,
            reprojectionError=4., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if okay and inliers is not None and len(inliers) >= 10 and (best is None or len(inliers) > len(best[2])):
            best = (rv, tv, inliers.ravel(), focal)
    if best is None:
        diagnostics.update(failure="insufficient_pnp_inliers", match_count=len(pairs))
        return None, simulated
    rv, tv, inliers, focal = best
    def residual(params):
        K = np.array([[params[6], 0, 323.5], [0, params[6], 241.5], [0, 0, 1.]])
        predicted = cv2.projectPoints(obj[inliers], params[:3], params[3:6], K, None)[0][:, 0]
        return (predicted - pixels[inliers]).ravel()
    fit = least_squares(residual, np.r_[rv.ravel(), tv.ravel(), focal], loss="soft_l1",
                        bounds=(np.r_[[-np.inf] * 6, 450.], np.r_[[np.inf] * 6, 750.]))
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    spread = np.ptp(pixels[inliers], axis=0)
    w2c = np.eye(4)
    w2c[:3, :3] = cv2.Rodrigues(fit.x[:3])[0]
    w2c[:3, 3] = fit.x[3:6]
    c2w = np.linalg.inv(w2c)
    # Reject spurious repetitive-floor/chair fits and large seed disagreements.
    offset = np.linalg.norm(c2w[:3, 3] - seed[:3, 3])
    diagnostics.update(match_count=len(pairs), inlier_count=len(inliers), median_px=float(np.median(errors)),
                       spread=spread.tolist(), offset_m=float(offset))
    if np.median(errors) > 3 or min(spread) < 60 or offset > .3 or not .1 < c2w[1, 3] < 2.5:
        diagnostics["failure"] = "reprojection_spread_or_seed_gate"
        return None, simulated
    return {"camera_to_world": c2w.tolist(), "focal_px": float(fit.x[6]),
            "rgb_inlier_count": len(inliers), "rgb_match_count": len(pairs),
            "reprojection_median_px": float(np.median(errors)), "inlier_spread_px": spread.tolist(),
            "rgb_depth_camera_offset_m": float(offset), "status": "estimated_rgb_checked"}, simulated


def recover(data, depth_data, output, stride=60):
    path = output / "review.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frames_file = output / "frames.json"
    frames = json.loads(frames_file.read_text()) if frames_file.exists() else prepare_frames(data, depth_data, output, stride)
    clouds = [read_cloud(data / f"{manifest['scene']}_{epoch}.ply") for epoch in manifest['source_epochs']]
    transform = np.array(manifest["registration"]["after_to_before"])
    clouds[1]["xyz"] = clouds[1]["xyz"] @ transform[:3, :3].T + transform[:3, 3]
    clouds[1]["normals"] = clouds[1]["normals"] @ transform[:3, :3].T
    targets = [prepare_features(cloud_geometry(c)) for c in clouds]
    meshes = [o3d.io.read_triangle_mesh(str(output / f"cache/{epoch}-depth8.ply")) for epoch in (0, 1)]
    # Reconstructed colors can be empty/averaged in sparse cells; anchor to raw
    # measured vertex colors for localization, never AI/synthetic textures.
    for epoch, mesh in enumerate(meshes):
        nearest = cKDTree(clouds[epoch]["xyz"]).query(np.asarray(mesh.vertices))[1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(clouds[epoch]["rgb"][nearest] / 255.)
    previous = [None, None]
    for f in frames:
        if f["status"] == "estimated_rgb_checked":
            previous[f["epoch"]] = np.array(f["camera_to_world"])
            continue
        if f.get("depth_camera_to_world"):
            from types import SimpleNamespace
            result = SimpleNamespace(transformation=np.array(f["depth_camera_to_world"]),
                                     fitness=f["depth_fitness"], inlier_rmse=f["depth_rmse_m"])
        else:
            depth = np.fromfile(f["depth_path"], dtype="<u2").reshape(480, 640)
            source = depth_cloud(depth)
            result = localize_depth(source, *targets[f["epoch"]], previous[f["epoch"]])
        f["attempted"] = True
        if result is not None:
            f["depth_fitness"], f["depth_rmse_m"] = result.fitness, result.inlier_rmse
            if result.fitness > .50 and result.inlier_rmse < .04:
                previous[f["epoch"]] = result.transformation
                image = cv2.imread(str(output / (f["id"] + ".jpg")))
                rgb, simulated = localize_rgb(meshes[f["epoch"]], image, result.transformation)
                cv2.imwrite(str(output / "frames-cache" / (f["id"] + "-seed.jpg")), simulated)
                if rgb:
                    f.update(rgb)
                else:
                    f["depth_camera_to_world"] = result.transformation.tolist()
                    f["status"] = "depth_only_rgb_unverified"
        print(f"{f['id']}: {f['status']} depth={f.get('depth_fitness',0):.2f} rgb={f.get('rgb_inlier_count',0)}", flush=True)
        frames_file.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    manifest["keyframes"] = frames
    manifest["camera_status"] = "estimated_cameras_with_per_frame_checks"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--depth-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=60)
    args = parser.parse_args()
    recover(args.data, args.depth_data, args.output, args.stride)
