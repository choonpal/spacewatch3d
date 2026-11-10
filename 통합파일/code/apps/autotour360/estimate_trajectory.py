#!/usr/bin/env python3
"""Estimate a timestamped spherical camera trajectory using CPU COLMAP SfM.

No unit-length step integration or invented poses: only registered cameras are
exported. Missing intervals remain gaps. Coordinates have monocular (not metric)
scale. The floor is fitted to observed sparse points, independently of the path.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

ENGINE_VERSION = "sphere-sfm-1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    if abs(stream["width"] / stream["height"] - 2) > .03:
        raise ValueError("스티칭이 완료된 2:1 비율의 360도 영상을 선택해 주세요.")
    return {"width": stream["width"], "height": stream["height"], "duration": float(info["format"]["duration"])}


def extract_frames(video, work, fps, width, report):
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    manifest = work / "frames.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())
        if entries and all((frames / e["name"]).exists() for e in entries):
            return frames, entries
    # select + showinfo preserves the source presentation timestamp, including VFR.
    interval = 1 / fps - 1e-5
    vf = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',scale={width}:{width//2},showinfo"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-threads", "4", "-i", str(video), "-an", "-vf", vf, "-vsync", "0", "-q:v", "3", str(frames / "frame_%06d.jpg")]
    entries = []
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    with (work / "extraction.log").open("w") as log:
        for line in process.stderr:
            log.write(line)
            match = re.search(r"\bn:\s*(\d+).*?\bpts_time:([\d.eE+-]+)", line)
            if match:
                i, timestamp = int(match[1]), float(match[2])
                entries.append({"name": f"frame_{i+1:06d}.jpg", "t": timestamp})
                if len(entries) % 10 == 0:
                    report("frames", min(18, 3 + len(entries) / 15), f"프레임 준비 중 · {len(entries)}개")
    if process.wait() != 0 or len(entries) < 6:
        raise ValueError("경로 추정에 필요한 영상 프레임을 읽지 못했습니다.")
    write_json(manifest, entries)
    return frames, entries


def fit_floor(points, cameras, rotations, rng=None):
    """Find an observed plane below cameras with a normal near image-up.

    Does not assume any real-world camera height. The estimate is rejected when
    too few measured points support it or it lies above too many camera centers.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    ups = -rotations[:, :, 1]
    up = np.median(ups, axis=0)
    up /= np.linalg.norm(up)
    center = np.median(cameras, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale = float(np.median(distances))
    if not scale > 1e-8:
        return None, up
    heights = (points - center) @ up
    candidates = points[(heights < -.025 * scale) & (distances < np.quantile(distances, .95))]
    if len(candidates) < 30:
        return None, up
    if len(candidates) > 12000:
        candidates = candidates[rng.choice(len(candidates), 12000, replace=False)]
    threshold = scale * .008
    best = None
    # Include fixed-up hypotheses so feature-poor floors need fewer random triples.
    for iteration in range(1400):
        triple = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = up.copy() if iteration % 3 == 0 else np.cross(triple[1]-triple[0], triple[2]-triple[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal @ up < 0:
            normal = -normal
        if normal @ up < math.cos(math.radians(12)):
            continue
        offset = -float(triple[0] @ normal)
        heights_at_camera = cameras @ normal + offset
        if np.quantile(heights_at_camera, .05) < .035 * scale:
            continue
        residual = np.abs(candidates @ normal + offset)
        inliers = residual < threshold
        score = int(inliers.sum())
        if score >= 25 and (best is None or score > best[0]):
            best = (score, normal, offset, inliers)
    if best is None:
        return None, up
    _, normal, offset, inliers = best
    selected = candidates[inliers]
    centroid = selected.mean(axis=0)
    _, _, vh = np.linalg.svd(selected - centroid, full_matrices=False)
    normal = vh[-1]
    if normal @ up < 0:
        normal = -normal
    offset = -float(normal @ centroid)
    camera_heights = cameras @ normal + offset
    residuals = np.abs(selected @ normal + offset)
    spread = np.linalg.svd(selected - centroid, compute_uv=False)
    valid = (normal @ up > math.cos(math.radians(12)) and np.quantile(camera_heights, .05) > .025 * scale and spread[1] > scale * .15)
    if not valid:
        return None, up
    return {"normal": normal.tolist(), "offset": offset, "support": len(selected), "candidateCount": len(candidates), "medianResidual": float(np.median(residuals)), "cameraHeight": float(np.median(camera_heights)), "method": "sparse-point-ransac", "estimated": True}, up


def export_trajectory(reconstruction, entries, source, info):
    timestamp = {e["name"]: e["t"] for e in entries}
    samples, rotations, centers = [], [], []
    for image in sorted(reconstruction.images.values(), key=lambda im: timestamp.get(im.name, float("inf"))):
        if not image.has_pose or image.name not in timestamp:
            continue
        world_from_cam = image.cam_from_world().inverse()
        p = np.asarray(world_from_cam.translation)
        r = np.asarray(world_from_cam.rotation.matrix())
        if not (np.isfinite(p).all() and np.isfinite(r).all()):
            continue
        samples.append({"t": timestamp[image.name], "p": p.tolist(), "q": Rotation.from_matrix(r).as_quat().tolist(), "observations": int(image.num_points3D)})
        centers.append(p)
        rotations.append(r)
    if len(samples) < 6:
        raise ValueError("안정적으로 연결된 촬영 위치가 부족합니다. 경로를 생성하지 않았습니다.")
    points = np.array([point.xyz for point in reconstruction.points3D.values() if point.error < 3 and point.track.length() >= 3])
    if len(points) < 30:
        raise ValueError("기하 검증을 통과한 공간 특징점이 부족합니다.")
    centers, rotations = np.array(centers), np.array(rotations)
    floor, up = fit_floor(points, centers, rotations)
    # Reject large jumps and wide temporal gaps instead of joining them visually.
    times = np.array([s["t"] for s in samples])
    dt = np.diff(times)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    speeds = steps / np.maximum(dt, 1e-6)
    median_speed = float(np.median(speeds))
    mad = float(np.median(np.abs(speeds - median_speed)))
    speed_limit = max(median_speed * 6, median_speed + 10 * mad, 1e-8)
    sample_interval = float(np.median(np.diff([e["t"] for e in entries])))
    max_gap = sample_interval * 2.6
    segments, start = [], 0
    for i in range(1, len(samples)):
        if dt[i-1] > max_gap or speeds[i-1] > speed_limit:
            if i - start >= 2:
                segments.append([start, i-1])
            start = i
    if len(samples) - start >= 2:
        segments.append([start, len(samples)-1])
    covered = sum(samples[b]["t"] - samples[a]["t"] for a, b in segments)
    errors = [float(p.error) for p in reconstruction.points3D.values() if np.isfinite(p.error)]
    return {
        "version": 1, "engine": f"COLMAP {pycolmap.__version__} · spherical SfM", "engineVersion": ENGINE_VERSION,
        "source": source, "duration": info["duration"], "coordinates": "camera-to-world; camera axes right/down/forward", "scale": "arbitrary",
        "samples": samples, "segments": segments, "up": up.tolist(), "floor": floor, "maxGap": max_gap,
        "quality": {"inputFrames": len(entries), "registeredFrames": len(samples), "coverage": min(1, covered / max(info["duration"], .001)), "points": len(reconstruction.points3D), "medianReprojectionError": float(np.median(errors)), "unregisteredFrames": len(entries)-len(samples)},
        "warnings": (["floor-unavailable"] if floor is None else []) + (["partial-trajectory"] if covered / info["duration"] < .9 else []) + ["monocular-scale", "no-wall-occlusion"],
    }


def estimate(video, output, work, status=None, fps=2., width=1920, threads=6):
    video, output, work = Path(video).resolve(), Path(output).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    def report(stage, progress, message):
        value = {"state": "running", "stage": stage, "progress": progress, "message": message, "elapsed": round(time.monotonic()-begin, 1)}
        if status:
            write_json(status, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    report("prepare", 1, "원본 영상 확인 중")
    info = probe(video)
    if info["duration"] > 600:
        raise ValueError("한 번에 최대 10분 길이의 영상을 분석할 수 있습니다. 영상을 나누어 주세요.")
    source = {"name": video.name, "size": video.stat().st_size, "sha256": sha256(video)}
    frames, entries = extract_frames(video, work, fps, width, report)
    mask = np.ones((width//2, width), dtype=np.uint8) * 255
    mask[:int(width//2*.08)] = 0
    mask[int(width//2*.78):] = 0  # Mask the photographer and strong nadir distortion.
    mask_path = work / "feature-mask.png"
    cv2.imwrite(str(mask_path), mask)
    database = work / "database.db"
    report("features", 20, f"360도 특징점 추출 중 · {len(entries)}개 프레임")
    extraction = pycolmap.FeatureExtractionOptions(num_threads=threads, use_gpu=False, max_image_size=width)
    extraction.sift.max_num_features = 4500
    extraction.sift.first_octave = 0
    pycolmap.extract_features(database, frames, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model="EQUIRECTANGULAR", camera_mask_path=mask_path),
        extraction_options=extraction, device=pycolmap.Device.cpu)
    report("matching", 40, "프레임 사이의 같은 지점을 찾는 중")
    pycolmap.match_sequential(database, pairing_options=pycolmap.SequentialPairingOptions(overlap=10, quadratic_overlap=True, num_threads=threads),
        matching_options=pycolmap.FeatureMatchingOptions(num_threads=threads, use_gpu=False), device=pycolmap.Device.cpu)
    report("mapping", 60, "촬영 위치와 방향을 복원하는 중")
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    options = pycolmap.IncrementalPipelineOptions(num_threads=threads, random_seed=7, max_num_models=3, min_model_size=8,
        ba_refine_focal_length=False, ba_refine_extra_params=False, ba_local_max_num_iterations=25, ba_global_max_num_iterations=50,
        ba_global_frames_ratio=1.3, ba_global_points_ratio=1.3, max_runtime_seconds=900)
    options.mapper.init_min_tri_angle = 6
    options.mapper.abs_pose_min_num_inliers = 35
    options.mapper.abs_pose_max_error = 6
    count = [0]
    def registered():
        count[0] += 1
        if count[0] % 10 == 0:
            report("mapping", min(91, 60 + 30 * count[0] / len(entries)), f"촬영 위치 복원 중 · {count[0]}개 등록")
    reconstructions = pycolmap.incremental_mapping(database, frames, sparse, options=options, next_image_callback=registered)
    if not reconstructions:
        raise ValueError("촬영 위치를 연결하지 못했습니다. 영상의 흔들림이나 반복 무늬가 원인일 수 있습니다.")
    reconstruction = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    report("floor", 93, "바닥 위치와 경로 품질을 확인하는 중")
    result = export_trajectory(reconstruction, entries, source, info)
    result["processingSeconds"] = round(time.monotonic()-begin, 2)
    write_json(output, result)
    if status:
        write_json(status, {"state": "complete", "stage": "complete", "progress": 100, "message": "촬영 경로 추정 완료", "elapsed": result["processingSeconds"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    try:
        result = estimate(args.video, args.output, args.work, args.status, args.fps, args.width, args.threads)
        print(json.dumps(result["quality"], ensure_ascii=False), flush=True)
    except Exception as error:
        if args.status:
            write_json(args.status, {"state": "failed", "progress": 0, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
