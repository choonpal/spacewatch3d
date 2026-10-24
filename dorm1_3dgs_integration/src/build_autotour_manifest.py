from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AUTO TOUR 360 project from rendered keyframes.")
    parser.add_argument("--render-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Dorm1 · 3DGS 가상 키프레임")
    parser.add_argument("--links-per-scene", type=int, default=2)
    parser.add_argument(
        "--link-mode",
        choices=("nearest", "sequence"),
        default="nearest",
        help="Connect spatial neighbors or previous/next records.",
    )
    return parser.parse_args()


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("zero-length direction")
    return vector / length


def level_camera_basis(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = np.asarray(target - position, dtype=np.float64)
    forward[1] = 0.0
    if np.linalg.norm(forward) < 1e-7:
        raise ValueError("camera and target must differ horizontally for a level panorama")
    forward = normalized(forward)
    right = np.cross(forward, world_up)
    right = normalized(right)
    return right, world_up, forward


def target_pitch(position: np.ndarray, target: np.ndarray) -> float:
    direction = normalized(target - position)
    return math.degrees(math.asin(float(np.clip(direction @ np.array([0.0, 1.0, 0.0]), -1.0, 1.0))))


def hotspot_angles(source: dict, destination: dict) -> tuple[float, float]:
    position = np.asarray(source["camera_position_glb"], dtype=np.float64)
    target = np.asarray(source["look_at_glb"], dtype=np.float64)
    destination_position = np.asarray(destination["camera_position_glb"], dtype=np.float64)
    direction = normalized(destination_position - position)
    right, up, forward = level_camera_basis(position, target)
    yaw = math.degrees(math.atan2(float(direction @ right), float(direction @ forward)))
    pitch = math.degrees(math.asin(float(np.clip(direction @ up, -1.0, 1.0))))
    return yaw, max(-85.0, min(85.0, pitch))


def scene_title(item: dict, index: int) -> str:
    source = item.get("source_camera", {})
    explicit_title = source.get("title") or item.get("title")
    if explicit_title:
        return str(explicit_title)
    obj = source.get("object", {})
    object_name = obj.get("class_name") or obj.get("name") or "객체"
    mode = source.get("camera_mode", "virtual")
    azimuth = source.get("azimuth_offset_deg")
    suffix = f"방위 {azimuth:+.0f}°" if isinstance(azimuth, (int, float)) else mode
    return f"{index + 1}. {object_name} · {suffix}"


def main() -> None:
    args = parse_args()
    records = json.loads(args.render_summary.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("render summary must contain at least one keyframe")
    unique_records = []
    for item in records:
        position = np.asarray(item["camera_position_glb"], dtype=np.float64)
        target = np.asarray(item["look_at_glb"], dtype=np.float64)
        duplicate = any(
            np.linalg.norm(position - np.asarray(other["camera_position_glb"], dtype=np.float64)) < 1e-5
            and np.linalg.norm(target - np.asarray(other["look_at_glb"], dtype=np.float64)) < 1e-5
            for other in unique_records
        )
        if not duplicate:
            unique_records.append(item)
    records = unique_records
    if len(records) > 200:
        raise ValueError("AUTO TOUR supports at most 200 scenes")

    assets = []
    scenes = []
    positions = []
    for index, item in enumerate(records):
        panorama = Path(item["panorama"]).resolve()
        if not panorama.is_file():
            raise FileNotFoundError(panorama)
        asset_id = f"generated-panorama-{index:03d}"
        scene_id = f"generated-keyframe-{index:03d}"
        stat = panorama.stat()
        assets.append(
            {
                "id": asset_id,
                "name": panorama.name,
                "type": "image",
                "size": stat.st_size,
                "lastModified": int(stat.st_mtime * 1000),
            }
        )
        scenes.append(
            {
                "id": scene_id,
                "assetId": asset_id,
                "title": scene_title(item, index),
                "description": item.get("source_camera", {}).get(
                    "description", "512 3D Gaussian Splatting에서 생성한 가상 위치의 360° 이미지"
                ),
                "startTime": 0,
                "view": item.get(
                    "initial_view",
                    {
                        "yaw": 0.0,
                        "pitch": target_pitch(
                            np.asarray(item["camera_position_glb"], dtype=np.float64),
                            np.asarray(item["look_at_glb"], dtype=np.float64),
                        ),
                        "fov": 75.0,
                    },
                ),
                "thumbnail": "",
                "hotspots": [],
            }
        )
        positions.append(np.asarray(item["camera_position_glb"], dtype=np.float64))

    link_count = max(0, min(args.links_per_scene, len(records) - 1))
    for source_index, scene in enumerate(scenes):
        if args.link_mode == "sequence":
            target_indices = [
                target_index
                for target_index in (source_index - 1, source_index + 1)
                if 0 <= target_index < len(records)
            ]
        else:
            distances = [
                (float(np.linalg.norm(positions[source_index] - positions[target_index])), target_index)
                for target_index in range(len(records))
                if target_index != source_index
            ]
            target_indices = [target_index for _, target_index in sorted(distances)[:link_count]]
        for link_index, target_index in enumerate(target_indices):
            yaw, pitch = hotspot_angles(records[source_index], records[target_index])
            scene["hotspots"].append(
                {
                    "id": f"link-{source_index:03d}-{target_index:03d}-{link_index}",
                    "type": "link",
                    "title": scenes[target_index]["title"],
                    "body": "",
                    "targetId": scenes[target_index]["id"],
                    "yaw": yaw,
                    "pitch": pitch,
                }
            )

    project = {
        "version": 1,
        "id": "dorm1-3dgs-generated-keyframes",
        "title": args.title,
        "assets": assets,
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "assets": len(assets), "scenes": len(scenes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
