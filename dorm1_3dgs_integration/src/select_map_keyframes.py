from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select spatially even map keyframes from supplied camera poses."
    )
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument(
        "--cameras-npz",
        type=Path,
        help="Optional pose archive whose camera_to_world_glb matrices override the JSON poses.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        help="Preserve an explicit ordered set of sample indices instead of reselecting positions.",
    )
    return parser.parse_args()


def existing_keyframe_indices(data: dict) -> list[int]:
    for key in ("keyframes", "selected_keyframes", "keyframe_indices"):
        values = data.get(key)
        if isinstance(values, list) and values:
            indices = []
            for value in values:
                if isinstance(value, int):
                    indices.append(value)
                elif isinstance(value, dict) and isinstance(value.get("sample_index"), int):
                    indices.append(value["sample_index"])
            if indices:
                return sorted(set(indices))
    return sorted(
        frame["sample_index"]
        for frame in data.get("frames", [])
        if frame.get("is_keyframe") is True and isinstance(frame.get("sample_index"), int)
    )


def farthest_point_indices(positions: np.ndarray, count: int) -> list[int]:
    if count < 1:
        raise ValueError("--count must be positive")
    count = min(count, len(positions))
    # Keep both traversal endpoints, then maximize distance to the selected set in XZ.
    selected = [0]
    if count > 1 and len(positions) > 1:
        selected.append(len(positions) - 1)
    while len(selected) < count:
        distance = np.min(
            np.linalg.norm(
                positions[:, None, (0, 2)] - positions[selected][None, :, (0, 2)], axis=2
            ),
            axis=1,
        )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return sorted(selected)


def horizontal_forward(frame: dict, frames: list[dict], index: int) -> np.ndarray:
    matrix = np.asarray(frame["camera_to_world_glb"], dtype=np.float64)
    forward = matrix[:3, 2].copy()
    forward[1] = 0.0
    if np.linalg.norm(forward) < 1e-7:
        previous = np.asarray(frames[max(0, index - 1)]["camera_to_world_glb"], dtype=np.float64)[:3, 3]
        following = np.asarray(frames[min(len(frames) - 1, index + 1)]["camera_to_world_glb"], dtype=np.float64)[:3, 3]
        forward = following - previous
        forward[1] = 0.0
    length = float(np.linalg.norm(forward))
    if length < 1e-7:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward /= length
    return forward


def main() -> None:
    args = parse_args()
    data = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("cameras JSON contains no frames")

    if args.cameras_npz:
        archive = np.load(args.cameras_npz)
        matrices = np.asarray(archive["camera_to_world_glb"], dtype=np.float64)
        sample_indices = np.asarray(archive["sample_index"], dtype=np.int64)
        pose_lookup = {int(sample): matrix for sample, matrix in zip(sample_indices, matrices)}
        for frame in frames:
            sample_index = int(frame["sample_index"])
            if sample_index not in pose_lookup:
                raise KeyError(f"sample {sample_index} is missing from {args.cameras_npz}")
            frame["camera_to_world_glb"] = pose_lookup[sample_index].tolist()

    positions = np.asarray(
        [np.asarray(frame["camera_to_world_glb"], dtype=np.float64)[:3, 3] for frame in frames]
    )
    supplied = args.sample_indices or existing_keyframe_indices(data)
    if supplied:
        lookup = {frame["sample_index"]: index for index, frame in enumerate(frames)}
        selected = [lookup[index] for index in supplied if index in lookup]
        if len(selected) != len(supplied):
            missing = sorted(set(supplied) - set(lookup))
            raise KeyError(f"requested sample indices are missing: {missing}")
        method = "explicit_sample_indices" if args.sample_indices else "supplied_keyframes"
    else:
        selected = farthest_point_indices(positions, args.count)
        method = "spatial_farthest_point_xz"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for order, index in enumerate(selected):
        frame = frames[index]
        position = positions[index]
        look_at = position + horizontal_forward(frame, frames, index)
        record = {
            "title": f"맵 키프레임 {order + 1:02d} · 프레임 {frame['source_frame_index']}",
            "description": f"전체 맵 공간 커버리지 지점 · 원본 {frame['timestamp_s']:.1f}초",
            "camera_mode": "map_coverage",
            "selection_method": method,
            "sequence_order": order,
            "sample_index": frame["sample_index"],
            "source_frame_index": frame["source_frame_index"],
            "timestamp_s": frame["timestamp_s"],
            "camera_position_glb": position.tolist(),
            "look_at_glb": look_at.tolist(),
            "camera_to_world_glb": frame["camera_to_world_glb"],
        }
        path = args.output_dir / f"map_keyframe_{order + 1:02d}_camera.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append({"camera_json": str(path.resolve()), **record})

    manifest = {
        "source": str(args.cameras_json.resolve()),
        "pose_source": str(args.cameras_npz.resolve()) if args.cameras_npz else str(args.cameras_json.resolve()),
        "selection_method": method,
        "requested_count": args.count,
        "selected_count": len(records),
        "keyframes": records,
    }
    manifest_path = args.output_dir / "map_keyframe_selection.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(manifest_path.resolve()),
                "method": method,
                "sample_indices": [record["sample_index"] for record in records],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
