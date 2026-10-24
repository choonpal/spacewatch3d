from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regularize shared 360-camera poses before 3DGS training.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--max-translation-m", type=float, default=0.08)
    parser.add_argument("--max-rotation-deg", type=float, default=2.0)
    return parser.parse_args()


def clamp_vectors(vectors: np.ndarray, maximum: float) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    scale = np.minimum(1.0, maximum / np.maximum(lengths, 1e-12))
    return vectors * scale[:, None]


def smooth_quaternions(matrices: np.ndarray, window: int, polyorder: int, max_degrees: float) -> np.ndarray:
    original = Rotation.from_matrix(matrices)
    quaternions = original.as_quat()
    for index in range(1, len(quaternions)):
        if float(quaternions[index] @ quaternions[index - 1]) < 0.0:
            quaternions[index] *= -1.0
    filtered = savgol_filter(quaternions, window, polyorder, axis=0, mode="interp")
    filtered /= np.linalg.norm(filtered, axis=1, keepdims=True)
    target = Rotation.from_quat(filtered)
    result = []
    maximum = np.deg2rad(max_degrees)
    for source_rotation, target_rotation in zip(original, target):
        delta = source_rotation.inv() * target_rotation
        angle = float(delta.magnitude())
        if angle <= maximum:
            result.append(target_rotation.as_matrix())
            continue
        interpolation = Slerp([0.0, 1.0], Rotation.concatenate([source_rotation, target_rotation]))
        result.append(interpolation([maximum / angle])[0].as_matrix())
    return np.asarray(result)


def main() -> None:
    args = parse_args()
    if args.window < 3 or args.window % 2 == 0:
        raise ValueError("--window must be an odd integer of at least 3")
    source = np.load(args.input)
    arrays = {key: source[key].copy() for key in source.files}
    glb = arrays["camera_to_world_glb"].copy()
    positions = glb[:, :3, 3]
    smoothed_positions = savgol_filter(positions, args.window, args.polyorder, axis=0, mode="interp")
    translation_correction = clamp_vectors(smoothed_positions - positions, args.max_translation_m)
    refined_positions = positions + translation_correction
    refined_rotations = smooth_quaternions(
        glb[:, :3, :3], args.window, args.polyorder, args.max_rotation_deg
    )

    refined_glb = glb.copy()
    refined_glb[:, :3, :3] = refined_rotations
    refined_glb[:, :3, 3] = refined_positions
    arrays["camera_to_world_glb"] = refined_glb
    arrays["world_to_camera_glb"] = np.linalg.inv(refined_glb)
    cv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    arrays["camera_to_world_glb_opengl"] = refined_glb @ cv_to_opengl

    source_to_glb = arrays["source_to_glb"]
    scale = float(np.cbrt(np.linalg.det(source_to_glb[:3, :3])))
    world_rotation = source_to_glb[:3, :3] / scale
    world_translation = source_to_glb[:3, 3]
    native = arrays["camera_to_world"].copy()
    native[:, :3, :3] = np.einsum("ij,njk->nik", world_rotation.T, refined_rotations)
    native[:, :3, 3] = (refined_positions - world_translation) @ np.linalg.inv(source_to_glb[:3, :3]).T
    arrays["camera_to_world"] = native
    arrays["world_to_camera"] = np.linalg.inv(native)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    rotation_delta = Rotation.from_matrix(np.einsum("nij,njk->nik", np.transpose(glb[:, :3, :3], (0, 2, 1)), refined_rotations)).magnitude()
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "samples": len(glb),
        "method": "shared-frame Savitzky-Golay SE(3) regularization with bounded corrections",
        "window": args.window,
        "polyorder": args.polyorder,
        "translation_correction_m": {
            "mean": float(np.linalg.norm(translation_correction, axis=1).mean()),
            "max": float(np.linalg.norm(translation_correction, axis=1).max()),
        },
        "rotation_correction_deg": {
            "mean": float(np.rad2deg(rotation_delta).mean()),
            "max": float(np.rad2deg(rotation_delta).max()),
        },
        "six_cubemap_faces_share_one_refined_pose": True,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
