from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated dorm1 Nerfstudio dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--camera-package", type=Path, required=True)
    parser.add_argument("--cameras-npz", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    transforms = json.loads((args.dataset / "transforms.json").read_text(encoding="utf-8"))
    source = np.load(args.cameras_npz if args.cameras_npz is not None else args.camera_package / "cameras.npz")
    frames = transforms["frames"]
    matrices = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    rotations = matrices[:, :3, :3]

    missing: list[str] = []
    image_shapes: set[tuple[int, int]] = set()
    mask_shapes: set[tuple[int, int]] = set()
    mask_values: set[int] = set()
    excluded_fractions: list[float] = []
    grouped_centers: dict[int, list[np.ndarray]] = {}
    for frame, matrix in zip(frames, matrices):
        image_path = args.dataset / frame["file_path"]
        mask_path = args.dataset / frame["mask_path"]
        if not image_path.exists():
            missing.append(str(image_path))
            continue
        if not mask_path.exists():
            missing.append(str(mask_path))
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            missing.append(f"unreadable:{image_path}:{mask_path}")
            continue
        image_shapes.add(image.shape[:2])
        mask_shapes.add(mask.shape)
        mask_values.update(int(value) for value in np.unique(mask))
        excluded_fractions.append(float((mask == 0).mean()))
        grouped_centers.setdefault(int(frame["sample_index"]), []).append(matrix[:3, 3])

    reference_centers = source["camera_to_world_glb"][:, :3, 3]
    center_errors = []
    face_counts = []
    for sample_index, reference in enumerate(reference_centers):
        centers = grouped_centers.get(sample_index, [])
        face_counts.append(len(centers))
        center_errors.extend(float(np.max(np.abs(center - reference))) for center in centers)

    report = {
        "status": "passed",
        "frames": len(frames),
        "missing_or_unreadable": missing,
        "image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
        "mask_values": sorted(mask_values),
        "mean_excluded_fraction": float(np.mean(excluded_fractions)),
        "faces_per_sample_min": min(face_counts),
        "faces_per_sample_max": max(face_counts),
        "max_camera_center_error": max(center_errors, default=float("inf")),
        "max_rotation_orthogonality_error": float(
            np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
        ),
        "rotation_determinant_min": float(np.linalg.det(rotations).min()),
        "rotation_determinant_max": float(np.linalg.det(rotations).max()),
        "point_cloud_exists": (args.dataset / transforms.get("ply_file_path", "")).is_file(),
    }
    checks = [
        not missing,
        image_shapes == {(transforms["h"], transforms["w"])},
        mask_shapes == {(transforms["h"], transforms["w"])},
        mask_values.issubset({0, 255}),
        min(face_counts) == max(face_counts) == 6,
        report["max_camera_center_error"] < 1e-9,
        report["max_rotation_orthogonality_error"] < 1e-4,
        report["rotation_determinant_min"] > 0.999,
        report["rotation_determinant_max"] < 1.001,
        report["point_cloud_exists"],
    ]
    report["status"] = "passed" if all(checks) else "failed"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
