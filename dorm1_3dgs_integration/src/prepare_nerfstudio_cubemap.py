from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


FACE_SPECS = {
    # name: (forward, image-up), expressed in the source ERP camera coordinates
    "front": ([0, 0, 1], [0, -1, 0]),
    "right": ([1, 0, 0], [0, -1, 0]),
    "back": ([0, 0, -1], [0, -1, 0]),
    "left": ([-1, 0, 0], [0, -1, 0]),
    "up": ([0, -1, 0], [0, 0, -1]),
    "down": ([0, 1, 0], [0, 0, 1]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dorm1 equirectangular frames and poses to a Nerfstudio pinhole dataset."
    )
    parser.add_argument("--camera-package", type=Path, required=True)
    parser.add_argument(
        "--cameras-npz",
        type=Path,
        help="Optional refined cameras.npz; defaults to the camera package file.",
    )
    parser.add_argument("--mesh-dataset", type=Path, required=True)
    parser.add_argument(
        "--erp-images",
        type=Path,
        help="Optional directory of higher-resolution frame_XXXXX JPG/PNG images",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-size", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--point-count", type=int, default=1_000_000)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def face_basis(forward_values: list[int], up_values: list[int]) -> np.ndarray:
    forward = np.asarray(forward_values, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    image_up = np.asarray(up_values, dtype=np.float64)
    image_up /= np.linalg.norm(image_up)
    down = -image_up
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.column_stack((right, down, forward))


def make_erp_maps(
    face_size: int,
    fov_deg: float,
    erp_width: int,
    erp_height: int,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    focal = 0.5 * face_size / math.tan(math.radians(fov_deg) * 0.5)
    u, v = np.meshgrid(np.arange(face_size, dtype=np.float64), np.arange(face_size, dtype=np.float64))
    rays_face = np.stack(
        ((u + 0.5 - face_size * 0.5) / focal, (v + 0.5 - face_size * 0.5) / focal, np.ones_like(u)),
        axis=-1,
    )
    rays_base = rays_face @ basis.T
    length = np.linalg.norm(rays_base, axis=-1)
    longitude = np.arctan2(rays_base[..., 0], rays_base[..., 2])
    latitude_down = np.arcsin(np.clip(rays_base[..., 1] / length, -1.0, 1.0))
    map_x = erp_width * (longitude / (2.0 * math.pi) + 0.5) - 0.5
    map_y = erp_height * (latitude_down / math.pi + 0.5) - 0.5
    return map_x.astype(np.float32), map_y.astype(np.float32)


def write_binary_point_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    records = np.empty(len(xyz), dtype=dtype)
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    records["red"], records["green"], records["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
        resolved = output.resolve()
        if output.name in {"", ".", ".."} or resolved == resolved.anchor:
            raise ValueError(f"refusing to remove unsafe output path: {resolved}")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)
    (output / "masks").mkdir(parents=True)


def find_frame(images_dir: Path, sample_index: int) -> Path:
    candidates = sorted(images_dir.glob(f"frame_{sample_index:05d}.*"))
    candidates = [path for path in candidates if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one image for sample {sample_index} in {images_dir}, got {candidates}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    if not (1.0 < args.fov_deg < 179.0):
        raise ValueError("--fov-deg must be between 1 and 179 degrees")
    prepare_output(args.output, args.overwrite)

    camera_npz = args.cameras_npz if args.cameras_npz is not None else args.camera_package / "cameras.npz"
    images_dir = args.erp_images if args.erp_images is not None else args.camera_package / "images_1036x518"
    exclusion_dir = args.camera_package / "exclusion_masks_1036x518"
    mesh_npz = args.mesh_dataset / "mesh" / "dorm1_mesh_labels.npz"
    for path in (camera_npz, images_dir, exclusion_dir, mesh_npz):
        if not path.exists():
            raise FileNotFoundError(path)

    cameras = np.load(camera_npz)
    base_c2w_cv = cameras["camera_to_world_glb"]
    sample_indices = cameras["sample_index"]
    source_indices = cameras["source_frame_index"]
    timestamps = cameras["timestamp_s"]
    sample_count = len(sample_indices) if args.limit_samples is None else min(args.limit_samples, len(sample_indices))

    first_image = cv2.imread(str(find_frame(images_dir, int(sample_indices[0]))), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError("could not read the first ERP image")
    erp_height, erp_width = first_image.shape[:2]

    face_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (forward, image_up) in FACE_SPECS.items():
        basis = face_basis(forward, image_up)
        map_x, map_y = make_erp_maps(args.face_size, args.fov_deg, erp_width, erp_height, basis)
        face_data[name] = (basis, map_x, map_y)

    focal = 0.5 * args.face_size / math.tan(math.radians(args.fov_deg) * 0.5)
    cv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    frames: list[dict] = []
    face_counts = {name: 0 for name in FACE_SPECS}

    for row in range(sample_count):
        sample_index = int(sample_indices[row])
        erp = cv2.imread(str(find_frame(images_dir, sample_index)), cv2.IMREAD_COLOR)
        exclusion = cv2.imread(
            str(exclusion_dir / f"frame_{sample_index:05d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if erp is None or exclusion is None:
            raise RuntimeError(f"missing RGB or mask for sample {sample_index}")
        if erp.shape[:2] != (erp_height, erp_width):
            raise ValueError(f"inconsistent ERP dimensions at sample {sample_index}")
        if exclusion.shape != (erp_height, erp_width):
            exclusion = cv2.resize(exclusion, (erp_width, erp_height), interpolation=cv2.INTER_NEAREST)

        for face_name, (basis, map_x, map_y) in face_data.items():
            rgb_face = cv2.remap(
                erp, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
            )
            exclusion_face = cv2.remap(
                exclusion, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP
            )
            # Nerfstudio mask convention: white=use, black=ignore.
            valid_mask = np.where(exclusion_face > 127, 0, 255).astype(np.uint8)

            filename = f"frame_{sample_index:05d}_{face_name}"
            image_rel = Path("images") / f"{filename}.jpg"
            mask_rel = Path("masks") / f"{filename}.png"
            if not cv2.imwrite(
                str(args.output / image_rel), rgb_face, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            ):
                raise RuntimeError(f"failed to write {image_rel}")
            if not cv2.imwrite(str(args.output / mask_rel), valid_mask):
                raise RuntimeError(f"failed to write {mask_rel}")

            relative = np.eye(4, dtype=np.float64)
            relative[:3, :3] = basis
            face_c2w_opengl = base_c2w_cv[row] @ relative @ cv_to_opengl
            frames.append(
                {
                    "file_path": image_rel.as_posix(),
                    "mask_path": mask_rel.as_posix(),
                    "transform_matrix": face_c2w_opengl.tolist(),
                    "sample_index": sample_index,
                    "source_frame_index": int(source_indices[row]),
                    "timestamp_s": float(timestamps[row]),
                    "cubemap_face": face_name,
                }
            )
            face_counts[face_name] += 1

    mesh = np.load(mesh_npz)
    source_xyz = mesh["xyz"]
    source_rgb = mesh["rgb"]
    source_to_glb = mesh["source_to_glb"]
    stride = max(1, math.ceil(len(source_xyz) / args.point_count))
    xyz = source_xyz[::stride].astype(np.float64)
    xyz = xyz @ source_to_glb[:3, :3].T + source_to_glb[:3, 3]
    rgb = source_rgb[::stride]
    finite = np.isfinite(xyz).all(axis=1)
    write_binary_point_cloud(args.output / "sparse_pc.ply", xyz[finite].astype(np.float32), rgb[finite])

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": focal,
        "fl_y": focal,
        "cx": args.face_size * 0.5,
        "cy": args.face_size * 0.5,
        "w": args.face_size,
        "h": args.face_size,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    (args.output / "transforms.json").write_text(
        json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "samples": sample_count,
        "perspective_frames": len(frames),
        "faces_per_sample": len(FACE_SPECS),
        "face_counts": face_counts,
        "face_size": args.face_size,
        "horizontal_and_vertical_fov_deg": args.fov_deg,
        "focal_length_px": focal,
        "source_erp_size": [erp_width, erp_height],
        "source_erp_images": str(images_dir.resolve()),
        "exclusion_masks_resized_to_source_erp": (erp_width, erp_height) != (1036, 518),
        "point_cloud_points": int(finite.sum()),
        "point_cloud_stride": stride,
        "camera_centers_preserved": True,
        "mask_convention": "white=use, black=ignore",
    }
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
