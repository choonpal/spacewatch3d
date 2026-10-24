from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.utils.eval_utils import eval_setup


FACE_SPECS = {
    # name: (forward, image-up), expressed in the keyframe's base camera coordinates.
    "front": ([0, 0, 1], [0, -1, 0]),
    "right": ([1, 0, 0], [0, -1, 0]),
    "back": ([0, 0, -1], [0, -1, 0]),
    "left": ([-1, 0, 0], [0, -1, 0]),
    "up": ([0, -1, 0], [0, 0, -1]),
    "down": ([0, 1, 0], [0, 0, 1]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 360-degree equirectangular images from 3DGS at keyframe viewpoints."
    )
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Portable Nerfstudio output root containing experiment/method/timestamp.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional portable Nerfstudio dataset path used to replace paths saved in config.yml.",
    )
    parser.add_argument("--camera-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face-size", type=int, default=512)
    parser.add_argument(
        "--face-fov-deg",
        type=float,
        default=100.0,
        help="Perspective face FOV. Values above 90 degrees provide overlap for seam blending.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        help="Equirectangular width; defaults to four times the cube-face size.",
    )
    parser.add_argument("--save-faces", action="store_true")
    return parser.parse_args()


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("camera position and look-at point must be different")
    return vector / length


def level_camera_to_world_cv(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a level CV camera frame whose +Z points toward the target in yaw only.

    An equirectangular panorama must keep its poles aligned with world up.  Tilting
    the entire frame toward a nearby object rotates the horizon and makes the ERP
    invalid for a conventional panorama viewer.  Target elevation is stored as
    the viewer's initial pitch instead.
    """
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = np.asarray(target - position, dtype=np.float64)
    forward[1] = 0.0
    if np.linalg.norm(forward) < 1e-7:
        raise ValueError("camera and target must differ horizontally for a level panorama")
    forward = normalized(forward)
    right = np.cross(forward, world_up)
    right = normalized(right)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, -world_up, forward))
    transform[:3, 3] = position
    return transform


def target_pitch_deg(position: np.ndarray, target: np.ndarray) -> float:
    direction = normalized(target - position)
    return math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))


def face_basis(forward_values: list[int], up_values: list[int]) -> np.ndarray:
    forward = normalized(np.asarray(forward_values, dtype=np.float64))
    image_up = normalized(np.asarray(up_values, dtype=np.float64))
    down = -image_up
    right = normalized(np.cross(down, forward))
    return np.column_stack((right, down, forward))


def make_face_camera(
    base_c2w_cv: np.ndarray,
    basis: np.ndarray,
    face_size: int,
    face_fov_deg: float,
    device: torch.device,
) -> Cameras:
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = basis
    cv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    face_c2w_opengl = base_c2w_cv @ relative @ cv_to_opengl
    focal = 0.5 * face_size / math.tan(math.radians(face_fov_deg) * 0.5)
    return Cameras(
        camera_to_worlds=torch.from_numpy(face_c2w_opengl[:3, :4]).float()[None].to(device),
        fx=focal,
        fy=focal,
        cx=face_size * 0.5,
        cy=face_size * 0.5,
        width=face_size,
        height=face_size,
        camera_type=CameraType.PERSPECTIVE,
    )


def render_cube_faces(
    pipeline, position: np.ndarray, target: np.ndarray, face_size: int, face_fov_deg: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base_c2w_cv = level_camera_to_world_cv(position, target)
    device = pipeline.device
    images: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, (forward, image_up) in FACE_SPECS.items():
            basis = face_basis(forward, image_up)
            camera = make_face_camera(base_c2w_cv, basis, face_size, face_fov_deg, device)
            outputs = pipeline.model.get_outputs_for_camera(camera)
            rgb = outputs["rgb"].detach().cpu().numpy()
            images[name] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
            bases[name] = basis
    return images, bases


def stitch_equirectangular(
    faces: dict[str, np.ndarray], bases: dict[str, np.ndarray], width: int, face_fov_deg: float
) -> np.ndarray:
    if width <= 0 or width % 2:
        raise ValueError("--output-width must be a positive even number")
    height = width // 2
    face_size = next(iter(faces.values())).shape[0]
    tangent = math.tan(math.radians(face_fov_deg) * 0.5)
    focal = 0.5 * face_size / tangent

    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    longitude = (u + 0.5) / width * (2.0 * math.pi) - math.pi
    latitude_down = (v + 0.5) / height * math.pi - math.pi * 0.5
    cos_lat = np.cos(latitude_down)
    directions = np.stack(
        (np.sin(longitude) * cos_lat, np.sin(latitude_down), np.cos(longitude) * cos_lat),
        axis=-1,
    )

    accumulated = np.zeros((height, width, 3), dtype=np.float64)
    accumulated_weight = np.zeros((height, width), dtype=np.float64)
    for name in FACE_SPECS:
        local = directions @ bases[name]
        z = np.maximum(local[..., 2], 1e-8)
        normalized_x = local[..., 0] / (z * tangent)
        normalized_y = local[..., 1] / (z * tangent)
        edge_distance = 1.0 - np.maximum(np.abs(normalized_x), np.abs(normalized_y))
        valid = (local[..., 2] > 0.0) & (edge_distance > 0.0)
        # The squared taper gives the face center priority while smoothly blending overlap.
        weight = np.where(valid, edge_distance * edge_distance, 0.0)
        map_x = (focal * local[..., 0] / z + face_size * 0.5 - 0.5).astype(np.float32)
        map_y = (focal * local[..., 1] / z + face_size * 0.5 - 0.5).astype(np.float32)
        sampled = cv2.remap(
            faces[name], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        accumulated += sampled.astype(np.float64) * weight[..., None]
        accumulated_weight += weight
    panorama = accumulated / np.maximum(accumulated_weight[..., None], 1e-8)
    return np.clip(panorama + 0.5, 0, 255).astype(np.uint8)


def load_keyframe(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "camera_position_glb" not in data:
        raise KeyError(f"{path} has no camera_position_glb")
    position = np.asarray(data["camera_position_glb"], dtype=np.float64)
    target_values = data.get("look_at_glb", data.get("object_center_glb"))
    if target_values is None:
        raise KeyError(f"{path} has neither look_at_glb nor object_center_glb")
    target = np.asarray(target_values, dtype=np.float64)
    if position.shape != (3,) or target.shape != (3,):
        raise ValueError(f"invalid camera vectors in {path}")
    return position, target, data


def main() -> None:
    args = parse_args()
    output_width = args.output_width or args.face_size * 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faces_root = args.output_dir / "cubemap_faces"
    if args.save_faces:
        faces_root.mkdir(parents=True, exist_ok=True)

    def make_portable(config):
        if args.checkpoint_root:
            config.output_dir = args.checkpoint_root.resolve()
        if args.data_dir:
            data_dir = args.data_dir.resolve()
            config.data = data_dir
            config.pipeline.datamanager.data = data_dir
            config.pipeline.datamanager.dataparser.data = data_dir
        return config

    _, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        test_mode="inference",
        update_config_callback=make_portable,
    )
    results = []
    for camera_path in args.camera_json:
        position, target, source = load_keyframe(camera_path)
        faces, bases = render_cube_faces(
            pipeline, position, target, args.face_size, args.face_fov_deg
        )
        panorama = stitch_equirectangular(faces, bases, output_width, args.face_fov_deg)

        stem = camera_path.stem.removesuffix("_camera")
        panorama_path = args.output_dir / f"{stem}_360.png"
        if not cv2.imwrite(str(panorama_path), cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {panorama_path}")

        face_paths: dict[str, str] = {}
        if args.save_faces:
            face_dir = faces_root / stem
            face_dir.mkdir(parents=True, exist_ok=True)
            for name, image in faces.items():
                face_path = face_dir / f"{name}.png"
                if not cv2.imwrite(str(face_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"failed to write {face_path}")
                face_paths[name] = str(face_path.resolve())

        metadata = {
            "source_camera_json": str(camera_path.resolve()),
            "source_camera": source,
            "camera_position_glb": position.tolist(),
            "look_at_glb": target.tolist(),
            "orientation": "level ERP: world +Y is up; look-at yaw is image center",
            "initial_view": {"yaw": 0.0, "pitch": target_pitch_deg(position, target), "fov": 75.0},
            "face_size": args.face_size,
            "face_fov_deg": args.face_fov_deg,
            "equirectangular_size": [output_width, output_width // 2],
            "model_config": str(args.load_config.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_step": int(step),
            "panorama": str(panorama_path.resolve()),
            "cubemap_faces": face_paths,
        }
        metadata_path = args.output_dir / f"{stem}_360.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(metadata)
        print(f"rendered {panorama_path}")

    summary_path = args.output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
