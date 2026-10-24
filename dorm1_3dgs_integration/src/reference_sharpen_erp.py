from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer only geometrically aligned high-frequency luminance detail from source ERP frames."
    )
    parser.add_argument("--render-summary", type=Path, required=True)
    parser.add_argument("--original-cameras", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--detail-limit", type=float, default=24.0)
    parser.add_argument("--agreement-sigma", type=float, default=32.0)
    return parser.parse_args()


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def level_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = target - position
    forward[1] = 0.0
    forward = normalized(forward)
    right = normalized(np.cross(forward, world_up))
    return np.column_stack((right, -world_up, forward))


def warp_reference(
    reference: np.ndarray, original_rotation: np.ndarray, destination_rotation: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    width, height = size
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    longitude = (u + 0.5) / width * (2.0 * math.pi) - math.pi
    latitude_down = (v + 0.5) / height * math.pi - math.pi * 0.5
    cos_lat = np.cos(latitude_down)
    rays_destination = np.stack(
        (np.sin(longitude) * cos_lat, np.sin(latitude_down), np.cos(longitude) * cos_lat), axis=-1
    )
    relative = original_rotation.T @ destination_rotation
    rays_original = rays_destination @ relative.T
    original_longitude = np.arctan2(rays_original[..., 0], rays_original[..., 2])
    original_latitude = np.arcsin(np.clip(rays_original[..., 1], -1.0, 1.0))
    source_height, source_width = reference.shape[:2]
    map_x = ((original_longitude + math.pi) / (2.0 * math.pi) * source_width - 0.5) % source_width
    map_y = np.clip(
        (original_latitude + math.pi * 0.5) / math.pi * source_height - 0.5,
        0.0,
        source_height - 1.0,
    )
    return cv2.remap(
        reference,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP,
    )


def transfer_detail(
    base: np.ndarray,
    reference: np.ndarray,
    strength: float,
    detail_limit: float,
    agreement_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_l = base_lab[..., 0]
    ref_l = ref_lab[..., 0]
    base_low = cv2.GaussianBlur(base_l, (0, 0), 2.0)
    ref_low = cv2.GaussianBlur(ref_l, (0, 0), 2.0)
    agreement = np.exp(-np.square((base_low - ref_low) / agreement_sigma))
    # Suppress detail transfer close to the ERP poles and in severely mismatched/parallax regions.
    rows = np.linspace(-1.0, 1.0, base.shape[0], dtype=np.float32)
    latitude_weight = np.power(np.clip(np.cos(rows * math.pi * 0.5), 0.0, 1.0), 0.35)[:, None]
    confidence = agreement * latitude_weight
    detail = ref_l - cv2.GaussianBlur(ref_l, (0, 0), 1.15)
    detail = np.clip(detail, -detail_limit, detail_limit)
    base_lab[..., 0] = np.clip(base_l + strength * confidence * detail, 0.0, 255.0)
    output = cv2.cvtColor(base_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return output, confidence


def main() -> None:
    args = parse_args()
    records = json.loads(args.render_summary.read_text(encoding="utf-8"))
    archive = np.load(args.original_cameras)
    original_lookup = {
        int(sample): matrix
        for sample, matrix in zip(archive["sample_index"], archive["camera_to_world_glb"])
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    for record in records:
        source = record["source_camera"]
        sample_index = int(source["sample_index"])
        base = cv2.imread(record["panorama"], cv2.IMREAD_COLOR)
        reference_path = args.frames_dir / f"frame_{sample_index:05d}.jpg"
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if base is None or reference is None:
            raise FileNotFoundError(f"missing rendered or reference image for sample {sample_index}")

        position = np.asarray(record["camera_position_glb"], dtype=np.float64)
        target = np.asarray(record["look_at_glb"], dtype=np.float64)
        destination_rotation = level_rotation(position, target)
        original_rotation = np.asarray(original_lookup[sample_index], dtype=np.float64)[:3, :3]
        aligned_reference = warp_reference(
            reference, original_rotation, destination_rotation, (base.shape[1], base.shape[0])
        )
        sharpened, confidence = transfer_detail(
            base, aligned_reference, args.strength, args.detail_limit, args.agreement_sigma
        )
        output_path = args.output_dir / Path(record["panorama"]).name
        if not cv2.imwrite(str(output_path), sharpened, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to write {output_path}")
        updated = dict(record)
        updated.update(
            {
                "panorama": str(output_path.resolve()),
                "reference_sharpening": {
                    "method": "spherical pose alignment + confidence-gated luminance detail transfer",
                    "reference": str(reference_path.resolve()),
                    "strength": args.strength,
                    "detail_limit_lab": args.detail_limit,
                    "mean_confidence": float(np.mean(confidence)),
                },
            }
        )
        output_records.append(updated)
        print(f"sharpened {output_path} (confidence {np.mean(confidence):.3f})")

    summary_path = args.output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(output_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
