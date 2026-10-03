"""Extract LT-Mem trial inputs from published SpaCeFormer predictions.

This runs no segmentation model and produces no cross-session identities.
The RGB histogram is a feature proxy, not the paper's visual embedding.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


SOURCE_URL = 'https://nvlabs.github.io/SpaCeFormer/assets/pointclouds/scene0462_00.json'
SOURCE_SHA256 = '5a363f8a444d13827e4113f6f1dad1676b90c9cbbca7af4490d1cdfad21f9abb'


def extract(source: Path) -> dict:
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    known_source = source_sha256 == SOURCE_SHA256
    data = json.loads(raw)
    positions, colors = data['positions'], data['colors']
    count = data['num_points']
    if len(positions) != count or len(colors) != count:
        raise ValueError('Point and color counts must match num_points')
    if not (len(data['masks']) == len(data['labels']) == len(data['scores']) == data['num_masks']):
        raise ValueError('Mask, label and score counts must match num_masks')
    for point, color in zip(positions, colors):
        if len(point) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in point):
            raise ValueError('Points must contain three finite coordinates')
        if len(color) != 3 or not all(type(x) is int and 0 <= x <= 255 for x in color):
            raise ValueError('Colors must contain three uint8 channels')
    observations = []
    assigned = set()
    overlapping = set()
    empty_masks = []
    for mask_id, indices in enumerate(data['masks']):
        if not indices:
            empty_masks.append(mask_id)
            continue
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < count for i in indices):
            raise ValueError(f'Mask {mask_id} has duplicate or invalid point indices')
        overlapping.update(assigned.intersection(indices))
        assigned.update(indices)
        points = [positions[i] for i in indices]
        low = [min(p[axis] for p in points) for axis in range(3)]
        high = [max(p[axis] for p in points) for axis in range(3)]
        histogram = [0.0] * 24
        for index in indices:
            for channel in range(3):
                histogram[channel * 8 + colors[index][channel] // 32] += 1 / len(indices)
        score = data['scores'][mask_id]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Mask {mask_id} has an invalid prediction score')
        observations.append({
            'observation_id': f'mask_{mask_id:02d}',
            'source_mask_id': mask_id,
            'class_name': data['labels'][mask_id],
            'prediction_score': score,
            'point_count': len(indices),
            'centroid': [fmean(p[axis] for p in points) for axis in range(3)],
            'bbox_min': low, 'bbox_max': high,
            'volume': math.prod(b - a for a, b in zip(low, high)),
            'volume_method': 'axis_aligned_bounding_box',
            'feature': histogram,
            'feature_method': 'rgb_histogram_8_bins_per_channel_proxy',
        })
    return {
        'format': 'lt_mem_trial_observations_v1',
        'scene_id': data['name'], 'session_id': 'scene0462_00_single_capture',
        'synthetic': False if known_source else None,
        'coordinates': {'frame_id': 'spaceformer_scene0462_00_centered', 'units': 'm', 'up_axis': 'z'},
        'alignment': {'status': 'single_capture', 'method': 'input_frame_preserved'},
        'coverage_complete': False,
        'provenance': {
            'source_url': SOURCE_URL if known_source else None, 'source_sha256': source_sha256,
            'source_matches_task4': known_source,
            'perception': 'published_SpaCeFormer_prediction_not_rerun',
            'official_lt_mem_code': 'not_used_code_TBD',
        },
        'audit': {
            'source_point_count': count, 'source_mask_count': data['num_masks'],
            'observation_count': len(observations), 'empty_mask_ids': empty_masks,
            'unique_masked_points': len(assigned), 'points_in_multiple_masks': len(overlapping),
            'masks_are_independent_predictions': True,
        },
        'observations': observations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Use a new output path: {args.output}')
    result = extract(args.source)
    if not result['provenance']['source_matches_task4']:
        raise ValueError('Source SHA-256 differs from the published Task 4 scene0462_00 input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['audit'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
