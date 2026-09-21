"""Create three README views of left_hall.ply using its original XYZ/RGB.

Run from the repository root with Python, NumPy, and Pillow installed:
    OPENBLAS_NUM_THREADS=4 python3 scripts/render_left_hall_previews.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import render_lingbot_ply_previews as renderer


def interior_camera():
    eye = np.array([2., -.1, 2.])
    target = np.array([7., -.3, 4.8])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., -1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return {
        'eye': eye.tolist(),
        'rotation': np.stack([right, down, forward], axis=1).tolist(),
        'principal': [0., 0.],
        'span': [1.4, 1.],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=renderer.REPO.parent / 'left_hall.ply')
    args = parser.parse_args()
    renderer.WIDTH, renderer.HEIGHT = 1500, 1000
    source = args.source.resolve()
    initial_stat = source.stat()
    digest = renderer.sha256(source)
    data = renderer.open_cloud(source)
    sample = data['xyz'][::max(1, len(data) // 200000)]
    views = [
        ('overview', '01  Oblique overview', renderer.overview_camera(sample, -40, 15)),
        ('elevated', '02  Elevated overview', renderer.overview_camera(sample, 0, 65)),
        ('interior', '03  Interior perspective', interior_camera()),
    ]
    rasters = [renderer.Raster(camera) for _, _, camera in views]
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    finite_count = 0
    for start in range(0, len(data), 1_000_000):
        block = data[start:start + 1_000_000]
        finite = np.isfinite(block['xyz']).all(axis=1)
        xyz, rgb = block['xyz'][finite], block['rgb'][finite]
        finite_count += len(xyz)
        if len(xyz):
            minimum = np.minimum(minimum, xyz.min(axis=0))
            maximum = np.maximum(maximum, xyz.max(axis=0))
        for raster in rasters:
            raster.add(xyz, rgb)
    assert finite_count > 0
    assert source.stat().st_size == initial_stat.st_size
    assert source.stat().st_mtime_ns == initial_stat.st_mtime_ns
    assert renderer.sha256(source) == digest

    assets = renderer.REPO / 'docs/assets'
    reports = renderer.REPO / 'docs/experiments'
    assets.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = {
        'source': str(source),
        'sha256': digest,
        'bytes': initial_stat.st_size,
        'vertices': len(data),
        'finite_vertices': finite_count,
        'xyz_min': minimum.tolist(),
        'xyz_max': maximum.tolist(),
        'source_file_unchanged': True,
        'source_downsampling_for_render': False,
        'method': 'All finite source vertices; perspective projection; nearest depth per pixel; original RGB. No mesh, hole filling, or RGB edits.',
        'camera_fit': 'Deterministic strided sample for overview framing only. 0.1/99.9 projection percentiles with margins. Interior view is manually positioned in reconstruction coordinates.',
        'coordinate_units': 'Reconstruction units; not calibrated to metres.',
        'views': [],
    }
    for (slug, title, _), raster in zip(views, rasters):
        canvas = Image.new('RGB', (renderer.WIDTH + 48, renderer.HEIGHT + 164), '#f6f8fb')
        draw = ImageDraw.Draw(canvas)
        renderer.text(draw, (24, 17), f'left_hall.ply  |  {title}', size=30)
        renderer.text(draw, (24, 64), f'{len(data):,} points  /  {initial_stat.st_size / 1e6:.2f} MB  /  original RGB', size=21)
        canvas.paste(raster.image(), (24, 110))
        renderer.text(draw, (24, renderer.HEIGHT + 125),
                      'All source points considered; nearest point per pixel. Camera framing may crop scene edges.', size=18, color='#526079')
        relative = f'docs/assets/lingbot-left-hall-{slug}.png'
        canvas.save(renderer.REPO / relative, optimize=True)
        report['views'].append({'name': slug, 'image': relative, **raster.report()})
        print(relative, flush=True)
    (reports / 'lingbot-left-hall-visualization.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {finite_count:,} finite points; source SHA-256 unchanged.', flush=True)


if __name__ == '__main__':
    main()
