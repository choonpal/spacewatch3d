"""Render the three local LingBot RGB point clouds for the project README.

CPU only; requires numpy and Pillow. Every source point is considered in the
final render. A sample is used only to choose the overview camera. Files remain
read-only. No mesh generation, point filtering, hole filling, or RGB edits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
NAMES = ('chair', 'chair_1fps', 'chairs_tables')
DTYPE = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
BACKGROUND = (23, 30, 43)
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
WIDTH, HEIGHT = 1100, 825


def open_cloud(path):
    with path.open('rb') as handle:
        lines = []
        for _ in range(128):
            line = handle.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError(f'Invalid PLY header: {path}')
        offset = handle.tell()
    assert lines[0] == 'ply' and lines[1] == 'format binary_little_endian 1.0'
    properties = [line for line in lines if line.startswith('property ')]
    assert properties == ['property float x', 'property float y', 'property float z',
                          'property uchar red', 'property uchar green', 'property uchar blue']
    elements = [line for line in lines if line.startswith('element ')]
    assert len(elements) == 1 and elements[0].startswith('element vertex ')
    count = int(elements[0].split()[-1])
    assert path.stat().st_size == offset + count * DTYPE.itemsize
    return np.memmap(path, mode='r', dtype=DTYPE, offset=offset, shape=(count,))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def overview_camera(xyz, azimuth=0, elevation=0):
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    target = np.median(xyz, axis=0)
    low, high = np.percentile(xyz, [.2, 99.8], axis=0)
    radius = np.linalg.norm(high - low) / 2
    azimuth, elevation = np.deg2rad([azimuth, elevation])
    eye = target + radius * 2.8 * np.array([
        np.sin(azimuth) * np.cos(elevation), -np.sin(elevation),
        -np.cos(azimuth) * np.cos(elevation),
    ])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., -1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=1)
    camera_points = (xyz - eye) @ rotation
    projected = camera_points[:, :2] / camera_points[:, 2:]
    low, high = np.percentile(projected, [.1, 99.9], axis=0)
    return dict(eye=eye.tolist(), rotation=rotation.tolist(),
                principal=(.5 * (low + high)).tolist(), span=(high - low).tolist())


def detail_camera(name):
    # Both chair reconstructions use exactly the same camera and image bounds.
    # These are visualization cameras, not calibrated input-camera estimates.
    if name.startswith('chair') and name != 'chairs_tables':
        principal, span = [0., -.045], [1.15, .84]
    else:
        principal, span = [-.38, -.29], [1.6, 1.16]
    return dict(eye=[0., 0., 0.], rotation=np.eye(3).tolist(),
                principal=principal, span=span)


class Raster:
    def __init__(self, camera):
        self.camera = camera
        self.depth = np.full(WIDTH * HEIGHT, np.inf, np.float64)
        self.rgb = np.full((WIDTH * HEIGHT, 3), BACKGROUND, np.uint8)
        self.in_frame = 0

    def add(self, xyz, rgb):
        eye, rotation = np.asarray(self.camera['eye']), np.asarray(self.camera['rotation'])
        projected = (xyz - eye) @ rotation
        front = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-7)
        projected, rgb = projected[front], rgb[front]
        uv = projected[:, :2] / projected[:, 2:]
        span = self.camera['span']
        scale = min((WIDTH - 64) / span[0], (HEIGHT - 64) / span[1])
        uv = (uv - self.camera['principal']) * scale + [WIDTH / 2, HEIGHT / 2]
        inside = ((uv[:, 0] >= -.49) & (uv[:, 0] < WIDTH - .51)
                  & (uv[:, 1] >= -.49) & (uv[:, 1] < HEIGHT - .51))
        pixels = np.rint(uv[inside]).astype(np.int64)
        indices = pixels[:, 1] * WIDTH + pixels[:, 0]
        z, rgb = projected[inside, 2], rgb[inside]
        self.in_frame += len(indices)
        # One-pixel splats with exact nearest-depth occlusion across all chunks.
        np.minimum.at(self.depth, indices, z)
        visible = z == self.depth[indices]
        self.rgb[indices[visible]] = rgb[visible]

    def image(self):
        return Image.fromarray(self.rgb.reshape(HEIGHT, WIDTH, 3), mode='RGB')

    def report(self):
        return {**self.camera, 'points_inside_view': self.in_frame,
                'visible_pixels': int(np.isfinite(self.depth).sum()),
                'resolution': [WIDTH, HEIGHT]}


def text(draw, xy, value, size=22, color='#233044'):
    draw.text(xy, value, font=ImageFont.truetype(FONT_PATH, size), fill=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=REPO.parent)
    args = parser.parse_args()
    assets = REPO / 'docs/assets'
    reports = REPO / 'docs/experiments'
    assets.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    sample_chairs = []
    for name in NAMES[:2]:
        data = open_cloud(args.source_dir / (name + '.ply'))
        sample_chairs.append(data['xyz'][::max(1, len(data) // 160000)].copy())
    shared_overview = overview_camera(np.concatenate(sample_chairs))
    manifest = {
        'source_directory': str(args.source_dir.resolve()),
        'method': 'All finite source vertices, perspective projection, one-pixel points, nearest-depth z-buffer, original RGB.',
        'overview_camera_fit': 'Deterministic strided sample; 0.1/99.9 projection percentiles with margins. Camera framing may crop scene edges.',
        'source_downsampling_for_render': False,
        'reconstructed_metric_scale_known': False,
        'chair_cameras_identical': True,
        'source_files_unchanged': True,
        'files': [],
    }
    details = []
    for name in NAMES:
        started = time.perf_counter()
        path = args.source_dir / (name + '.ply')
        signature = (path.stat().st_size, path.stat().st_mtime_ns)
        digest = sha256(path)
        data = open_cloud(path)
        if name == 'chairs_tables':
            sample = data['xyz'][::max(1, len(data) // 160000)]
            overview = overview_camera(sample, azimuth=30, elevation=15)
        else:
            overview = shared_overview
        rasters = [Raster(overview), Raster(detail_camera(name))]
        minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
        valid_count = 0
        for start in range(0, len(data), 1_000_000):
            block = data[start:start + 1_000_000]
            xyz, rgb = block['xyz'], block['rgb']
            finite = np.isfinite(xyz).all(axis=1)
            xyz, rgb = xyz[finite], rgb[finite]
            valid_count += len(xyz)
            if len(xyz):
                minimum = np.minimum(minimum, xyz.min(axis=0))
                maximum = np.maximum(maximum, xyz.max(axis=0))
            for raster in rasters:
                raster.add(xyz, rgb)
        assert valid_count > 0
        assert (path.stat().st_size, path.stat().st_mtime_ns) == signature
        assert sha256(path) == digest
        preview = Image.new('RGB', (WIDTH * 2 + 72, HEIGHT + 174), '#f6f8fb')
        draw = ImageDraw.Draw(preview)
        text(draw, (24, 17), name + '.ply', size=32)
        text(draw, (24, 66), f'{len(data):,} points  /  {signature[0] / 1e6:.2f} MB  /  original RGB', size=20)
        for i, (raster, label) in enumerate(zip(rasters, ('01  Overview', '02  Close view'))):
            x = 24 + i * (WIDTH + 24)
            text(draw, (x, 107), label, size=19, color='#526079')
            preview.paste(raster.image(), (x, 140))
        text(draw, (24, HEIGHT + 147), 'All source points considered; nearest point per pixel. Views crop edges. No mesh or hole filling.', size=16, color='#526079')
        image_name = f'lingbot-{name.replace("_", "-")}-pointcloud.png'
        preview.save(assets / image_name, optimize=True)
        details.append(rasters[1].image())
        manifest['files'].append({
            'name': path.name, 'sha256': digest, 'bytes': signature[0],
            'vertices': len(data), 'finite_vertices': valid_count,
            'xyz_min': minimum.tolist(), 'xyz_max': maximum.tolist(),
            'image': 'docs/assets/' + image_name,
            'views': dict(zip(('overview', 'close'), [r.report() for r in rasters])),
        })
        print(f'{name}: {len(data):,} points, {time.perf_counter() - started:.1f}s', flush=True)

    panel_width, panel_height = 760, 570
    sheet = Image.new('RGB', (panel_width * 3 + 96, 774), '#f6f8fb')
    draw = ImageDraw.Draw(sheet)
    text(draw, (24, 20), 'LingBot-Map | RGB point cloud results', size=34)
    text(draw, (24, 72), 'Three local PLY files / same close-view camera for the two chair reconstructions', size=21, color='#526079')
    for i, (name, image, record) in enumerate(zip(NAMES, details, manifest['files'])):
        x = 24 + i * (panel_width + 24)
        text(draw, (x, 114), name + '.ply', size=25)
        text(draw, (x, 153), f'{record["vertices"]:,} points', size=21, color='#526079')
        sheet.paste(image.resize((panel_width, panel_height), Image.LANCZOS), (x, 194))
    sheet.save(assets / 'lingbot-three-ply-comparison.png', optimize=True)
    manifest['comparison_image'] = 'docs/assets/lingbot-three-ply-comparison.png'
    (reports / 'lingbot-three-ply-visualization.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Rendered 4 PNGs; source SHA-256 values unchanged.', flush=True)


if __name__ == '__main__':
    main()
