"""CPU helpers. A detector box is never substituted for a failed mask."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def region_prompt(box, size):
    w, h = size
    values = np.floor(np.asarray(box) / np.array([w, h, w, h]) * 1000)
    values = np.clip(values, 0, 999).astype(int)
    return '<REGION_TO_SEGMENTATION>' + ''.join(f'<loc_{v}>' for v in values)


def rasterize(parsed, size):
    """Union all polygons of a single prompted object, preserving empty output."""
    w, h = size
    canvas = Image.new('1', (w, h))
    draw = ImageDraw.Draw(canvas)
    valid, invalid = 0, 0
    groups = parsed.get('polygons', []) if isinstance(parsed, dict) else []
    for group in groups:
        for poly in group:
            a = np.asarray(poly, dtype=float)
            if a.size < 6 or a.size % 2 or not np.isfinite(a).all():
                invalid += 1
                continue
            a = a.reshape(-1, 2)
            area = abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1))) / 2
            if len(np.unique(a, axis=0)) < 3 or area < 1 or (a < 0).any() or (a[:, 0] > w).any() or (a[:, 1] > h).any():
                invalid += 1
                continue
            draw.polygon([tuple(p) for p in a], fill=1)
            valid += 1
    mask = np.asarray(canvas, dtype=bool)
    return mask, dict(valid_polygons=valid, invalid_polygons=invalid, pixels=int(mask.sum()),
                      usable=bool(valid and mask.sum() >= 16),
                      rejection_reason=None if valid and mask.sum() >= 16 else 'no_usable_polygon')


def mask_box_metrics(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = box
    yy, xx = np.mgrid[:h, :w]
    inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    return dict(outside_box_fraction=float((mask & ~inside).sum() / max(1, mask.sum())),
                border_touch=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
