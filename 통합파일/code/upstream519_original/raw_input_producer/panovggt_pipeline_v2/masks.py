"""Deterministic, explicit mask correction. No visual judgement happens at runtime."""
import numpy as np
import cv2


def correct_mask(base, frame_index, rules):
    w, h = rules['coordinate_size']
    if base.shape != (h, w):
        raise ValueError('Base mask resolution mismatch')
    mask = (base > 0).astype(np.uint8) * 255
    kx, ky = rules['periodic_dilation_kernel']
    pad = kx // 2
    padded = np.pad(mask, ((0, 0), (pad, pad)), mode='wrap')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    corrected = cv2.dilate(padded, kernel)[:, pad:pad+w].copy()
    for polygon in rules['polygons'].get(str(frame_index), []):
        points = np.asarray(polygon)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError('Each exclusion polygon requires at least three (x,y) vertices')
        if np.any(points < 0) or np.any(points > [w, h]):
            raise ValueError('Polygon exceeds its reference image')
        cv2.fillPoly(corrected, [points.astype(np.int32)], 255)
    if (corrected > 0).mean() >= rules['max_fraction']:
        raise ValueError(f'Mask exceeds configured area bound at frame {frame_index}')
    return corrected
