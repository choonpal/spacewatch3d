"""Small compatible fallback used when the native fpsample wheel cannot be built."""

from __future__ import annotations

import numpy as np


def bucket_fps_kdline_sampling(points, n_samples: int, h: int = 3):
    """Return greedy farthest-point indices.

    Nerfstudio uses this only for optional camera-order sampling. Camera arrays are
    small enough that this portable O(N*K) implementation is sufficient.
    """
    del h
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("points must be a non-empty 2D array")
    count = min(int(n_samples), len(data))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.random.randint(len(data)))
    min_squared_distance = np.sum((data - data[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_squared_distance))
        squared_distance = np.sum((data - data[selected[index]]) ** 2, axis=1)
        np.minimum(min_squared_distance, squared_distance, out=min_squared_distance)
    return selected
