"""CPU-only geometry and deterministic confidence-weighted semantic voting."""
import numpy as np

UNKNOWN = 255
RGB_DTYPE = np.dtype([(k, '<f4') for k in ('x', 'y', 'z')] + [(k, 'u1') for k in ('red', 'green', 'blue')])
VOTE_DTYPE = np.dtype([('index', '<u4'), ('label', 'u1'), ('weight', '<f4')])
OUTPUT_DTYPE = np.dtype(RGB_DTYPE.descr + [('semantic_id', 'u1'), ('vote_agreement', '<f4'),
    ('winning_mean_confidence', '<f4'), ('observations', '<u2'), ('source_map_index', '<u4')])

def select_frames(manifest, start, end):
    if not 0 <= start <= end:
        raise ValueError('Require 0 <= start_seconds <= end_seconds')
    selected = [i for i, f in enumerate(manifest['frames']) if start <= f['timestamp_s'] <= end]
    if not selected:
        raise ValueError('No generator frames in the requested time interval')
    return selected

def transform_points(points, raw_pose, optimized_pose, scale):
    # Identical float64 arithmetic and float32 rounding to generator v1 export.py.
    rot = optimized_pose[:3, :3] @ raw_pose[:3, :3].T
    translation = optimized_pose[:3, 3] - scale * rot @ raw_pose[:3, 3]
    return (scale * (points.astype(np.float64) @ rot.T) + translation).astype(np.float32)

def voxel_keys(xyz, voxel):
    return np.floor(xyz / voxel).astype(np.int64)

def buckets(keys, count):
    return ((keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)) % count

def structured_keys(keys):
    return np.ascontiguousarray(keys, dtype='<i8').view([('x', '<i8'), ('y', '<i8'), ('z', '<i8')]).reshape(-1)

def exact_lookup(sorted_keys, keys):
    base, query = structured_keys(sorted_keys), structured_keys(keys)
    loc = np.searchsorted(base, query)
    valid = loc < len(base)
    valid[valid] &= base[loc[valid]] == query[valid]
    if not valid.all():
        raise ValueError(f'{int((~valid).sum())} source voxels have no exact map correspondence')
    return loc

def resize_labels_to_source(labels, confidence, height, width):
    # Nearest pixel-center sampling, with no seam shift or perspective conversion.
    y = np.minimum(((np.arange(height) + .5) * labels.shape[0] / height).astype(int), labels.shape[0] - 1)
    x = np.minimum(((np.arange(width) + .5) * labels.shape[1] / width).astype(int), labels.shape[1] - 1)
    return labels[y[:, None], x], confidence[y[:, None], x].astype(np.float32)

def weighted_winners(indices, labels, weights):
    """Return one winner per index; ties deterministically select the lower class ID."""
    if not len(indices):
        return (np.empty(0, np.uint32), np.empty(0, np.uint8),
                np.empty(0, np.float64), np.empty(0, np.float64))
    codes = indices.astype(np.uint64) * 256 + labels.astype(np.uint64)
    order = np.argsort(codes, kind='stable')
    codes = codes[order]
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    totals = np.add.reduceat(weights[order].astype(np.float64), starts)
    group_ids = (codes[starts] // 256).astype(np.uint32)
    classes = (codes[starts] % 256).astype(np.uint8)
    ids, first, inverse = np.unique(group_ids, return_index=True, return_inverse=True)
    maximum = np.zeros(len(ids), dtype=np.float64)
    np.maximum.at(maximum, inverse, totals)
    candidate = np.where(totals == maximum[inverse], np.arange(len(totals)), len(totals))
    best = np.full(len(ids), len(totals), dtype=np.int64)
    np.minimum.at(best, inverse, candidate)
    total = np.add.reduceat(totals, first)
    winner = classes[best].copy()
    winner[maximum == 0] = UNKNOWN
    return ids, winner, maximum, total

def frame_votes(indices, labels, probabilities, threshold):
    labels = labels.copy()
    weights = probabilities.astype(np.float32).copy()
    rejected = weights < threshold
    weights[rejected] = 0
    labels[rejected] = UNKNOWN
    ids, winner, maximum, _ = weighted_winners(indices, labels, weights)
    unique_ids, count = np.unique(indices, return_counts=True)
    assert np.array_equal(ids, unique_ids)
    result = np.empty(len(ids), VOTE_DTYPE)
    result['index'], result['label'], result['weight'] = ids, winner, maximum / count
    return result

def fuse_votes(records, min_agreement, min_mean_confidence, min_observations):
    ids, winner, maximum, total = weighted_winners(records['index'], records['label'], records['weight'])
    unique_ids, observations = np.unique(records['index'], return_counts=True)
    assert np.array_equal(ids, unique_ids)
    agreement = np.divide(maximum, total, out=np.zeros_like(maximum), where=total > 0)
    mean_confidence = maximum / observations
    winner[(agreement < min_agreement) | (mean_confidence < min_mean_confidence) |
           (observations < min_observations)] = UNKNOWN
    return ids, winner, agreement.astype(np.float32), mean_confidence.astype(np.float32), observations

def ply_header(count, dtype):
    types = {'f4': 'float', 'u1': 'uchar', 'u2': 'ushort', 'u4': 'uint'}
    props = ''.join(f'property {types[np.dtype(t).str[1:]]} {name}\n' for name, t in dtype.descr)
    return f'ply\nformat binary_little_endian 1.0\nelement vertex {count}\n{props}end_header\n'.encode()
