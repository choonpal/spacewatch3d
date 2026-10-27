"""Fuse all supplied camera observations before issuing stable 3D candidate IDs."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def fuse(samples, voxel, minimum):
    if not samples:
        return [], np.empty((0, 3), 'f4'), np.empty(0, 'i4')
    points = np.concatenate([s['points'] for s in samples])
    phases = np.concatenate([np.full(len(s['points']), s['phase']) for s in samples])
    views = np.concatenate([np.full(len(s['points']), s['view']) for s in samples])
    cells, inverse = np.unique(np.floor(points / voxel).astype('i8'), axis=0, return_inverse=True)
    pairs = cKDTree(cells).query_pairs(1.8, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, groups = connected_components(graph, directed=False)
    labels = groups[inverse]
    candidates = []; out_points = []; out_labels = []
    for group in np.unique(labels):
        mask = labels == group
        if len(np.unique(inverse[mask])) < minimum:
            continue
        idx = len(candidates)
        q = points[mask]
        bounds = np.stack([q.min(0)-voxel, q.max(0)+voxel])
        candidates.append({'id': f'C{idx+1:05d}', 'bounds': bounds.tolist(),
                           'center': bounds.mean(0).tolist(), 'type': 'unclassified',
                           'observations': sorted(np.unique(views[mask]).astype(int).tolist()),
                           'phase_support_pixels': [int((phases[mask] == p).sum()) for p in [0, 1]],
                           'reason': 'SimSaC surface-change candidate; no object-identity/add-remove claim'})
        out_points.append(q); out_labels.append(np.full(len(q), idx, 'i4'))
    return (candidates, np.concatenate(out_points).astype('f4') if out_points else np.empty((0, 3), 'f4'),
            np.concatenate(out_labels) if out_labels else np.empty(0, 'i4'))
