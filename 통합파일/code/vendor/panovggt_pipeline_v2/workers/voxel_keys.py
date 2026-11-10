"""Exact lexicographic unique rows for signed int64 XYZ voxel keys."""
import numpy as np


def unique_voxel_keys(keys):
    """Return the same keys/inverse as np.unique(keys, axis=0).

    Sorting numeric columns avoids NumPy's slower structured-record sort.
    Inverse indices map the ORIGINAL point order to lexicographic key order,
    preserving downstream bincount summation order and PLY row identities.
    """
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError('Voxel keys must have shape (N, 3)')
    if keys.dtype != np.dtype(np.int64):
        raise TypeError('Voxel keys must be signed int64')
    if not len(keys):
        return keys.copy(), np.empty(0, dtype=np.intp)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    first = np.empty(len(keys), dtype=bool)
    first[0] = True
    first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(keys), dtype=np.intp)
    inverse[order] = np.cumsum(first, dtype=np.intp) - 1
    return ordered[first], inverse
