"""CPU-only occupied-voxel components; counts are points, not occupied cells.

XYZ/RGB are copied byte-for-byte in original order. The reconstruction's world
axes and zero origin define the grid; there is no visual alignment or tuning.
"""
from pathlib import Path
import time
import numpy as np
from plyfile import PlyData
from common import digest, save
from slab_components import label_slabs

DTYPE = np.dtype([(n, '<f4') for n in ['x', 'y', 'z']] +
                 [(n, 'u1') for n in ['red', 'green', 'blue']])


def header(n):
    return ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(n) + '\n' +
            ''.join('property float '+k+'\n' for k in ['x', 'y', 'z']) +
            ''.join('property uchar '+k+'\n' for k in ['red', 'green', 'blue']) +
            'end_header\n').encode()


def component_selection(counts, remove_max_points_inclusive):
    keep = counts > remove_max_points_inclusive
    keep[0] = False
    return keep


def filter_cloud(source, target, audit_dir, options, expected_sha256, progress=print):
    start = time.perf_counter()
    source, target, audit_dir = Path(source), Path(target), Path(audit_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() or source.resolve() == target.resolve():
        raise ValueError('Refuse to replace a cloud or filter in place')
    source_hash = digest(source)
    if source_hash != expected_sha256:
        raise ValueError('Unfiltered point cloud hash mismatch')
    v = PlyData.read(source, mmap='r')['vertex'].data
    if v.dtype != DTYPE or not len(v):
        raise ValueError('Expected nonempty float32 XYZ / uint8 RGB PLY')
    n = len(v)
    chunk = options['chunk_points']
    grid = np.lib.format.open_memmap(audit_dir/'voxel_indices.npy', mode='w+', dtype='<i4', shape=(n, 3))
    for begin in range(0, n, chunk):
        block = v[begin:begin+chunk]
        xyz = np.column_stack([block[k] for k in ['x', 'y', 'z']]).astype(np.float64)
        q = np.floor((xyz-np.asarray(options['grid_origin']))/options['voxel_size_normalized'])
        if not np.isfinite(q).all() or (np.abs(q) >= 2**30).any():
            raise ValueError('Nonfinite or excessively large component grid coordinates')
        grid[begin:begin+len(block)] = q.astype(np.int32)
    grid.flush()
    grid_seconds = time.perf_counter()-start
    progress(f'GRID points={n} seconds={grid_seconds:.2f}')
    labels, graph = label_slabs(grid, audit_dir/'component_ids.npy',
                               slab_width=options['slab_width'], max_dense_cells=options['max_dense_cells'],
                               progress=progress)
    counts = np.bincount(labels)
    assert len(counts) == graph['component_count']+1 and counts[0] == 0
    np.save(audit_dir/'component_point_counts.npy', counts)
    selected = component_selection(counts, options['remove_max_points_inclusive'])
    kept = int(counts[selected].sum())
    if not kept:
        raise RuntimeError('Every component would be removed; input and audit retained, no final map exported')
    keep_path = audit_dir/'keep_mask.npy'
    keep = np.lib.format.open_memmap(keep_path, mode='w+', dtype='bool', shape=(n,))
    partial = target.with_suffix('.partial.ply')
    removed = audit_dir/'removed.ply'
    removed_partial = removed.with_suffix('.partial.ply')
    with partial.open('wb') as out, removed_partial.open('wb') as reject:
        out.write(header(kept)); reject.write(header(n-kept))
        for begin in range(0, n, chunk):
            block = v[begin:begin+chunk]
            mask = selected[labels[begin:begin+len(block)]]
            keep[begin:begin+len(block)] = mask
            block[mask].tofile(out); block[~mask].tofile(reject)
    keep.flush()
    partial.replace(target); removed_partial.replace(removed)
    # Independently read both outputs and compare every retained/rejected byte.
    out_v = PlyData.read(target, mmap='r')['vertex'].data
    reject_v = PlyData.read(removed, mmap='r')['vertex'].data
    assert len(out_v) == kept and len(reject_v) == n-kept
    a = b = 0
    for begin in range(0, n, chunk):
        block = v[begin:begin+chunk]; mask = keep[begin:begin+len(block)]
        good, bad = block[mask], block[~mask]
        assert good.tobytes() == out_v[a:a+len(good)].tobytes()
        assert bad.tobytes() == reject_v[b:b+len(bad)].tobytes()
        a += len(good); b += len(bad)
    assert a == kept and b == n-kept and digest(source) == source_hash
    cutoff = options['remove_max_points_inclusive']
    assert counts[selected].min() > cutoff
    rejected_counts = counts[(counts > 0) & ~selected]
    assert not len(rejected_counts) or rejected_counts.max() <= cutoff
    with (audit_dir/'components.csv').open('w') as f:
        f.write('component_id,point_count,kept\n')
        for i in np.argsort(-counts[1:], kind='stable')+1:
            f.write(f'{i},{counts[i]},{int(selected[i])}\n')
    report = dict(source=str(source), source_sha256=source_hash, output=str(target),
                  sha256=digest(target), bytes=target.stat().st_size,
                  removed_file=str(removed), removed_sha256=digest(removed),
                  input_points=n, kept_points=kept, removed_points=n-kept,
                  removed_percent=100*(n-kept)/n, options=options,
                  component_count=graph['component_count'], kept_components=int(selected.sum()),
                  removed_components=int(len(rejected_counts)),
                  smallest_kept_component=int(counts[selected].min()),
                  largest_removed_component=int(rejected_counts.max()) if len(rejected_counts) else 0,
                  components_exactly_at_cutoff=int((counts[1:] == cutoff).sum()),
                  top_components=[dict(id=int(i), points=int(counts[i]), kept=bool(selected[i]))
                                  for i in (np.argsort(-counts[1:], kind='stable')+1)[:25]],
                  graph=graph, grid_seconds=grid_seconds,
                  validation=dict(exact_retained_xyz_rgb=True, exact_removed_xyz_rgb=True,
                                  source_unchanged=True, inclusive_threshold_checked=True,
                                  whole_map_connectivity=True, source_order_preserved=True),
                  seconds=time.perf_counter()-start,
                  limitations=['Grid adjacency, not exact Euclidean-distance connectivity.',
                               'Coordinates are normalized, not metres.',
                               'Small real disconnected objects can be removed.',
                               'Noise connected to a retained component or larger than cutoff remains.'])
    save(audit_dir/'report.json', report)
    return report
