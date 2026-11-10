"""Exact generator v1/v2 provenance recovery and pixel-to-original-map projection."""
from pathlib import Path
import shutil
import time
import numpy as np
from plyfile import PlyData
from common import read_json, save_json, atomic_npy, progress, valid_stage, complete_stage, digest
from core import RGB_DTYPE, transform_points, voxel_keys, buckets, exact_lookup, resize_labels_to_source, frame_votes
from filtered_source import source_layout, build_row_map, REMOVED

def source_frames(config, selected=None, colors=False):
    root = Path(config['source_run'])
    settings = read_json(root / 'config.json')
    with np.load(root / 'prepared.npz') as z:
        scales, owners = z['scales'], z['owners']
    with np.load(root / 'optimized.npz') as z:
        poses = z['poses']
    selected = set(range(len(owners))) if selected is None else set(selected)
    step = settings['output']['pixel_step']
    for wi in range(len(settings['window_indices'])):
        frames = [int(f) for f in np.flatnonzero(owners[:, 0] == wi) if f in selected]
        if not frames:
            continue
        path = root / 'windows' / f'window_{wi:03d}.npz'
        expected = read_json(path.with_suffix('.json'))['sha256']
        assert digest(path) == expected, f'Generator window changed: {path}'
        with np.load(path) as z:
            assert z['indices'].tolist() == settings['window_indices'][wi]
            pp, vv, rawposes = z['points'], z['valid'], z['poses']
            rgb = z['colors'] if colors else None
        for f in frames:
            li = owners[f, 1]
            mask = vv[li].copy()
            grid = np.zeros_like(mask)
            grid[::step, ::step] = True
            mask &= grid
            xyz = transform_points(pp[li][mask], rawposes[li], poses[f], scales[wi])
            assert np.isfinite(xyz).all()
            yield f, xyz, mask, (rgb[li][mask] if colors else None)
        del pp, vv, rawposes, rgb

def build_index(config, plan):
    out = Path(config['output'])
    identity = plan['identity']
    dest = out / 'source_index'
    dest.mkdir(exist_ok=True)
    final = dest / 'complete.json'
    if valid_stage(final, identity):
        progress(out, 'source_index_cached')
        return
    start = time.perf_counter()
    settings = read_json(Path(config['source_run']) / 'config.json')
    voxel, count = settings['voxel_normalized'], settings['output']['voxel_buckets']
    layout = source_layout(config, settings)
    cloud = PlyData.read(layout['unfiltered'], mmap='r')['vertex'].data
    assert len(cloud) < 2**32, 'This release uses uint32 source map indices'
    mapping_file = None
    filtered_info = {}
    if layout['keep_mask'] is not None:
        target_cloud = PlyData.read(layout['target'], mmap='r')['vertex'].data
        keep = np.load(layout['keep_mask'], mmap_mode='r')
        mapping_file = dest / 'unfiltered_to_target.npy'
        progress(out, 'verifying_filtered_source', unfiltered_points=len(cloud), target_points=len(target_cloud))
        filtered_info = build_row_map(cloud, target_cloud, keep, mapping_file)
        del target_cloud, keep
    work = dest / 'work'
    work.mkdir(exist_ok=True)
    shard_meta = work / 'collected.json'
    shard_paths = [work / f'{i:02d}.bin' for i in range(count)]
    if not valid_stage(shard_meta, identity):
        handles = [p.open('wb') for p in shard_paths]
        raw_count = 0
        try:
            for f, xyz, mask, rgb in source_frames(config, colors=True):
                keys = voxel_keys(xyz, voxel)
                bucket = buckets(keys, count)
                data = np.empty(len(xyz), RGB_DTYPE)
                for j, name in enumerate(('x', 'y', 'z')):
                    data[name] = xyz[:, j]
                for j, name in enumerate(('red', 'green', 'blue')):
                    data[name] = rgb[:, j]
                for k, handle in enumerate(handles):
                    data[bucket == k].tofile(handle)
                raw_count += len(data)
                if f % 10 == 0:
                    progress(out, 'source_index_collect', frame=f, total_frames=plan['source_frames'], raw_points=raw_count)
        finally:
            for handle in handles:
                handle.close()
        complete_stage(shard_meta, identity, shard_paths, raw_points=raw_count)
    offset = 0
    index_rows = []
    for k in range(count):
        keyfile = dest / f'keys_{k:02d}.npy'
        meta = dest / f'bucket_{k:02d}.json'
        if valid_stage(meta, identity):
            row = read_json(meta)
            assert row['offset'] == offset
            offset += row['points']
            index_rows.append(dict(bucket=k, offset=row['offset'], points=row['points'], path=str(keyfile)))
            continue
        data = np.fromfile(shard_paths[k], dtype=RGB_DTYPE)
        xyz = np.column_stack([data[n] for n in ('x', 'y', 'z')])
        keys, inverse = np.unique(voxel_keys(xyz, voxel), axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        rebuilt = np.empty(len(keys), RGB_DTYPE)
        for name in ('x', 'y', 'z'):
            rebuilt[name] = np.bincount(inverse, weights=data[name]) / counts
        for name in ('red', 'green', 'blue'):
            rebuilt[name] = np.rint(np.bincount(inverse, weights=data[name]) / counts).clip(0, 255)
        # Bit-for-bit correspondence is required, including averaged RGB and coordinates.
        assert np.array_equal(rebuilt, cloud[offset:offset + len(keys)]), f'Original map mismatch in bucket {k}'
        atomic_npy(keyfile, keys)
        complete_stage(meta, identity, [keyfile], bucket=k, offset=offset, points=len(keys), exact_xyz_rgb_match=True)
        index_rows.append(dict(bucket=k, offset=offset, points=len(keys), path=str(keyfile)))
        offset += len(keys)
        progress(out, 'source_index_verify', bucket=k + 1, total_buckets=count, verified_points=offset)
        del data, xyz, keys, inverse, counts, rebuilt
    assert offset == len(cloud), 'Map point count mismatch'
    save_json(dest / 'index.json', dict(buckets=index_rows, points=offset, voxel=voxel, exact_xyz_rgb_match=True,
              target_points=filtered_info.get('retained_points', offset),
              unfiltered_to_target=str(mapping_file) if mapping_file else None, **filtered_info))
    outputs = [dest / 'index.json'] + [Path(r['path']) for r in index_rows]
    if mapping_file:
        outputs.append(mapping_file)
    complete_stage(final, identity, outputs,
                   seconds=time.perf_counter() - start, points=offset, exact_xyz_rgb_match=True,
                   **filtered_info)
    shutil.rmtree(work)

def project(config, plan):
    out = Path(config['output'])
    dest = out / 'projected'
    dest.mkdir(exist_ok=True)
    identity = plan['identity']
    index = read_json(out / 'source_index/index.json')
    key_arrays = [np.load(r['path'], mmap_mode='r') for r in index['buckets']]
    row_map = np.load(index['unfiltered_to_target'], mmap_mode='r') if index.get('unfiltered_to_target') else None
    pending = [r['index'] for r in plan['frames'] if not valid_stage(dest / f"{r['index']:05d}.json", identity)]
    for f, xyz, mask, _ in source_frames(config, pending):
        keys = voxel_keys(xyz, index['voxel'])
        bucket = buckets(keys, len(key_arrays))
        ids = np.empty(len(keys), dtype=np.uint32)
        for k, base in enumerate(key_arrays):
            sel = bucket == k
            ids[sel] = exact_lookup(base, keys[sel]) + index['buckets'][k]['offset']
        with np.load(out / 'frames' / f'{f:05d}.npz') as z:
            labels, confidence = resize_labels_to_source(z['labels'], z['confidence'], *mask.shape)
        pixel_labels, pixel_confidence = labels[mask], confidence[mask]
        excluded_pixels = 0
        if row_map is not None:
            ids = row_map[ids]
            retained = ids != REMOVED
            excluded_pixels = int((~retained).sum())
            ids, pixel_labels, pixel_confidence = ids[retained], pixel_labels[retained], pixel_confidence[retained]
        votes = frame_votes(ids, pixel_labels, pixel_confidence, config['fusion']['min_pixel_confidence'])
        p = dest / f'{f:05d}.npy'
        atomic_npy(p, votes)
        complete_stage(dest / f'{f:05d}.json', identity, [p], frame=f,
                       valid_source_pixels=len(xyz), observed_voxels=len(votes), unmatched_source_pixels=0,
                       filtered_source_pixels=excluded_pixels)
        progress(out, 'project', frame=f, observed_voxels=len(votes), source_pixels=len(xyz))
