from pathlib import Path
import shutil
import time
import numpy as np
from plyfile import PlyData
from common import read_json, save_json, complete_stage, valid_stage, progress, ROOT
from core import OUTPUT_DTYPE, RGB_DTYPE, ply_header, fuse_votes

def export(config, plan):
    out = Path(config['output'])
    result = out / 'result'
    result.mkdir(exist_ok=True)
    identity = plan['identity']
    if valid_stage(result / 'complete.json', identity):
        progress(out, 'export_cached')
        return
    started = time.perf_counter()
    source = Path(config['source_run'])
    settings = read_json(source / 'config.json')
    cloud = PlyData.read(source / 'result' / settings['output']['ply_name'], mmap='r')['vertex'].data
    frames = [np.load(out / 'projected' / f"{r['index']:05d}.npy", mmap_mode='r') for r in plan['frames']]
    records = np.concatenate(frames)
    progress(out, 'fusing', frame_voxel_observations=len(records))
    fusion = config['fusion']
    ids, labels, agreement, confidence, observations = fuse_votes(records,
        fusion['min_vote_agreement'], fusion['min_winning_mean_confidence'], fusion['min_observations'])
    del records
    assert observations.max() <= len(plan['frames'])
    classes = read_json(out / 'classes.json')
    palette = np.zeros((256, 3), dtype=np.uint8)
    for item in classes:
        palette[item['id']] = item['rgb']
    selected = result / 'segmented_rgb.ply'
    colored = result / 'segmented_semantic.ply'
    mapping = result / 'source_map_labels.npy'
    # A full-map label array makes the scope explicit: all unobserved points stay 255.
    full_labels = np.lib.format.open_memmap(mapping.with_suffix('.partial.npy'), mode='w+', dtype=np.uint8, shape=(len(cloud),))
    full_labels[:] = 255
    full_labels[ids] = labels
    full_labels.flush()
    del full_labels
    mapping.with_suffix('.partial.npy').replace(mapping)
    sample_ids = np.sort(np.random.default_rng(config['seed']).choice(len(ids), min(config['preview_points'], len(ids)), replace=False))
    preview = np.empty(len(sample_ids), OUTPUT_DTYPE)
    handles = [selected.with_suffix('.partial.ply').open('wb'), colored.with_suffix('.partial.ply').open('wb')]
    try:
        for h in handles:
            h.write(ply_header(len(ids), OUTPUT_DTYPE))
        for start in range(0, len(ids), 1000000):
            end = min(start + 1000000, len(ids))
            original = cloud[ids[start:end]]
            data = np.empty(end - start, OUTPUT_DTYPE)
            for name in RGB_DTYPE.names:
                data[name] = original[name]
            data['semantic_id'] = labels[start:end]
            data['vote_agreement'] = agreement[start:end]
            data['winning_mean_confidence'] = confidence[start:end]
            data['observations'] = observations[start:end]
            data['source_map_index'] = ids[start:end]
            data.tofile(handles[0])
            which = (sample_ids >= start) & (sample_ids < end)
            preview[which] = data[sample_ids[which] - start]
            for j, name in enumerate(('red', 'green', 'blue')):
                data[name] = palette[labels[start:end], j]
            data.tofile(handles[1])
    finally:
        for h in handles:
            h.close()
    for p in (selected, colored):
        p.with_suffix('.partial.ply').replace(p)
    # Verify every exported point against its original row, not just a sample.
    actual = PlyData.read(selected, mmap='r')['vertex'].data
    assert len(actual) == len(ids)
    for start in range(0, len(actual), 1000000):
        part = actual[start:start + 1000000]
        original = cloud[part['source_map_index']]
        assert all(np.array_equal(part[name], original[name]) for name in RGB_DTYPE.names)
        assert np.array_equal(part['semantic_id'], labels[start:start + len(part)])
    np.save(result / 'preview.npy', preview)
    counts = np.bincount(labels, minlength=256)
    histogram = [dict(**c, points=int(counts[c['id']]), percent=round(float(counts[c['id']] / len(ids) * 100), 4))
                 for c in classes if counts[c['id']] > 0]
    histogram.sort(key=lambda c: -c['points'])
    source_validation = read_json(source / 'result/validation.json')
    summary = dict(name=config['name'], version=config['version'], status='complete',
        source_generator_version=settings.get('generator_version', '1.0.0'),
        source_run=str(source), source_map_sha256=config['source_map_sha256'],
        source_filter_applied=settings.get('generator_version') == '2.0.0',
        scene=config.get('scene', source.name.split('_')[0]),
        source_unverified_bridges=source_validation.get('unverified_bridges', []), source_map_points=len(cloud),
        observed_points=len(ids), classified_points=int((labels != 255).sum()),
        unclassified_observed_points=int((labels == 255).sum()), outside_time_scope_points=len(cloud) - len(ids),
        frames=len(plan['frames']), requested_seconds=[config['start_seconds'], config['end_seconds']],
        actual_frame_seconds=[plan['frames'][0]['timestamp_s'], plan['frames'][-1]['timestamp_s']],
        observations_total=int(observations.sum()), observations_mean=float(observations.mean()),
        observations_at_least_2=int((observations >= 2).sum()),
        exact_source_xyz_rgb_preserved=True, exact_source_voxel_correspondence=True,
        inference_forward_seconds=sum(read_json(out / 'frames' / f"{r['index']:05d}.json")['forward_seconds'] for r in plan['frames']),
        max_gpu_reserved_mib=max(read_json(out / 'frames' / f"{r['index']:05d}.json")['peak_reserved_mib'] for r in plan['frames']),
        class_taxonomy='ADE20K 150 classes, zero-based; 255 = unclassified',
        semantics='Class segmentation, not object-instance segmentation. MTPano runs independently per frame; 3D voting is external.',
        scope='Original map voxels observed by generator frames in the requested interval, inclusive. No geometry regenerated.',
        units='Normalized monocular units; metric scale unknown',
        fusion=config['fusion'], classes=histogram, export_seconds=time.perf_counter() - started,
        accuracy='No ground-truth labels; numerical provenance checks do not measure semantic accuracy.')
    save_json(result / 'summary.json', summary)
    complete_stage(result / 'complete.json', identity, [selected, colored, mapping, result / 'preview.npy', result / 'summary.json'],
                   exact_source_xyz_rgb_preserved=True)

def build_viewer(config, plan):
    out = Path(config['output'])
    dest = out / 'viewer'
    dest.mkdir(exist_ok=True)
    preview = np.load(out / 'result/preview.npy', mmap_mode='r')
    summary = read_json(out / 'result/summary.json')
    xyz = np.column_stack([preview[n] for n in ('x', 'y', 'z')]) * np.array([1, -1, -1], dtype=np.float32)
    center = np.median(xyz, axis=0)
    xyz -= center
    rgb = np.column_stack([preview[n] for n in ('red', 'green', 'blue')])
    palette = np.zeros((256, 3), dtype=np.uint8)
    for c in read_json(out / 'classes.json'):
        palette[c['id']] = c['rgb']
    xyz.astype('<f4').tofile(dest / 'xyz.bin')
    rgb.tofile(dest / 'rgb.bin')
    palette[preview['semantic_id']].tofile(dest / 'semantic.bin')
    preview['semantic_id'].tofile(dest / 'labels.bin')
    poses = read_json(Path(config['source_run']) / 'result/camera_poses.json')
    trajectory = np.array([poses[r['index']]['camera_to_world'] for r in plan['frames']])[:, :3, 3]
    trajectory = trajectory * [1, -1, -1] - center
    save_json(dest / 'metadata.json', dict(**summary, preview_points=len(preview), trajectory=trajectory.tolist(),
              bounds=np.quantile(xyz, [.005, .995], axis=0).tolist(),
              cut_height=float(np.quantile(xyz[:, 1], .80)),
              frame_samples=[dict(index=r['index'], timestamp_s=r['timestamp_s'],
                  image=f"frame_{r['index']:05d}.jpg") for r in plan['frames'][::10]]))
    for row in plan['frames'][::10]:
        shutil.copy2(out / 'frames' / f"{row['index']:05d}_overlay.jpg", dest / f"frame_{row['index']:05d}.jpg")
    for name in ('index.html', 'viewer_vendor.js'):
        shutil.copy2(ROOT / 'viewer' / name, dest / name)
    render_overview(dest, xyz, rgb, palette[preview['semantic_id']], trajectory, summary)

def render_overview(dest, xyz, rgb, colors, trajectory, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout='constrained')
    selected = np.arange(0, len(xyz), max(1, len(xyz) // 250000))
    lo, hi = np.quantile(xyz[:, [0, 2]], [.005, .995], axis=0)
    for ax, color, title in zip(axes, (rgb, colors), ('Original RGB', 'MTPano semantic fusion')):
        ax.scatter(xyz[selected, 0], xyz[selected, 2], c=color[selected] / 255., s=.14, linewidths=0, rasterized=True)
        ax.plot(trajectory[:, 0], trajectory[:, 2], color='#ffb84d', lw=1)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), title=title, xlabel='X (normalized)', ylabel='Z (normalized)')
        ax.set_aspect('equal')
    begin, end = summary['requested_seconds']
    fig.suptitle(f"2D to 3D Segmenter v1 | {summary.get('scene', 'dorm3')} {begin:g}-{end:.2f} s | {summary['observed_points']:,} original-map points")
    fig.savefig(dest / 'overview.png', dpi=150)
    plt.close(fig)
