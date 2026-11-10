"""v2 final map and presentation artifacts, derived only from retained vertices."""
from pathlib import Path
import shutil
import time
from runtime import C, R, read, save, digest
import numpy as np
from plyfile import PlyData
from component_filter import filter_cloud, header


def main():
    started = time.perf_counter()
    original = R/'unfiltered_result'; out = R/'result'
    previous = read(original/'validation.json')
    report = filter_cloud(original/C['output']['ply_name'], out/C['output']['ply_name'],
                          R/'component_filter', C['postprocessing']['connected_components'],
                          previous['sha256'], progress=lambda s: print(s, flush=True))
    v = PlyData.read(out/C['output']['ply_name'], mmap='r')['vertex'].data
    ids = np.sort(np.random.default_rng(C['output']['preview_seed']).choice(
        len(v), min(C['output']['preview_points'], len(v)), replace=False))
    sample = v[ids]
    with (out/'preview.ply').open('wb') as f:
        f.write(header(len(sample))); sample.tofile(f)
    xyz = np.column_stack([sample[k] for k in ['x', 'y', 'z']])
    rgb = np.column_stack([sample[k] for k in ['red', 'green', 'blue']])
    np.savez_compressed(out/'viewer_sample.npz', xyz=xyz, rgb=rgb)
    shutil.copyfile(original/'camera_poses.json', out/'camera_poses.json')
    save(out/'settings.json', C)
    validation = {**previous, 'generator_version': '2.0.0', 'point_cloud': str(out/C['output']['ply_name']),
                  'points': len(v), 'sha256': report['sha256'], 'bytes': report['bytes'],
                  'unfiltered_points': previous['points'], 'unfiltered_sha256': previous['sha256'],
                  'component_filter': report, 'export_seconds': previous['seconds'],
                  'seconds': time.perf_counter()-started,
                  'limitations': previous['limitations']+report['limitations']}
    save(out/'validation.json', validation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    poses = np.asarray([r['camera_to_world'] for r in read(out/'camera_poses.json')])
    select = np.arange(0, len(xyz), max(1, len(xyz)//180000))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
    for ax, (i, j, title) in zip(axes, [(0, 2, 'Horizontal projection (X/Z)'), (0, 1, 'Side projection (X/Y)')]):
        ax.scatter(xyz[select,i], xyz[select,j], c=rgb[select]/255, s=.08, rasterized=True)
        ax.plot(poses[:,i,3], poses[:,j,3], color='#e64980', lw=1)
        for bridge in previous.get('unverified_bridges', []):
            pair = [bridge['a'], bridge['b']]
            ax.plot(poses[pair,i,3], poses[pair,j,3], color='red', lw=3)
        lo, hi = np.quantile(xyz[:,[i,j]], [.005,.995], axis=0); pad = .03*(hi-lo)
        ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
        ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('Normalized monocular units')
    fig.suptitle(f"{C['name']} | removed components <= {report['options']['remove_max_points_inclusive']:,} points | {len(v):,} retained")
    fig.savefig(out/'overview.png', dpi=180); plt.close(fig)
    save(R/'component_filter_status.json', dict(status='complete', seconds=time.perf_counter()-started,
         report=report, files={p.name: dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()}))
    print('COMPONENT_FILTER_COMPLETE', report['kept_points'], report['removed_points'], flush=True)


if __name__ == '__main__':
    main()
