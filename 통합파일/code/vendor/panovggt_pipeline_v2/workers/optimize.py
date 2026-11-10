from runtime import R, C, save, read
import numpy as np
from common import retry_optimization
from refine import Problem, initialize, metrics


def main():
    groups = read(R/'observation_summary.json')['components']
    edges = read(R/'initial_pair_transforms.json')
    with np.load(R/'observations.npz') as data:
        train, evaluation = data['train'], data['evaluation']
    full = np.tile(np.eye(4), (C['frames'], 1, 1))
    initial_full = full.copy()
    reports = []
    out = R/'components'
    out.mkdir(exist_ok=True)
    for ci, ids in enumerate(groups):
        remap = {f: i for i, f in enumerate(ids)}
        obs = train[np.isin(train[:, 0], ids) & np.isin(train[:, 1], ids)].copy()
        ev = evaluation[np.isin(evaluation[:, 0], ids) & np.isin(evaluation[:, 1], ids)].copy()
        for data in (obs, ev):
            data[:, 0] = [remap[int(f)] for f in data[:, 0]]
            data[:, 1] = [remap[int(f)] for f in data[:, 1]]
        local_edges = [{**e, 'a': remap[e['a']], 'b': remap[e['b']]} for e in edges
                       if e['a'] in remap and e['b'] in remap]
        initial = initialize(np.tile(np.eye(4), (len(ids), 1, 1)), local_edges)
        def solve(poses, options):
            return Problem(obs, poses).solve(options['max_nfev'], options['linear_solver'])
        def checkpoint(index, poses, report):
            np.savez(out/f'component_{ci}_attempt_{index}.npz', poses=poses, indices=ids)
            save(out/f'component_{ci}_attempt_{index}.json', report)
            print('OPTIMIZE', ci, index, report, flush=True)
        if len(ids) == 1:
            fitted, attempts = initial, [{'success': True, 'seconds': 0, 'message': 'Singleton gauge; no internal relative pose'}]
        else:
            fitted, attempts = retry_optimization(initial, C['optimization']['attempts'], solve, checkpoint)
        full[ids], initial_full[ids] = fitted, initial
        reports.append({'indices': ids, 'attempts': attempts,
                        'before': metrics(initial, ev, ids), 'after': metrics(fitted, ev, ids)})
    np.savez(R/'components_optimized.npz', poses=full, initial=initial_full)
    save(R/'components_optimization_report.json', {'status': 'complete', 'components': reports,
         'seconds': sum(a['seconds'] for component in reports for a in component['attempts']),
         'depth_shape_optimized': False, 'camera_pose_dof_per_nonanchor_frame': 6})


if __name__ == '__main__':
    main()
