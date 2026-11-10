from runtime import R, C, read, save
from common import check_boundary_policy
from refine import metrics
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    report = read(R/'components_optimization_report.json')
    if report['status'] != 'complete':
        raise ValueError('Component optimization is incomplete')
    groups = [c['indices'] for c in report['components']]
    pairs = check_boundary_policy(groups, C['boundary'])
    manifest = read(R/'input/manifest.json')
    with np.load(R/'components_optimized.npz') as z:
        local, initial_local = z['poses'], z['initial']
    with np.load(R/'prepared.npz') as z:
        scales = z['scales']
    full, before, bridges = local.copy(), initial_local.copy(), []
    for group_index, (a, b) in enumerate(pairs, 1):
        candidates = []
        for wi, ids in enumerate(C['window_indices']):
            if a not in ids or b not in ids:
                continue
            with np.load(R/f'windows/window_{wi:03d}.npz') as z:
                pa, pb = z['poses'][ids.index(a)], z['poses'][ids.index(b)]
            tf = np.eye(4)
            tf[:3, :3] = Rotation.from_matrix(pa[:3, :3].T @ pb[:3, :3]).as_matrix()
            tf[:3, 3] = scales[wi] * pa[:3, :3].T @ (pb[:3, 3]-pa[:3, 3])
            score = abs((ids.index(a)+ids.index(b))/2 - (len(ids)-1)/2)
            candidates.append((score, wi, tf))
        if not candidates:
            raise ValueError(f'No raw prediction window contains boundary {a},{b}')
        _, wi, tf = min(candidates, key=lambda x: (x[0], x[1]))
        full[groups[group_index]] = (full[a] @ tf)[None] @ local[groups[group_index]]
        before[groups[group_index]] = (before[a] @ tf)[None] @ initial_local[groups[group_index]]
        bridges.append({'a': a, 'b': b, 'source_window': wi, 'relative_camera_transform': tf.tolist(),
                        'timestamp_a': manifest['frames'][a]['timestamp_s'],
                        'timestamp_b': manifest['frames'][b]['timestamp_s'], 'visually_verified': False,
                        'reason': 'Configured original-relative-pose fallback after all recovery stages'})
    if not np.isfinite(full).all():
        raise ValueError('Nonfinite camera poses')
    np.savez(R/'optimized.npz', poses=full, initial=before)
    with np.load(R/'observations.npz') as z:
        evaluation = z['evaluation']
    save(R/'registration_report.json', {
        'status': 'converged_components', 'components': len(groups), 'input_frames': len(full),
        'unverified_bridge_count': len(bridges), 'unverified_bridges': bridges,
        'internal_components_converged': True, 'geometry_approved': False,
        'before_optimization': metrics(before, evaluation), 'after_optimization': metrics(full, evaluation),
        'motion_prior': False, 'full_bundle_adjustment': False, 'loop_closure': False,
        'depth': 'Shared-frame scale calibration; fixed per-pixel depth shape',
        'boundary_policy': C['boundary']})
    save(R/'unverified_bridges.json', bridges)


if __name__ == '__main__':
    main()
