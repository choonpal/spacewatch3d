"""Compare two independent clean CPU replays of the same sealed experiment."""
from pathlib import Path
import argparse
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read, digest, save


def compare(first, second):
    roots = [Path(first), Path(second)]
    for root in roots:
        if read(root/'status.json')['status'] != 'complete':
            raise ValueError(f'Run incomplete: {root}')
    validation = [read(r/'result/validation.json') for r in roots]
    hashes = [digest(v['point_cloud']) for v in validation]
    for v, h in zip(validation, hashes):
        if v['sha256'] != h:
            raise ValueError('Final PLY changed after export')
    poses = [np.load(r/'optimized.npz')['poses'] for r in roots]
    with np.load(roots[0]/'observations.npz') as a, np.load(roots[1]/'observations.npz') as b:
        observations_equal = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    masks = [read(r/'reviewed_mask_manifest.json') for r in roots]
    boundaries = [read(r/'unverified_bridges.json') for r in roots]
    audits = [read(r/'edge_audit.json') for r in roots]
    signatures = [[(e['a'], e['b'], e['match_source'], e['source_sha256'], e['accepted']) for e in audit] for audit in audits]
    report = {
        'runs': [str(r) for r in roots],
        'input_code_environment_identity_equal': read(roots[0]/'run_identity.json')['sha256'] == read(roots[1]/'run_identity.json')['sha256'],
        'mask_hashes_equal': masks[0] == masks[1], 'observations_equal': observations_equal,
        'accepted_matches_and_sources_equal': signatures[0] == signatures[1],
        'camera_poses_equal': np.array_equal(poses[0], poses[1]),
        'max_camera_pose_absolute_difference': float(np.max(abs(poses[0]-poses[1]))),
        'boundary_policy_results_equal': boundaries[0] == boundaries[1],
        'points': [v['points'] for v in validation], 'ply_sha256': hashes,
        'full_ply_bitwise_equal': hashes[0] == hashes[1],
        'scope': 'Two clean CPU replays from frozen RGB/masks/raw predictions/matches. Not a repeated fresh GPU-inference test.'}
    report['passed'] = all(report[k] for k in [
        'input_code_environment_identity_equal', 'mask_hashes_equal', 'observations_equal',
        'accepted_matches_and_sources_equal', 'camera_poses_equal', 'boundary_policy_results_equal', 'full_ply_bitwise_equal'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    save(args.output, result)
    print('PASS' if result['passed'] else 'MISMATCH', result['points'], result['ply_sha256'])
    raise SystemExit(0 if result['passed'] else 1)
