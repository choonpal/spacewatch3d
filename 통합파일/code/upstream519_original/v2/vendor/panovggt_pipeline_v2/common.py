"""IO and immutable run contracts; importing this module never initializes CUDA."""
from pathlib import Path
import hashlib
import json


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record(path):
    path = Path(path).absolute()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def verify(item):
    path = Path(item['path'])
    if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
        raise ValueError(f'Input/artifact hash mismatch: {path}')
    return path


def windows(n, size, overlap):
    if not 1 <= overlap < size <= n:
        raise ValueError('Require 1 <= overlap < window_size <= frame_count')
    starts = list(range(0, n-size+1, size-overlap))
    if starts[-1] != n-size:
        starts.append(n-size)
    return [list(range(s, s+size)) for s in starts]


def validate_config(c):
    if c['schema_version'] != 1:
        raise ValueError('Unsupported schema version')
    if c['execution']['server_address'] != '155.230.28.172':
        raise ValueError('Inference is restricted to the user-provided GPU server')
    if type(c['execution']['physical_gpu']) is not int or c['execution']['physical_gpu'] < 0:
        raise ValueError('Select exactly one nonnegative physical GPU index')
    if c['model']['precision'] != 'float32':
        raise ValueError('This validated implementation supports FP32 only')
    w, h = c['input']['width'], c['input']['height']
    if w != 2*h or w % 14 or h % 14:
        raise ValueError('ERP dimensions must be 2:1 and divisible by patch size 14')
    if c['mask']['coordinate_size'] != [w, h]:
        raise ValueError('Mask polygon reference dimensions must match the input')
    if c['mask']['base'] != 'provided_pre_review_masks':
        raise ValueError('Provide frozen pre-review masks; no implicit segmentation')
    if any(k < 1 or k % 2 != 1 for k in c['mask']['periodic_dilation_kernel']):
        raise ValueError('Mask dilation dimensions must be positive odd integers')
    if c['output']['pixel_step'] < 1 or c['output']['voxel_normalized'] <= 0:
        raise ValueError('Invalid cloud sampling/voxel configuration')
    import math
    if c.get('generator_version') != '2.0.0':
        raise ValueError('This runner requires generator_version 2.0.0')
    post = c['postprocessing']
    cc = post['connected_components']
    if cc['enabled'] is not True or cc['connectivity'] != 26:
        raise ValueError('v2 requires enabled 26-neighbour occupied-voxel components')
    if not math.isfinite(cc['voxel_size_normalized']) or cc['voxel_size_normalized'] <= 0:
        raise ValueError('Component voxel size must be finite and positive')
    if cc['coordinate_frame'] != 'reconstruction_world' or cc['grid_origin'] != [0, 0, 0]:
        raise ValueError('Component grid is fixed to the reconstruction world axes and origin')
    for key in ['remove_max_points_inclusive', 'slab_width', 'max_dense_cells', 'chunk_points']:
        if type(cc[key]) is not int or cc[key] < (0 if key == 'remove_max_points_inclusive' else 1):
            raise ValueError('Invalid component parameter: ' + key)
    if post['radius_outlier_removal'] or post['statistical_outlier_removal']:
        raise ValueError('v2 adds only component-size filtering; ROR/SOR are not implemented')
    if Path(c['output']['ply_name']).name != c['output']['ply_name']:
        raise ValueError('ply_name must be a filename')
    if c['boundary']['policy'] not in ('fail', 'original_relative_pose'):
        raise ValueError('Boundary policy must be fail or original_relative_pose')
    if c['boundary']['on_unlisted_boundary'] != 'fail':
        raise ValueError('Unlisted boundaries must fail closed')
    if c['boundary']['window_selection'] != 'closest_pair_to_window_center':
        raise ValueError('Unsupported bridge window selection')
    if c['matching']['source_order'] != ['matches', 'bridge_matches', 'enhanced_matches', 'hires_matches']:
        raise ValueError('Source priority must match the explicit sequential recovery order')
    if not c['matching']['high_resolution']['all_directions']:
        raise ValueError('High-resolution recovery requires all-direction matching')
    if not 0 < c['registration']['train_fraction'] < 1:
        raise ValueError('Train fraction must lie between zero and one')
    if not c['optimization']['attempts'] or any(a['max_nfev'] < 1 for a in c['optimization']['attempts']):
        raise ValueError('Provide explicit nonempty optimizer retry schedule')
    if c['optimization']['loss'] != 'soft_l1':
        raise ValueError('This implementation reports the soft_l1 objective only')
    windows(c['input']['expected_frames'], c['model']['window_size'], c['model']['overlap'])


def make_runtime(c, root):
    """Keep scientific options visible while adapting the numerical workers."""
    return {
        **c, 'server_root': str(root), 'width': c['input']['width'], 'height': c['input']['height'],
        'frames': c['input']['expected_frames'], 'sample_interval_s': c['input']['nominal_interval_seconds'],
        'physical_gpu': c['execution']['physical_gpu'], 'seed': c['execution']['seed'],
        'window_indices': windows(c['input']['expected_frames'], c['model']['window_size'], c['model']['overlap']),
        'checkpoint_sha256': c['model']['checkpoint_sha256'], 'revision': c['model']['revision'],
        'top_fraction_excluded': c['model']['top_fraction_excluded'], 'depth_percentiles': c['model']['depth_percentiles'],
        'voxel_normalized': c['output']['voxel_normalized'], 'pixel_step': c['output']['pixel_step'],
    }


def select_match_files(root, sources):
    files = {}
    for source in sources:
        if source not in ('matches', 'bridge_matches', 'enhanced_matches', 'hires_matches'):
            raise ValueError(f'Unknown match source {source}')
        files.update({p.name: p for p in sorted((Path(root)/source).glob('*.npz'))})
    return files


def cache_contract(c):
    """Only stages preceding geometric fitting may be reused from the legacy run."""
    return {
        'source': c['input']['source_sha256'],
        'shape': [c['input']['width'], c['input']['height'], c['input']['expected_frames']],
        'seed': c['execution']['seed'],
        'model': {k: v for k, v in c['model'].items() if k not in ('repository', 'checkpoint')},
        'mask': c['mask'], 'matching': c['matching'],
    }


def retry_optimization(initial, attempts, solve, save_attempt):
    """Every retry starts from this run's preceding result, never another run's file."""
    poses = initial.copy()
    reports = []
    for index, options in enumerate(attempts):
        poses, report = solve(poses, options)
        report = {**report, 'attempt': index, 'options': options}
        save_attempt(index, poses, report)
        reports.append(report)
        if report['success']:
            return poses, reports
    raise RuntimeError('Optimizer did not converge within the configured retry schedule; checkpoints retained')


def check_boundary_policy(groups, policy):
    pairs = []
    for previous, current in zip(groups, groups[1:]):
        a, b = max(previous), min(current)
        if b != a+1:
            raise ValueError('Interleaved/noncontiguous components cannot be bridged by a temporal boundary')
        pairs.append([a, b])
    if not pairs:
        return pairs
    if policy['policy'] == 'fail':
        raise ValueError(f'Unverified boundaries remain: {pairs}')
    if len(pairs) > policy['max_unverified_bridges']:
        raise ValueError('Unverified bridge count exceeds configured limit')
    for pair in pairs:
        if pair not in policy['allowed_pairs']:
            raise ValueError(f'Unlisted boundary: {pair}')
    return pairs
