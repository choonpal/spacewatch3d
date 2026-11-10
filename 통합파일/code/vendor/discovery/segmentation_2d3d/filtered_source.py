"""Exact v2 keep-mask mapping; no nearest-neighbor transfer of semantic labels."""
from pathlib import Path
import numpy as np
from common import read_json

REMOVED = np.iinfo(np.uint32).max


def source_layout(config, settings):
    root = Path(config['source_run'])
    target = root / 'result' / settings['output']['ply_name']
    version = settings.get('generator_version', '1.0.0')
    if version == '1.0.0':
        return dict(target=target, unfiltered=target, keep_mask=None, report=None)
    if version != '2.0.0':
        raise ValueError(f'Unsupported pointcloud generator version: {version}')
    report = root / 'component_filter/report.json'
    info = read_json(report)
    unfiltered = root / 'unfiltered_result' / settings['output']['ply_name']
    if Path(info['source']).resolve() != unfiltered.resolve() or Path(info['output']).resolve() != target.resolve():
        raise ValueError('Component filter provenance does not refer to this source run')
    return dict(target=target, unfiltered=unfiltered,
                keep_mask=root / 'component_filter/keep_mask.npy', report=report)


def build_row_map(unfiltered, target, keep, output, chunk_size=1000000):
    """Map unfiltered PLY row order to compact v2 PLY rows, verifying every row."""
    if keep.dtype != np.bool_ or keep.ndim != 1 or len(keep) != len(unfiltered):
        raise ValueError('Expected a boolean keep mask matching the unfiltered map')
    if not 0 < len(target) < REMOVED:
        raise ValueError('Target must be nonempty and fit uint32 with a removed sentinel')
    output = Path(output)
    temporary = output.with_suffix('.partial.npy')
    mapping = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint32, shape=(len(keep),))
    offset = 0
    try:
        for start in range(0, len(keep), chunk_size):
            end = min(start + chunk_size, len(keep))
            mask = keep[start:end]
            selected = unfiltered[start:end][mask]
            n = len(selected)
            if not np.array_equal(selected, target[offset:offset+n]):
                raise ValueError(f'Filtered XYZ/RGB or point order mismatch at source row {start}')
            block = np.full(end-start, REMOVED, dtype=np.uint32)
            block[mask] = np.arange(offset, offset+n, dtype=np.uint32)
            mapping[start:end] = block
            offset += n
        if offset != len(target):
            raise ValueError('Filtered target point count does not match keep mask')
        mapping.flush()
    finally:
        del mapping
    temporary.replace(output)
    return dict(unfiltered_points=len(unfiltered), retained_points=offset,
                removed_points=len(unfiltered)-offset, exact_filtered_xyz_rgb_match=True)
