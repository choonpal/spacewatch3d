#!/usr/bin/env python3
"""Seal the existing dorm3 inputs and raw caches, without trusting old geometry results."""
from pathlib import Path
import argparse
import shutil
from common import read, save, record, verify, digest, validate_config, cache_contract, windows


def freeze(config, legacy, base_masks, destination):
    c = read(config)
    validate_config(c)
    legacy, base_masks, out = Path(legacy).resolve(), Path(base_masks).resolve(), Path(destination).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty input-bundle directory')
    old = read(legacy/'config.json')
    for key, expected in [('source_sha256', c['input']['source_sha256']), ('width', c['input']['width']),
                          ('height', c['input']['height']), ('frames', c['input']['expected_frames']),
                          ('seed', c['execution']['seed']), ('checkpoint_sha256', c['model']['checkpoint_sha256']),
                          ('revision', c['model']['revision']), ('depth_percentiles', c['model']['depth_percentiles']),
                          ('top_fraction_excluded', c['model']['top_fraction_excluded'])]:
        if old[key] != expected:
            raise ValueError(f'Legacy/config mismatch: {key}')
    if old['precision'] != 'torch.float32':
        raise ValueError('Legacy prediction precision differs')
    if old['window_indices'] != windows(c['input']['expected_frames'], c['model']['window_size'], c['model']['overlap']):
        raise ValueError('Legacy window schedule differs')
    for key in ['face_size', 'confidence', 'gaps', 'unique_erp_cell']:
        if old['matching'][key] != c['matching'][key]:
            raise ValueError('Legacy matching configuration differs: '+key)
    # This importer intentionally understands exactly the audited legacy recovery recipe.
    if (c['matching']['model'] != 'indoor_new' or c['matching']['sift']['ratio'] != .65
            or c['matching']['sift']['max_gap'] != 8 or c['matching']['sift']['features'] != 10000
            or c['matching']['sift']['contrast'] != .003 or c['matching']['sift']['edge_threshold'] != 10
            or c['matching']['high_resolution']['face_size'] != 768 or c['matching']['border_pixels'] != 4):
        raise ValueError('Legacy recovery recipe is incompatible; generate fresh matches instead')
    if read(legacy/'hires_matching_summary.json')['status'] != 'complete':
        raise ValueError('Legacy high-resolution matcher has not finished')
    manifest = read(legacy/'input/manifest.json')
    out.mkdir(parents=True, exist_ok=True)
    (out/'base_masks').mkdir()
    frames, bases = {}, {}
    for i, row in enumerate(manifest['frames']):
        p = legacy/'input'/row['file']
        if digest(p) != row['sha256']:
            raise ValueError('Corrupt input frame')
        frames[str(i)] = record(p)
        name = Path(row['file']).stem+'.png'
        shutil.copyfile(base_masks/name, out/'base_masks'/name)
        bases[str(i)] = record(out/'base_masks'/name)
    shutil.copyfile(legacy/'input/manifest.json', out/'frame_manifest.json')
    source = record(legacy/'source'/old['source'])
    if source['sha256'] != c['input']['source_sha256']:
        raise ValueError('Source video hash mismatch')
    expected = read(legacy/'reviewed_mask_manifest.json')
    # The generated correction must match independently frozen final masks before any cache is blessed.
    import numpy as np
    from PIL import Image
    from masks import correct_mask
    for i, row in enumerate(manifest['frames']):
        name = Path(row['file']).stem+'.png'
        corrected = correct_mask(np.asarray(Image.open(bases[str(i)]['path']).convert('L')), i, c['mask'])
        if not np.array_equal(corrected, np.asarray(Image.open(legacy/'masks'/name).convert('L'))):
            raise ValueError(f'Explicit mask rules do not reproduce legacy mask {name}')
    bundle = {'schema_version': 1, 'source': source, 'frame_manifest': record(out/'frame_manifest.json'),
              'frames': frames, 'base_masks': bases, 'expected_corrected_masks': expected,
              'base_mask_provenance': 'Frozen masks after SAM/envelope/head processing, before reviewed dilation and polygons. This importer does not rerun SAM.'}
    save(out/'assets.json', bundle)
    files = {}
    for folder in ['windows', 'matches', 'bridge_matches', 'enhanced_matches', 'hires_matches']:
        for p in sorted((legacy/folder).iterdir()):
            if p.suffix not in ('.npz', '.json'):
                continue
            files[f'{folder}/{p.name}'] = record(p)
            if folder == 'windows' and p.suffix == '.npz':
                if read(p.with_suffix('.json'))['sha256'] != files[f'{folder}/{p.name}']['sha256']:
                    raise ValueError('Corrupt prediction window')
    save(out/'replay_cache.json', {'schema_version': 1, 'cache_contract': cache_contract(c),
         'input_bundle_sha256': digest(out/'assets.json'), 'files': files,
         'note': 'Only raw inference/matching imported. All observations, optimization and outputs MUST be recomputed.'})
    print('FROZEN', out, len(frames), 'frames;', len(files), 'cache files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--legacy-run', required=True)
    parser.add_argument('--base-masks', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    freeze(args.config, args.legacy_run, args.base_masks, args.output)
