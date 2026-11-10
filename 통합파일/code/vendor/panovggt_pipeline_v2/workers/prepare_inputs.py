from runtime import C, R, save, read, verify, digest
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from masks import correct_mask


def main():
    bundle = read(C['input']['bundle'])
    if bundle['source']['sha256'] != C['input']['source_sha256']:
        raise ValueError('Mask rules are bound to a different source video')
    verify(bundle['source'])
    manifest = read(verify(bundle['frame_manifest']))
    if len(manifest['frames']) != C['frames']:
        raise ValueError('Unexpected number of frames')
    if manifest.get('source_sha256') != C['input']['source_sha256']:
        raise ValueError('Frame manifest/video identity mismatch')
    (R/'input/frames').mkdir(parents=True, exist_ok=True)
    (R/'masks').mkdir(exist_ok=True)
    corrected_hashes = {}; mask_audit = []
    for i, row in enumerate(manifest['frames']):
        name = Path(row['file']).stem + '.png'
        source = verify(bundle['frames'][str(i)])
        if digest(source) != row['sha256']:
            raise ValueError('Frame manifest hash mismatch')
        if Image.open(source).size != (C['width'], C['height']):
            raise ValueError('RGB resolution mismatch')
        dest = R/'input'/row['file']
        if not dest.resolve().is_relative_to((R/'input').resolve()):
            raise ValueError('Invalid frame path')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        base = np.asarray(Image.open(verify(bundle['base_masks'][str(i)])).convert('L'))
        corrected = correct_mask(base, i, C['mask'])
        Image.fromarray(corrected).save(R/'masks'/name)
        corrected_hashes[name] = digest(R/'masks'/name)
        expected = bundle.get('expected_corrected_masks', {}).get(name)
        if expected is not None and corrected_hashes[name] != expected:
            raise ValueError(f'Mask correction differs from the frozen experiment: {name}')
        mask_audit.append({'frame': i, 'fraction': float((corrected > 0).mean()),
                           'explicit_polygons': len(C['mask']['polygons'].get(str(i), []))})
    save(R/'input/manifest.json', manifest)
    save(R/'reviewed_mask_manifest.json', corrected_hashes)
    save(R/'mask_review.json', {'reviewed': True, 'origin': 'Frozen, explicitly configured prior review rules',
                              'runtime_visual_review': False, 'frames': mask_audit})
    (R/'source').mkdir(exist_ok=True)
    target = R/'source/video.mp4'
    if not target.exists():
        target.symlink_to(Path(bundle['source']['path']).resolve())
    save(R/'input_identity.json', {'source_sha256': C['input']['source_sha256'],
                                 'frames': len(manifest['frames']), 'mask_rules': C['mask']})


if __name__ == '__main__':
    main()
