"""Automatic RGB box segmentation and conservative transfer onto source faces."""
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def generate_boxes(config, output):
    settings = config.get('box_segmentation')
    if not settings:
        return
    tasks = []
    for source in config['captures']:
        run = Path(source['source_run'])
        manifest = json.loads((run/'input/manifest.json').read_text())
        stride = config.get('frame_stride', 6)
        selected = sorted(set(range(0, len(manifest['frames']), stride)) | {len(manifest['frames'])-1})
        cache = Path(settings['cache'])/(source['id']+'-'+manifest['source_sha256'][:16])
        source['box_masks'] = str(cache)
        tasks.extend(dict(image=str(run/'input'/manifest['frames'][i]['file']), output=str(cache/f'{i:05d}.npz')) for i in selected)
    task_file = Path(output)/'box-tasks.json'
    task_file.write_text(json.dumps(dict(dino=settings['dino'], sam2=settings['sam2'], tasks=tasks)))
    import os
    env = os.environ.copy()
    env.update(settings.get('environment', {}))
    subprocess.run([settings['python'], str(Path(__file__).with_name('ground_boxes.py')), '--config', str(task_file)],
                   env=env, check=True)


def load_boxes(folder, frame, shape, source_image_sha256):
    path = Path(folder)/f'{frame:05d}.npz'
    record = json.loads(path.with_suffix('.json').read_text())
    if record['image_sha256'] != source_image_sha256:
        raise ValueError('Box mask belongs to a different source RGB')
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['mask_sha256']:
        raise ValueError('Box segmentation cache changed')
    with np.load(path) as data:
        masks = data['masks']
    mask = masks.any(0) if len(masks) else np.zeros(shape, bool)
    mask = np.asarray(Image.fromarray(mask).resize(shape[::-1], Image.Resampling.NEAREST))
    return mask, record


def box_face_instances(faces, ids, classes, frame_hits, box_class):
    """Require mask coverage in two independent frames, then edge connectivity.

    Source vertices/faces are preserved. A mask with no supporting source face
    stays RGB-only; a box surface is never synthesized to fill missing geometry.
    """
    support = np.zeros(len(faces), 'u4')
    observed = np.zeros_like(support)
    for hit, mask in frame_hits:
        valid = hit < len(faces)
        total = np.bincount(hit[valid].astype(int), minlength=len(faces))
        positive = np.bincount(hit[valid & mask].astype(int), minlength=len(faces))
        visible = total >= 2
        observed += visible
        support += visible & (positive >= 2) & (positive >= .6*total)
    selected = np.flatnonzero((support >= 2) & (support >= .6*observed))
    records = []
    if not len(selected):
        return ids, classes, records
    # Only triangles sharing a source vertex can become one mesh instance.
    owners = {}
    rows, cols = [], []
    for index, face in enumerate(faces[selected]):
        for vertex in face:
            if int(vertex) in owners:
                other = owners[int(vertex)]; rows.extend([index, other]); cols.extend([other, index])
            else:
                owners[int(vertex)] = index
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(selected), len(selected))).tocsr()
    count, labels = connected_components(graph, directed=False)
    ids, classes = ids.copy(), classes.copy()
    next_id = int(ids.max())+1
    for group in range(count):
        indices = selected[labels == group]
        if len(indices) < 2:
            continue
        ids[indices], classes[indices] = next_id, box_class
        records.append(dict(instance_id=next_id, label='box', face_count=len(indices),
            identity_type='rgb_segmented_source_mesh_component', physical_identity_verified=False,
            source='Grounding DINO base + SAM 2.1 small', minimum_independent_frame_support=int(support[indices].min())))
        next_id += 1
    return ids, classes, records
