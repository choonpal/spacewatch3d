"""Adapt an existing semantic triangle map and saved scene regions to RGB inputs.

This is input preparation, not segmentation or change inference. Region IDs are
candidate surface components, not verified physical object or video track IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def triangle_lookup(vertices, faces):
    lookup = {}
    for i, face in enumerate(faces):
        key = tuple(sorted(tuple(v) for v in vertices[face]))
        lookup.setdefault(key, []).append(i)
    return lookup


def region_faces(vertices, faces, obj_path, lookup=None, claimed=None, face_classes=None, class_id=None):
    """Recover exact source triangles; never match a region by centroid alone."""
    points, triangles = [], []
    for line in Path(obj_path).read_text().splitlines():
        part = line.split()
        if part and part[0] == 'v':
            points.append([float(v) for v in part[1:4]])
        elif part and part[0] == 'f':
            if len(part) != 4:
                raise ValueError('Only source triangles can be imported')
            triangles.append([int(v.split('/')[0])-1 for v in part[1:]])
    points, triangles = np.asarray(points), np.asarray(triangles)
    distance, _ = cKDTree(vertices).query(points)
    if len(points) == 0 or np.max(distance) > 1e-6:
        raise ValueError('Scene region does not belong to this source mesh')
    lookup = triangle_lookup(vertices, faces) if lookup is None else lookup
    used = set() if claimed is None else set(np.flatnonzero(claimed))
    result = []
    try:
        for f in points[triangles]:
            candidates = lookup[tuple(sorted(tuple(v) for v in f))]
            eligible = [i for i in candidates if i not in used and (face_classes is None or face_classes[i] == class_id)]
            if not eligible:
                raise ValueError('Region triangles overlap or disagree with the source class')
            # Duplicate triangles have identical geometry. Preserve one-to-one
            # source face accounting; never infer distinct physical identities.
            chosen = eligible[0]; used.add(chosen); result.append(chosen)
        return np.asarray(result, 'i8')
    except KeyError as e:
        raise ValueError('Region contains triangles outside the source mesh') from e


def associate(class_id, hit, face_ids, face_classes, person):
    """Class agreement plus native first-hit mesh association, not a 2D track."""
    valid = hit < len(face_ids)
    safe = np.where(valid, hit, 0).astype(int)
    valid &= (face_ids[safe] > 0) & (class_id == face_classes[safe] + 1) & ~person
    return np.where(valid, face_ids[safe], 0).astype('u4')


def sample_ply(path, limit):
    # The registered point-cloud producer writes XYZ float32 + RGB uint8.
    with Path(path).open('rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Invalid point-cloud header')
            header.append(line.decode('ascii').strip())
            if line.strip() == b'end_header':
                break
        offset = f.tell()
    properties = [s for s in header if s.startswith('property ')]
    expected = ['property float x', 'property float y', 'property float z',
                'property uchar red', 'property uchar green', 'property uchar blue']
    if 'format binary_little_endian 1.0' not in header or properties != expected:
        raise ValueError('Expected the source XYZ/RGB binary point cloud')
    count = int(next(s.split()[-1] for s in header if s.startswith('element vertex ')))
    dtype = np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))
    indices = np.linspace(0, count-1, min(count, limit), dtype='i8')
    points = np.array(data['xyz'][indices])
    return points[np.isfinite(points).all(1)], count


def export(config_path, output):
    import open3d as o3d
    config, output = read(config_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    doc = dict(schema='mesh-simsac-input-v1', scope=config.get('scope',
        'Independent captures; automatically inferred object changes require review.'),
        settings=config.get('comparison_settings', {}))
    audit = dict(segmentation_rerun=False, change_predictions_imported=False, captures=[])
    upright = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    for phase, source in zip(['before', 'after'], config['captures']):
        folder = output/phase
        for sub in ['rgb', 'segmentation', 'person']:
            (folder/sub).mkdir(parents=True, exist_ok=True)
        mesh_path, scene_path = Path(source['mesh']), Path(source['scene'])
        mesh = dict(np.load(mesh_path, allow_pickle=False))
        vertices, faces = mesh['vertices'], mesh['faces']
        scene = read(scene_path)
        if scene['capture_id'] != source['id']:
            raise ValueError('Scene capture ID differs from the selected input')
        if scene['coordinates']['up_axis'] != '+Z':
            raise ValueError('This importer requires the declared +Z mesh frame')
        ids = np.zeros(len(faces), 'i4')
        records = []
        lookup = triangle_lookup(vertices, faces)
        for number, obj in enumerate(scene['objects'], 1):
            indices = region_faces(vertices, faces, scene_path.parent/obj['mesh_uri'], lookup,
                                   ids > 0, mesh['face_class_id'], obj['class_id'])
            if len(indices) != obj['source_triangle_count'] or (ids[indices] > 0).any():
                raise ValueError('Scene regions overlap or have a different triangle count')
            if not (mesh['face_class_id'][indices] == obj['class_id']).all():
                raise ValueError('Region class differs from the original semantic mesh')
            ids[indices] = number
            records.append(dict(instance_id=number, label=obj['class_name'],
                source_region_id=obj['instance_id'], face_count=len(indices),
                identity_type='semantic_surface_candidate', physical_identity_verified=False))
        mesh['face_instance_id'] = ids
        np.savez_compressed(folder/'mesh.npz', **mesh)
        report = dict(instance_records=records, provenance=dict(source_mesh=str(mesh_path),
            source_mesh_sha256=sha(mesh_path), source_scene=str(scene_path),
            source_scene_sha256=sha(scene_path), geometry_unchanged=True,
            coincident_source_faces=sum(len(v)-1 for v in lookup.values()),
            coincident_face_policy='First unclaimed exact-coordinate source triangle of the same class; physical identity unverified',
            segmentation_model=source.get('segmentation_model', 'provided semantic labels'),
            instance_method='Existing scene regions mapped to exact original triangles; no new segmentation',
            image_instance_method='Native first-hit mesh association gated by original 2D class; not video instance tracking'))
        (folder/'objects.json').write_text(json.dumps(report, ensure_ascii=False))
        run = Path(source['source_run'])
        frame_manifest = read(run/'input/manifest.json')
        prediction_manifest = read(source['prediction_manifest'])
        if prediction_manifest['source_sha256'] != frame_manifest['source_sha256']:
            raise ValueError('Semantic masks belong to a different source video')
        video = Path(source['video'])
        if sha(video) != frame_manifest['source_sha256']:
            raise ValueError('Video differs from the reconstruction source')
        poses = read(run/'result/camera_poses.json')
        coordinates = read(source['coordinates'])
        source_to_mesh = np.asarray(coordinates['source_to_mesh'])
        labels = [x['name'] for x in sorted(read(source['classes']), key=lambda x: x['id'])]
        points, original_count = sample_ply(run/'result/map.ply', config.get('point_limit', 2000000))
        np.savez_compressed(folder/'points.npz', points=points)
        cloud_provenance = dict(source=str(run/'result/map.ply'), source_sha256=sha(run/'result/map.ply'),
            source_points=original_count, export_points=len(points), sha256=sha(folder/'points.npz'),
            policy='Deterministic evenly spaced original PLY records; no mesh surface samples')
        spec = dict(id=source['id'], mesh=phase+'/mesh.npz', objects=phase+'/objects.json',
            pointcloud=phase+'/points.npz', pointcloud_provenance=cloud_provenance,
            world_from_mesh=upright.tolist(), world_from_pointcloud=(upright@source_to_mesh).tolist(),
            projection='erp', segmentation_labels=labels, frames=[],
            source_video=dict(path=str(video), sha256=frame_manifest['source_sha256']),
            segmentation_provenance=report['provenance'])
        ray_scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        ray_scene.add_triangles(o3d.core.Tensor(vertices.astype('f4')), o3d.core.Tensor(faces.astype('u4')))
        stride = config.get('frame_stride', 6)
        selected = sorted(set(range(0, len(poses), stride)) | {len(poses)-1})
        for i in selected:
            f, p = frame_manifest['frames'][i], poses[i]
            if prediction_manifest['frames'][i]['sha256'] != f['sha256']:
                raise ValueError('Semantic mask input differs from the reconstruction RGB')
            if p['frame'] != i or abs(f['timestamp_s']-p['timestamp_s']) > .01:
                raise ValueError('RGB frame and camera pose disagree')
            image = run/'input'/f['file']
            if sha(image) != f['sha256']:
                raise ValueError('RGB differs from the original input receipt')
            infer = Path(source['predictions'])/f'{i:05d}.npz'
            with np.load(infer, allow_pickle=False) as masks:
                raw = masks['labels']
                cls = np.where(raw < len(labels), raw.astype('u2')+1, 0).astype('u2')
            person = raw == labels.index('person') if 'person' in labels else np.zeros_like(raw, bool)
            mask_path = run/'masks'/f'frame_{i:05d}.png'
            if not mask_path.is_file():
                raise ValueError('Original reconstruction person mask is missing')
            person |= np.asarray(Image.open(mask_path).convert('L').resize(raw.shape[::-1], Image.Resampling.NEAREST)) > 0
            pose = source_to_mesh@np.asarray(p['camera_to_world'])
            scale = np.linalg.det(pose[:3, :3])**(1/3)
            pose[:3, :3] /= scale
            h, w = raw.shape
            y, x = np.mgrid[:h, :w]
            lon, lat = ((x+.5)/w-.5)*2*np.pi, ((y+.5)/h-.5)*np.pi
            directions = np.stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)], -1)@pose[:3, :3].T
            rays = np.concatenate([np.broadcast_to(pose[:3, 3], directions.shape), directions], -1).astype('f4')
            hit = ray_scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)['primitive_ids'].numpy()
            associated = associate(cls, hit, ids, mesh['face_class_id'], person)
            stem = f'{i:05d}'
            shutil.copyfile(image, folder/'rgb'/f'{stem}.jpg')
            np.savez_compressed(folder/'segmentation'/f'{stem}.npz', class_id=cls, instance_id=associated, person=person)
            original_size = Image.open(image).size
            Image.fromarray(person.astype('u1')*255).resize(original_size, Image.Resampling.NEAREST).save(folder/'person'/f'{stem}.png')
            spec['frames'].append(dict(id=str(i), timestamp_s=f['timestamp_s'], source_frame_index=f['source_frame_index'],
                image=f'{phase}/rgb/{stem}.jpg', sha256=f['sha256'], world_from_camera=(upright@pose).tolist(),
                person_mask=f'{phase}/person/{stem}.png', segmentation=f'{phase}/segmentation/{stem}.npz',
                segmentation_rgb_sha256=f['sha256'], source_semantic_sha256=sha(infer)))
            print('INPUT_FRAME', source['id'], i, flush=True)
        doc[phase] = spec
        audit['captures'].append(dict(id=source['id'], mesh_files=1, candidate_regions=len(records),
            source_triangles=len(faces), original_frames=len(poses), sampled_rgb_frames=len(selected),
            labels=labels, source=report['provenance'], pointcloud=cloud_provenance))
    (output/'manifest.json').write_text(json.dumps(doc, ensure_ascii=False))
    (output/'source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    status = Path(args.output).with_suffix('.status.json')
    try:
        export(args.config, args.output)
        status.write_text(json.dumps({'state':'complete'}))
    except Exception as error:
        status.write_text(json.dumps({'state':'failed','message':str(error)}))
        raise
