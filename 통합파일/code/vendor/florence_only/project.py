"""Florence polygon masks -> exact reconstructed pixels -> candidate instance IDs.

Uses geometry only from the existing reconstruction. No semantic mesh labels,
SAM/DINO/OOOPS outputs, class size priors, or chair-specific split rules.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from florence_core import read, save, sha


def associate(observations, clouds, radius, minimum_overlap=.3):
    """Complete-link spatial matching; never merge two detections in one view."""
    samples = []
    for p in clouds:
        _, ids = np.unique(np.floor(p / (radius / 3)).astype('i8'), axis=0, return_index=True)
        samples.append(p[ids])
    trees = [cKDTree(p) for p in samples]
    score = np.zeros((len(clouds), len(clouds)))
    for i, a in enumerate(observations):
        for j in range(i):
            b = observations[j]
            if a['label'] != b['label'] or (a['frame'], a['view']) == (b['frame'], b['view']):
                continue
            ab = (trees[j].query(samples[i])[0] <= radius).mean()
            ba = (trees[i].query(samples[j])[0] <= radius).mean()
            score[i, j] = score[j, i] = min(ab, ba)
    groups = []
    for i in range(len(clouds)):
        choices = [(float(score[i, group].mean()), gi) for gi, group in enumerate(groups)
                   if score[i, group].min() >= minimum_overlap]
        if choices:
            groups[max(choices)[1]].append(i)
        else:
            groups.append([i])
    return groups, score


def run(args):
    started = time.perf_counter()
    sys.path.insert(0, str(Path(args.vendor).resolve()))
    from geometry import source_frames
    from scene_discovery.common import erp_to_view
    from scene_discovery.measure import dimensions, clean_points
    output = Path(args.output)
    prepared = read(Path(args.discovery) / 'prepared.json')
    scene = next(s for s in prepared['scenes'] if s['id'] == args.scene)
    coordinate = read(scene['coordinates'])
    tf = np.asarray(coordinate['source_to_mesh'])
    if not np.allclose(coordinate.get('up', [0, 0, 1]), [0, 0, 1]):
        raise ValueError('Dimension reporting requires a Z-up coordinate transform')
    settings = read(Path(scene['source_run']) / 'config.json')
    singular = np.linalg.svd(tf[:3, :3], compute_uv=False)
    if not np.allclose(singular, singular[0], rtol=1e-5):
        raise ValueError('Expected a uniform reconstruction-to-display scale')
    voxel = float(settings['voxel_normalized'] * singular[0])
    grouped = defaultdict(list)
    inference = read(output / 'inference_summary.json')
    paths = [output / 'predictions' / (key + '.json') for key in inference['prediction_keys'] if key.endswith('_region')]
    for path in sorted(paths):
        record = read(path)
        record['record_path'] = str(path)
        if sha(path.with_suffix('.npz')) != record['mask_sha256']:
            raise ValueError('Mask changed: ' + str(path))
        grouped[record['view']['frame']].append(record)
    observations, clouds, colors, pixel_clouds, rejected, maps = [], [], [], [], [], {}
    # Use only points surviving reconstruction component filtering, when indexed.
    source_index = None
    from spacewatch_pipeline.correspondence import ExportCorrespondence
    shared = ExportCorrespondence.available(scene['source_run'])
    if scene.get('source_index'):
        from core import voxel_keys, buckets, exact_lookup
        from filtered_source import source_layout, build_row_map
        from plyfile import PlyData
        indexdir = Path(scene['source_index'])
        for path, item in read(indexdir/'complete.json')['files'].items():
            if sha(path) != item['sha256']: raise ValueError('Source index changed: '+path)
        source_index = read(indexdir/'index.json')
        layout=source_layout(dict(source_run=scene['source_run']),settings)
        build_row_map(PlyData.read(layout['unfiltered'],mmap='r')['vertex'].data,
                      PlyData.read(layout['target'],mmap='r')['vertex'].data,
                      np.load(layout['keep_mask'],mmap_mode='r'),output/'verified_rowmap.npy')
        rowmap=np.load(source_index['unfiltered_to_target'],mmap_mode='r')
        if not np.array_equal(np.load(output/'verified_rowmap.npy',mmap_mode='r'),rowmap):
            raise ValueError('Source index belongs to another reconstruction')
        keys=[np.load(r['path'],mmap_mode='r') for r in source_index['buckets']]
    for frame, xyz, valid, rgb in source_frames(dict(source_run=scene['source_run']), selected=grouped, colors=True):
        source_row = read(Path(scene['source_run']) / 'input/manifest.json')['frames'][frame]
        p = xyz @ tf[:3, :3].T + tf[:3, 3]
        pixels = np.flatnonzero(valid)
        if source_index is not None:
            if shared is not None:
                rows, shared_mask = shared.frame(frame)
                if not np.array_equal(shared_mask, valid):
                    raise ValueError('Export correspondence mask changed')
            else:
                cells=voxel_keys(xyz,source_index['voxel']);bucket=buckets(cells,len(keys));rows=np.empty(len(xyz),np.uint32)
                for k,base in enumerate(keys):
                    sel=bucket==k
                    if sel.any():rows[sel]=exact_lookup(base,cells[sel])+source_index['buckets'][k]['offset']
            retained=rowmap[rows]!=np.uint32(2**32-1)
            p,rgb,pixels=p[retained],rgb[retained],pixels[retained]
        for record in grouped[frame]:
            if source_row['sha256'] != record['view']['source_sha256']:
                raise ValueError('Reconstruction image provenance mismatch')
            key, quality, box = record['key'], record['quality'], record['box']
            base = dict(key=key, label=record['label'], frame=frame, view=record['view']['view']['name'])
            if not quality['usable'] or record['result']['output_tokens']>=inference['max_tokens']+1:
                rejected.append(dict(**base, reason=quality.get('rejection_reason') or 'generation_limit_reached'))
                continue
            if quality['outside_box_fraction']>.1:
                rejected.append(dict(**base, reason='mask_outside_detection_box', fraction=quality['outside_box_fraction']))
                continue
            mask = np.load(Path(record['record_path']).with_suffix('.npz'))['mask']
            if mask.shape[0] != mask.shape[1]:
                raise ValueError('Prepared perspective view must be square')
            view = record['view']['view']
            map_key = (valid.shape, view['yaw'], view['pitch'], mask.shape, view.get('fov', args.fov))
            if map_key not in maps:
                maps[map_key] = erp_to_view(valid.shape, view, size=mask.shape[0], fov=view.get('fov', args.fov))
            visible, index = maps[map_key]
            selected = visible[pixels] & binary_erosion(mask, iterations=1).ravel()[index[pixels]]
            raw = p[selected]
            if len(raw) < args.min_points:
                rejected.append(dict(**base, reason='insufficient_projected_points', points=len(raw)))
                continue
            clean, purity = clean_points(raw)
            if len(clean) < args.min_points:
                rejected.append(dict(**base, reason='insufficient_connected_depth_points', points=len(clean)))
                continue
            dims = dimensions(clean)
            if dims.min() <= 1e-5:
                rejected.append(dict(**base, reason='degenerate_extent', points=len(clean)))
                continue
            _, color_ids = cKDTree(raw).query(clean)
            color = rgb[selected][color_ids]
            _, ids = np.unique(np.floor(clean / voxel).astype('i8'), axis=0, return_index=True)
            clean, color = clean[ids], color[ids]
            reasons = []
            if quality['border_touch']: reasons.append('view_border_truncation')
            if quality['outside_box_fraction'] > .1: reasons.append('mask_outside_detection_box')
            if purity < .75: reasons.append('fragmented_depth_support')
            if mask.mean() > .75: reasons.append('mask_covers_most_of_view')
            if record['label'] in args.dynamic_labels.split(','): reasons.append('dynamic_category')
            observations.append(dict(**base, observation_id=len(observations), points=len(clean),
                raw_points=len(raw), connected_fraction=purity, dimensions=dims.tolist(),
                bounds=np.quantile(clean, [.01, .99], axis=0).tolist(),
                reference_rejection=reasons, box=box, mask_quality=quality))
            clouds.append(clean.astype('f4'))
            colors.append(color.astype('u1'))
            pixel_clouds.append(pixels[selected][color_ids][ids].astype('i8'))
        print('PROJECT', frame, 'observations', len(observations), 'rejected', len(rejected), flush=True)
    # Resolution-relative tolerance; no class-specific size or shape assumption.
    spacings = []
    for p in clouds:
        if len(p) > 1:
            nn = cKDTree(p).query(p[::max(1, len(p)//1000)], k=2)[0][:, 1]
            spacings.append(float(np.median(nn[nn > 0])))
    radius = max(4 * voxel, 2 * float(np.median(spacings))) if spacings else 4 * voxel
    groups, scores = associate(observations, clouds, radius, args.overlap)
    instances, allpoints, allrgb, allids, allobs, allframes, allpixels = [], [], [], [], [], [], []
    for candidate_id, group in enumerate(groups, start=1):
        members = [observations[i] for i in group]
        good = [m for m in members if not m['reference_rejection']]
        frames = sorted(set(m['frame'] for m in members))
        good_frames = sorted(set(m['frame'] for m in good))
        good_dims = [m['dimensions'] for m in good]
        ratio = np.max(good_dims, axis=0)/np.maximum(np.min(good_dims, axis=0), 1e-8) if good_dims else None
        reasons = []
        if len(good_frames) < 2: reasons.append('fewer_than_two_complete_frames')
        if ratio is not None and ratio.max() > 1.4: reasons.append('inconsistent_dimensions')
        cloud = np.concatenate([clouds[i] for i in group])
        rgb = np.concatenate([colors[i] for i in group])
        for i in group: observations[i]['candidate_id'] = candidate_id
        instances.append(dict(candidate_id=candidate_id, label=members[0]['label'],
            observation_ids=group, observation_keys=[m['key'] for m in members], frames=frames,
            complete_frames=good_frames, dimensions=dimensions(cloud).tolist(),
            size_ratio=None if ratio is None else ratio.tolist(), reference_eligible=not reasons,
            reference_rejection=reasons, physical_instance_verified=False))
        allpoints.append(cloud); allrgb.append(rgb); allids.append(np.full(len(cloud), candidate_id, 'i4'))
        for i in group:
            allobs.append(np.full(len(clouds[i]),i,'i4'))
            allframes.append(np.full(len(clouds[i]),observations[i]['frame'],'i4'))
            allpixels.append(pixel_clouds[i])
    points = np.concatenate(allpoints) if allpoints else np.empty((0, 3), 'f4')
    rgb = np.concatenate(allrgb) if allrgb else np.empty((0, 3), 'u1')
    ids = np.concatenate(allids) if allids else np.empty(0, 'i4')
    np.savez_compressed(output / 'candidates.npz', points=points, rgb=rgb, candidate_id=ids, association_scores=scores,
        observation_id=np.concatenate(allobs) if allobs else np.empty(0,"i4"),
        frame=np.concatenate(allframes) if allframes else np.empty(0,"i4"),
        pixel=np.concatenate(allpixels) if allpixels else np.empty(0,"i8"))
    from plyfile import PlyData, PlyElement
    vertex = np.empty(len(points), dtype=[('x','f4'), ('y','f4'), ('z','f4'), ('red','u1'), ('green','u1'), ('blue','u1'), ('candidate_id','i4')])
    for j, name in enumerate(['x','y','z']): vertex[name] = points[:,j]
    for j, name in enumerate(['red','green','blue']): vertex[name] = rgb[:,j]
    vertex['candidate_id'] = ids
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(output / 'candidates.ply'))
    save(output / 'observations.json', observations)
    save(output / 'projection_rejected.json', rejected)
    save(output / 'instances.json', instances)
    by_label = defaultdict(list)
    for r in observations: by_label[r['label']].append(r)
    save(output / 'classes.json', [dict(label=label, observations=len(rows),
        distinct_frames=sorted(set(r['frame'] for r in rows)),
        repeated_in_multiple_frames=len(set(r['frame'] for r in rows))>=2) for label,rows in sorted(by_label.items())])
    import csv
    with (output / 'sizes.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['candidate_id','label','short_horizontal','long_horizontal','height','distinct_frames','reference_eligible','units'])
        for r in instances: writer.writerow([r['candidate_id'],r['label'],*r['dimensions'],len(r['frames']),r['reference_eligible'],'input_mesh_units'])
    summary=dict(observations=len(observations), rejected=len(rejected), candidate_instances=len(instances),
        candidates_by_label=dict(Counter(r['label'] for r in instances)),
        multiframe_candidates=sum(len(r['frames'])>=2 for r in instances),
        reference_eligible=sum(r['reference_eligible'] for r in instances),
        physical_instances_verified=0, points=len(points), voxel=voxel, association_radius=radius,
        association_min_bidirectional_overlap=args.overlap, metric_scale_verified=coordinate.get('metric_scale_verified',False),
        units='input_mesh_units', source_run=scene['source_run'], coordinate_path=scene['coordinates'],
        coordinate_sha256=sha(scene['coordinates']), coordinate_scale_note=coordinate.get('coordinate_note'),
        geometry_reused=True, retained_source_index_verified=source_index is not None, other_segmentation_masks_used=False, shape_specific_rules=False,
        source_manifest_sha256=sha(Path(scene['source_run'])/'input/manifest.json'),
        seconds=time.perf_counter()-started)
    save(output/'projection_summary.json', summary)
    print('SUMMARY', summary, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--discovery', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--vendor', required=True)
    p.add_argument('--scene', default='dorm2')
    p.add_argument('--fov', type=float, default=100)
    p.add_argument('--min-points', type=int, default=64)
    p.add_argument('--overlap', type=float, default=.3)
    p.add_argument('--dynamic-labels', default='person')
    run(p.parse_args())
