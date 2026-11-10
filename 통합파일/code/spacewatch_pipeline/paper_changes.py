"""Current-plan original-RGB change stages using the accepted final detector.

No historical frame selection, registration, probabilities or event decisions are
imported. Geometry/semantic upstream reuse, when configured, remains explicitly
receipted by the pipeline. GPU inference is run by the outer worker's GPU lease.
"""
import copy
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image

from .io import digest, read, record, result, save, sha, verify

PHASES = ('before', 'after')
UPRIGHT = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                    [0., -1., 0., 0.], [0., 0., 0., 1.]])
PRODUCER = 'accepted final mesh_simsac snapshot; current plans/geometry/RGB'


def validate_options(config):
    if config.get('changes', {}).get('keypoint_nearby') is not True:
        raise ValueError('Paper change stages require changes.keypoint_nearby=true')
    captures = config.get('captures', [])
    if len(captures) != 2 or len({c['id'] for c in captures}) != 2:
        raise ValueError('Two distinct ordered before/after captures are required')
    if config['changes'].get('native_rgb', True) is not True:
        raise ValueError('Paper change inference requires native original video RGB')
    required = dict(size=224, registration='auto', regional_correction=True,
                    probability_threshold=.5, object_crops=True, fusion='mean',
                    max_refs=2, min_frames=2)
    settings = config['changes'].get('settings', {})
    for key, expected in required.items():
        if key in settings and settings[key] != expected:
            raise ValueError('Paper change setting differs: ' + key)
    threads = config.get('runtime', {}).get('cpu_threads', 1)
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('runtime.cpu_threads must be an integer from1to4')


def paper_settings(config):
    settings = dict(size=224, frame_stride=1, max_refs=2, min_frames=2,
                    registration='auto', regional_correction=True,
                    probability_threshold=.5, object_crops=True, fusion='mean')
    settings.update(config['changes'].get('settings', {}))
    settings['cpu_threads'] = config.get('runtime', {}).get('cpu_threads', 1)
    return settings


def _frame_key(frame):
    value = str(frame['id'])
    return (0, int(value)) if value.isdecimal() else (1, value)


def select_lowmask(frames, target, radius, count=3, min_gap_s=1.,
                   max_bystander_pixels=200, min_label_pixels=32):
    """Accepted final source-frame ranking; budgets never depend on known changes."""
    target = np.asarray(target, float)
    if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Finite target and positive radius required')
    candidates = []
    for frame in frames:
        distance = float(np.linalg.norm(np.asarray(frame['common_position']) - target))
        if not np.isfinite(distance) or distance > radius:
            continue
        labels = {name: pixels for name, pixels in frame['label_counts'].items()
                  if name not in {'wall', 'floor', 'ceiling', 'person'} and pixels >= min_label_pixels}
        candidates.append(dict(frame, distance_u=distance, label_type_count=len(labels),
                               visible_labels=sorted(labels), object_label_pixels=sum(labels.values())))
    candidates.sort(key=lambda f: (f['bystander_pixels'] > max_bystander_pixels,
        int(f['masked_fraction'] / .02), -f['label_type_count'], -f['object_label_pixels'],
        -f['sharpness'], f['distance_u'], _frame_key(f)))
    chosen = []
    for frame in candidates:
        if all(abs(frame['timestamp_s'] - other['timestamp_s']) >= min_gap_s for other in chosen):
            chosen.append(frame)
        if len(chosen) >= count:
            break
    for frame in candidates:
        if len(chosen) >= count:
            break
        if str(frame['id']) not in {str(other['id']) for other in chosen}:
            chosen.append(frame)
    return chosen


def extend_photographer(capturer, person, frequent_region):
    """Final worker's10% camera-fixed prior and4/3 pixel dilation policy."""
    from scipy.ndimage import binary_dilation
    capturer, person = np.asarray(capturer, bool), np.asarray(person, bool)
    if capturer.shape != person.shape or person.shape != frequent_region.shape:
        raise ValueError('Photographer mask grids disagree')
    return binary_dilation(capturer | (person & frequent_region), iterations=3)


def _catalog(folder, manifest):
    """Extend only current exported masks, and measure current source frame labels."""
    from scipy.ndimage import binary_dilation
    catalogs, mask_audits = [], []
    for phase in PHASES:
        side = manifest[phase]
        labels, frames = side['segmentation_labels'], side['frames']
        if not frames:
            raise ValueError('No original frames for ' + side['id'])
        person_class = labels.index('person') + 1 if 'person' in labels else None
        frequency = None
        for frame in frames:
            with np.load(folder / frame['segmentation'], allow_pickle=False) as values:
                person = values['class_id'] == person_class if person_class is not None else np.zeros(values['class_id'].shape, bool)
                if frequency is None:
                    frequency = np.zeros(person.shape, 'f4')
                if frequency.shape != person.shape:
                    raise ValueError('Semantic ERP dimensions changed within capture')
                frequency += person
        frequent_region = binary_dilation(frequency / len(frames) >= .1, iterations=4)
        records, before_pixels, after_pixels = [], 0, 0
        for frame in frames:
            path = folder / frame['segmentation']
            with np.load(path, allow_pickle=False) as values:
                arrays = dict(values)
            classes = arrays['class_id']
            person = classes == person_class if person_class is not None else np.zeros(classes.shape, bool)
            prior = arrays['person'].astype(bool)
            photographer = extend_photographer(prior, person, frequent_region)
            before_pixels += int(prior.sum()); after_pixels += int(photographer.sum())
            arrays['person'] = photographer
            arrays['instance_id'] = np.where(photographer, 0, arrays['instance_id']).astype(arrays['instance_id'].dtype)
            np.savez_compressed(path, **arrays)
            image = folder / frame['image']
            with Image.open(image) as im:
                size = im.size
                gray = np.asarray(im.convert('L'), dtype=float)
            Image.fromarray(photographer.astype('uint8')*255).resize(size, Image.Resampling.NEAREST).save(folder / frame['person_mask'])
            counts = np.bincount(classes[~photographer].ravel(), minlength=len(labels)+1)
            lap = -4*gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
            records.append(dict(frame,
                label_counts={name: int(counts[i+1]) for i, name in enumerate(labels)},
                sharpness=float(lap.var()), bystander_pixels=int((person & ~photographer).sum()),
                masked_fraction=float((person | photographer).mean()),
                source_image=str(image.resolve()), source_image_sha256=sha(image)))
        catalogs.append(records)
        mask_audits.append(dict(capture=side['id'], frames=len(frames),
            photographer_pixels_before=before_pixels, photographer_pixels_after=after_pixels,
            frequent_person_fraction=.1, frequent_region_dilation=4, mask_dilation=3,
            ordinary_people_retained=True, person_class_index=person_class))
    return catalogs, mask_audits


def select_current_plan(catalog, plans, registration, anchor_capture, work, regional=None):
    """Join arbitrary current selected viewpoints to their actual nearby cameras."""
    if regional is None:
        from tools.mesh_simsac.regional import FreshRegionalAlignment
        regional = FreshRegionalAlignment(registration['fresh_structural'])
    capture_ids = [plan['capture_id'] for plan in plans]
    if len(set(capture_ids)) != 2 or anchor_capture not in capture_ids:
        raise ValueError('Anchor capture not present in the current plans')
    anchor_epoch = capture_ids.index(anchor_capture)
    radius = .65 * registration['fresh_structural']['height_u']
    selected, rows = [{}, {}], []
    lookup = [{str(frame['id']): frame for frame in records} for records in catalog]
    if not plans[anchor_epoch]['viewpoints']:
        raise ValueError('The current anchor plan selected no viewpoints')
    for index, view in enumerate(plans[anchor_epoch]['viewpoints']):
        native = UPRIGHT @ np.asarray(view['reference_from_camera'], float)
        if native.shape != (4, 4) or not np.isfinite(native).all():
            raise ValueError('Current viewpoint requires finite reference_from_camera')
        region = int(regional.assign(native[None, :3, 3], anchor_epoch)[0])
        transform = np.asarray(registration['region_matrices'][anchor_epoch][region]) @ np.asarray(registration['common_from_world'][anchor_epoch])
        target = (transform @ native)[:3, 3]
        kid = f'{anchor_capture}-K{index+1:02d}'
        row = dict(id=kid, anchor_capture=anchor_capture, keypoint=index+1,
            viewpoint_id=view['viewpoint_id'], region=f'R{region:02d}',
            common_position=target.tolist(), native_position=np.asarray(view['reference_from_camera'])[:3, 3].tolist(),
            nearby_radius_u=radius,
            registration_supported=registration['fresh_structural']['regions'][region-1]['supported'], sides=[])
        for epoch, records in enumerate(catalog):
            matrix = np.asarray(registration['region_matrices'][epoch][region]) @ np.asarray(registration['common_from_world'][epoch])
            candidates = [dict(frame, common_position=(matrix @ np.asarray(frame['world_from_camera']))[:3, 3].tolist()) for frame in records]
            chosen = select_lowmask(candidates, target, radius)
            for frame in chosen:
                record = selected[epoch].setdefault(str(frame['id']), dict(lookup[epoch][str(frame['id'])], keypoint_ids=[]))
                record['keypoint_ids'].append(kid)
            row['sides'].append(dict(capture=capture_ids[epoch],
                status='selected' if len(chosen) >= 2 else 'insufficient_nearby_frames', frames=chosen,
                nearest_available_distance_u=min((float(np.linalg.norm(np.asarray(f['common_position'])-target)) for f in candidates), default=None)))
        rows.append(row)
    return dict(schema='keypoint-nearby-rgb-v1', work=str(work),
        selection='nearby; bystander-free first; least person/photographer mask (2% steps); object labels excluding person; prefer >=1s separation',
        frame_budget_per_capture_per_keypoint=3, nearby_radius_u=radius,
        anchor_count=len(rows), canonical_anchor_capture=anchor_capture,
        keypoints=rows, unique_frames=[len(items) for items in selected],
        selected_frames=[list(items.values()) for items in selected], min_gap_s=1., max_bystander_pixels=200,
        time_stride_used=False, known_change_ids_used=False)


def _require_plan_dependencies(job):
    missing = [capture['id']+'.plan' for capture in job['config']['captures']
               if capture['id']+'.plan' not in job['dependencies']]
    if missing:
        raise ValueError('Change inputs must wait for current selected plans: ' + ', '.join(missing))


def current_crop_importance(job, manifest):
    """Bind both current importance-stage receipts, not legacy static rankings.

    Five saved responses have already been identity-checked by paper_importance.
    Recheck the receipt, mean arithmetic and shared manual/class/model identity at
    the hand-off. Semantic correctness remains UNVERIFIED; no new model call.
    """
    from tools.mesh_simsac.objcrop import crop_importance
    mapping, identity, sources, captures = None, None, {}, []
    for capture in job['config']['captures']:
        cid = capture['id']
        dependency = job['dependencies'].get(cid + '.importance')
        if not dependency or not dependency.get('ratings') or not (dependency.get('cache') or dependency.get('importance_cache')):
            raise ValueError('Paper crop preparation requires current ' + cid + '.importance ratings/cache')
        ratings_path = Path(dependency['ratings']).resolve()
        cache_path = Path(dependency.get('cache') or dependency['importance_cache']).resolve()
        ratings, cache = read(ratings_path), read(cache_path)
        if cache.get('capture_id') != cid or verify(cache['rating_source']).resolve() != ratings_path:
            raise ValueError('Crop importance cache/ratings capture binding mismatch: ' + cid)
        if ratings.get('mode') != 'cached_response_reconstruction' or ratings.get('current_model_calls') != 0 or ratings.get('repeats') != 5:
            raise ValueError('Crop importance requires the verified five-response cache stage')
        current_identity = {key: ratings.get(key) for key in ('manual_hash', 'class_list_hash', 'model')}
        if not all(isinstance(value, str) and value for value in current_identity.values()):
            raise ValueError('Missing crop importance manual/class/model identity')
        if (cache.get('manual_hash') != current_identity['manual_hash'] or
                cache.get('model_requested') != current_identity['model'] or
                dependency.get('class_list_hash') != current_identity['class_list_hash']):
            raise ValueError('Crop importance stage identity mismatch: ' + cid)
        names = ratings.get('class_names')
        classes = ratings.get('classes', {})
        if not isinstance(names, list) or names != sorted(classes) or len(set(names)) != len(names) or digest(names) != current_identity['class_list_hash']:
            raise ValueError('Crop importance class-list identity mismatch')
        current = {}
        for name in names:
            row = classes[name]
            values = row.get('ratings', [])
            if len(values) != 5 or any(type(value) is not int or not 1 <= value <= 5 for value in values):
                raise ValueError('Crop importance must retain five integer 1--5 responses: ' + name)
            mean = sum(values) / 5
            if row.get('n') != 5 or row.get('mean') != mean or row.get('score') != mean / 5:
                raise ValueError('Crop importance mean arithmetic mismatch: ' + name)
            current[name] = mean
        for item in cache.get('scores', {}).values():
            name = item.get('class_name')
            if name not in current or item.get('class_mean') != current[name] or item.get('score') != current[name] / 5:
                raise ValueError('Current scene score differs from class crop mean: ' + str(name))
        if mapping is not None and (current != mapping or current_identity != identity):
            raise ValueError('Before/after crop importance manual/class/model bindings differ')
        mapping, identity = current, current_identity
        captures.append(cid)
        for path in (ratings_path, cache_path):
            sources[str(path)] = record(path)
    requested = job['config'].get('requested_inputs', {}).get('object_classes')
    if requested is not None and sorted(requested) != sorted(mapping):
        raise ValueError('Crop importance classes differ from current requested object classes')
    labels = sorted({name for phase in PHASES for name in manifest[phase].get('segmentation_labels', [])})
    mapping = crop_importance({'crop_importance_by_class': mapping}, labels)
    configured = job['config'].get('changes', {}).get('settings', {}).get('crop_importance_by_class')
    if configured is not None and configured != mapping:
        raise ValueError('Configured crop importance differs from current verified cache')
    return mapping, dict(mode='current_importance_stage_class_mean_binding', captures=captures,
        identity=identity, class_means=mapping, source_files=list(sources.values()),
        semantic_status='UNVERIFIED', new_model_calls=0, legacy_static_map_used=False,
        class_eligibility='All supplied nonstructural/nonperson classes; no fixture exclusion')


def registration_is_current(inputs):
    """Imported stages retain their old fit, not a fresh-fit claim for this run."""
    if 'registration_computed_in_current_pipeline' in inputs:
        return inputs['registration_computed_in_current_pipeline'] is True
    return inputs.get('fresh_registration') is True and not inputs.get('stage_reuse')


def rgb_inputs(job):
    from . import mesh_rgb
    from tools.mesh_simsac.prepare import prepare
    validate_options(job['config']); _require_plan_dependencies(job)
    out, config = Path(job['output']), job['config']
    catalog_job = copy.deepcopy(job)
    catalog_job['output'] = str(out / 'catalog')
    catalog_job['config']['changes'].pop('keypoint_nearby')
    catalog_job['config']['changes'].update(frame_stride=1, native_rgb=False, settings=paper_settings(config))
    (out / 'catalog').mkdir(exist_ok=False)
    mesh_rgb.rgb_inputs(catalog_job)
    folder = out / 'catalog/inputs'
    manifest = read(folder / 'manifest.json')
    catalog, masks = _catalog(folder, manifest)
    manifest['settings'] = paper_settings(config)
    save(folder / 'manifest.json', manifest)
    # The fitting manifest stays immutable even as the final one is narrowed to
    # current selected frames and rebound to decoded original-video pixels.
    fitting = folder / 'registration-inputs.json'
    save(fitting, manifest)
    fresh_root = out / 'fresh-registration'
    registration = prepare(fitting, fresh_root, capturer_checkpoint=None, registration_only=True)
    plans = [read(job['dependencies'][capture['id']+'.plan']['plan']) for capture in config['captures']]
    anchor = config['captures'][1]['id']
    plan = select_current_plan(catalog, plans, registration, anchor, out)
    plan['source_plan_sha256'] = {capture['id']: sha(job['dependencies'][capture['id']+'.plan']['plan']) for capture in config['captures']}
    plan['fresh_registration_sha256'] = sha(fresh_root / 'registration.json')
    save(out / 'keypoint-plan.json', plan)
    for epoch, phase in enumerate(PHASES):
        selected = {str(frame['id']): frame for frame in plan['selected_frames'][epoch]}
        if not selected:
            raise ValueError('No original cameras in current anchor neighbourhoods: ' + manifest[phase]['id'])
        manifest[phase]['frames'] = [dict(frame, keypoint_ids=selected[str(frame['id'])]['keypoint_ids'])
            for frame in manifest[phase]['frames'] if str(frame['id']) in selected]
    shutil.copy2(fresh_root / 'registration.json', folder / 'frozen-registration.json')
    manifest['frozen_structural_registration'] = dict(path='frozen-registration.json',
        sha256=sha(folder / 'frozen-registration.json'),
        input_hashes={manifest[phase]['id']: {kind: sha(folder / manifest[phase][kind]) for kind in ('mesh', 'pointcloud')} for phase in PHASES},
        computed_in_current_stage=True)
    manifest['settings'].update(registration='provided', frame_stride=1)
    manifest['keypoint_plan'] = dict(schema=plan['schema'], sha256=sha(out / 'keypoint-plan.json'),
        anchor_count=plan['anchor_count'], keypoint_ids=[row['id'] for row in plan['keypoints']],
        nearby_radius_u=plan['nearby_radius_u'], canonical_anchor_capture=anchor)
    manifest['scope'] = 'Current selected plans, freshly registered current geometry, actual nearby original-video RGB; no stored detector decisions'
    save(folder / 'manifest.json', manifest)
    mesh_rgb.materialize_native_frames(folder, out / 'catalog/adapter.json', config['runtime'])
    save(out / 'input-lineage.json', dict(producer=PRODUCER,
        config_sha256=digest(config), source_plan_sha256=plan['source_plan_sha256'],
        fresh_registration=str(fresh_root / 'registration.json'), fresh_registration_sha256=plan['fresh_registration_sha256'],
        extended_photographer_masks=masks, gaussian_inputs_used=False,
        old_predictions_imported=False, fresh_change_inference_required=True,
        native_original_rgb=True, known_change_ids_used=False, hardcoded_anchor_count=False))
    result(out, manifest=str(folder / 'manifest.json'), inputs_manifest=str(folder / 'manifest.json'),
        keypoint_plan=str(out / 'keypoint-plan.json'), registration=str(fresh_root / 'registration.json'),
        lineage=str(out / 'input-lineage.json'), source_mode='original_rgb',
        captures=[c['id'] for c in config['captures']], frames=plan['unique_frames'], keypoints=plan['anchor_count'],
        fresh_registration=True, fresh_frame_selection=True)


def rgb_prepare(job):
    from . import mesh_rgb
    out = Path(job['output'])
    inputs = job['dependencies']['pair.rgb_inputs']
    source_manifest = Path(inputs['manifest']).resolve()
    source_hash = sha(source_manifest)
    manifest = read(source_manifest)
    mapping, binding = current_crop_importance(job, manifest)
    folder = out / 'current-inputs'
    mesh_rgb.clone(source_manifest.parent, folder)
    current_manifest = folder / source_manifest.name
    manifest['settings']['crop_importance_by_class'] = mapping
    manifest['crop_importance_binding'] = binding
    manifest['crop_importance_binding']['source_input_manifest'] = record(source_manifest)
    save(current_manifest, manifest)
    save(out / 'crop-importance-binding.json', binding)
    prepared_job = dict(job, dependencies=dict(job['dependencies']))
    prepared_job['dependencies']['pair.rgb_inputs'] = dict(inputs, manifest=str(current_manifest))
    mesh_rgb.rgb_prepare(prepared_job)
    if sha(source_manifest) != source_hash:
        raise ValueError('Preparation changed its immutable predecessor manifest')
    value = read(out / 'result.json')
    value.update(inputs_manifest=str(current_manifest), manifest=str(current_manifest),
                 keypoint_plan=inputs['keypoint_plan'], crop_importance_binding=str(out / 'crop-importance-binding.json'),
                 registration_computed_in_current_pipeline=registration_is_current(inputs),
                 source_input_manifest=str(source_manifest), source_input_manifest_sha256=source_hash)
    save(out / 'result.json', value)


def bind_prepared_source_images(prepared_root, inputs_manifest, destination):
    """Privately bind prepared ERP metadata to exact immutable source RGB.

    Only manifest paths/provenance change. No array, projected RGB, pose, crop,
    pair or probability is recomputed. Native erp_infer accepts absolute paths.
    """
    from . import mesh_rgb
    from tools.mesh_simsac.bundle import local_file
    source = Path(prepared_root).resolve()
    input_path = Path(inputs_manifest).resolve()
    destination = Path(destination).resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError('Use a new private prepared-binding directory outside the source')
    source_manifest = source / 'manifest.json'
    source_hash, input_hash = sha(source_manifest), sha(input_path)
    prepared, inputs = read(source_manifest), read(input_path)
    if prepared.get('schema') != 'mesh-simsac-prepared-v1' or inputs.get('schema') != 'mesh-simsac-input-v1':
        raise ValueError('Prepared/current input schema mismatch')
    frames, source_images, captures = {}, {}, set()
    for epoch, phase in enumerate(PHASES):
        side = inputs[phase]
        capture = side['id']
        if not isinstance(capture, str) or not capture or capture in captures:
            raise ValueError('Distinct current input capture IDs are required')
        captures.add(capture)
        for frame in side['frames']:
            fid = str(frame['id']); key = (epoch, fid)
            if key in frames:
                raise ValueError('Duplicate current source frame ID')
            path = local_file(input_path.parent, frame['image'])
            expected = frame.get('sha256')
            if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
                raise ValueError('Current source frame requires a SHA256 receipt')
            stamp = frame.get('source_frame_index')
            if type(stamp) is not int or stamp < 0:
                raise ValueError('Current source frame requires its exact source_frame_index')
            previous = source_images.get(str(path))
            if previous is not None and previous['sha256'] != expected:
                raise ValueError('Conflicting current source-image receipts')
            if previous is None:
                if sha(path) != expected:
                    raise ValueError('Current source RGB checksum mismatch: ' + str(path))
                source_images[str(path)] = dict(path=str(path), sha256=expected, bytes=path.stat().st_size)
            frames[key] = dict(frame=frame, capture=capture, path=path, sha256=expected)
    seen, identities, bindings, assets = set(), [], [], []
    for view in prepared.get('views', []):
        vid = view.get('id')
        if not isinstance(vid, str) or re.fullmatch('[A-Za-z0-9_-]+', vid) is None or vid in seen:
            raise ValueError('Prepared view ID is missing, duplicate or unsafe')
        seen.add(vid)
        epoch = view.get('epoch')
        if type(epoch) is not int or epoch not in (0, 1):
            raise ValueError('Prepared view epoch must identify a current capture')
        row = frames.get((epoch, str(view.get('frame_id'))))
        if row is None or view.get('capture_id') != row['capture']:
            raise ValueError('Prepared view capture/frame is not in current inputs: ' + vid)
        frame = row['frame']
        if view.get('source_frame_index') != frame['source_frame_index']:
            raise ValueError('Prepared view source-frame index mismatch: ' + vid)
        old = view.get('source_image')
        if not isinstance(old, str) or not old:
            raise ValueError('Prepared view is missing its declared source image')
        if Path(old).is_absolute():
            matches = Path(old).resolve() == row['path']
        else:
            # The pre-inference prepared record must declare the exact frame's
            # bundle-relative image, not a path chosen by a different stage.
            matches = old == frame['image']
        if not matches:
            raise ValueError('Prepared source-image identity/path mismatch: ' + vid)
        if view.get('source_image_sha256', row['sha256']) != row['sha256']:
            raise ValueError('Prepared source-image SHA mismatch: ' + vid)
        identities.append({key: view.get(key) for key in ('id', 'geometry_sha256', 'rgb_sha256')})
        for suffix in ('.npz', '.jpg'):
            path = source / 'views' / (vid + suffix)
            if not path.is_file():
                raise ValueError('Prepared raster asset is missing: ' + str(path))
            stat = path.stat()
            assets.append((path, path.relative_to(source), stat.st_size, stat.st_mtime_ns))
        bindings.append(dict(id=vid, capture_id=row['capture'], frame_id=str(frame['id']),
            source_frame_index=frame['source_frame_index'], previous_source_image=old,
            source_image=str(row['path']), source_image_sha256=row['sha256']))
        view['source_image'] = str(row['path'])
        view['source_image_sha256'] = row['sha256']
    if not seen:
        raise ValueError('No prepared views to bind')
    mesh_rgb.clone(source, destination)
    hardlinks, copies = 0, 0
    for path, relative, size, mtime in assets:
        copied = destination / relative
        if path.stat().st_size != size or path.stat().st_mtime_ns != mtime or copied.stat().st_size != size:
            raise ValueError('Prepared raster asset changed during private clone')
        if os.path.samefile(path, copied):
            hardlinks += 1
        else:
            copies += 1  # clone's copy2 fallback; no array is opened or edited.
    preserved = [{key: view.get(key) for key in ('id', 'geometry_sha256', 'rgb_sha256')}
                 for view in prepared['views']]
    if preserved != identities or sha(source_manifest) != source_hash or sha(input_path) != input_hash:
        raise ValueError('Immutable prepared/current-input source changed during binding')
    receipt = dict(mode='absolute_current_source_rgb_path_binding_only',
        source_prepared_manifest=record(source_manifest), inputs_manifest=record(input_path),
        views=len(bindings), source_images=list(source_images.values()), bindings=bindings,
        geometry_and_projected_rgb_metadata_sha256=digest(identities),
        prepared_raster_assets=len(assets), hardlinked_raster_assets=hardlinks,
        copied_raster_assets=copies, raster_arrays_opened=False,
        geometry_recomputed=False, model_calls=0, source_manifests_unchanged=True)
    prepared.setdefault('source_hashes', {}).update({str(source_manifest):source_hash, str(input_path):input_hash})
    prepared['source_hashes'].update({path: item['sha256'] for path,item in source_images.items()})
    prepared['source_image_binding'] = dict(receipt='source-image-binding.json',
        inputs_manifest_sha256=input_hash, source_prepared_manifest_sha256=source_hash,
        views=len(bindings), original_frames=len(source_images),
        geometry_and_projected_rgb_metadata_sha256=receipt['geometry_and_projected_rgb_metadata_sha256'])
    save(destination / 'source-image-binding.json', receipt)
    save(destination / 'manifest.json', prepared)
    if sha(source_manifest) != source_hash:
        raise ValueError('Private image binding modified its source manifest')
    return dict(root=str(destination), inputs_manifest=str(input_path),
        source_image_binding=str(destination / 'source-image-binding.json'),
        bound_views=len(bindings), bound_original_frames=len(source_images))


def rgb_infer(job):
    from . import mesh_rgb
    out = Path(job['output'])
    prepared = job['dependencies']['pair.rgb_prepare']
    binding = bind_prepared_source_images(prepared['root'], prepared['inputs_manifest'], out / 'prepared-binding')
    infer_job = dict(job, dependencies=dict(job['dependencies']))
    infer_job['dependencies']['pair.rgb_prepare'] = dict(prepared, root=binding['root'])
    mesh_rgb.rgb_infer(infer_job)
    value = read(out / 'result.json')
    for key in ('inputs_manifest', 'keypoint_plan', 'crop_importance_binding', 'registration_computed_in_current_pipeline'):
        if key in prepared:
            value[key] = prepared[key]
    value.update(source_image_binding=binding['source_image_binding'],
        source_prepared_root=prepared['root'], bound_views=binding['bound_views'],
        bound_original_frames=binding['bound_original_frames'])
    save(out / 'result.json', value)


def rgb_finalize(job):
    from . import mesh_rgb
    from tools.mesh_simsac.verify_changes import verify
    inputs = job['dependencies']['pair.rgb_inputs']
    inference = job['dependencies'].get('pair.rgb_infer', {})
    prepared = job['dependencies'].get('pair.rgb_prepare', {})
    # Legacy already-running reference stages lack this new field. New strict
    # stages always propagate the cloned manifest through prepare and inference.
    current_manifest = inference.get('inputs_manifest', prepared.get('inputs_manifest', inputs['manifest']))
    final_job = dict(job, dependencies=dict(job['dependencies']))
    final_job['dependencies']['pair.rgb_inputs'] = dict(inputs, manifest=current_manifest)
    mesh_rgb.rgb_finalize(final_job)
    out = Path(job['output']); value = read(out / 'result.json')
    root = Path(value['viewer_dir']); verify(root)
    data = read(root / 'viewer-data.json')
    current_registration = inference.get('registration_computed_in_current_pipeline',
        prepared.get('registration_computed_in_current_pipeline', registration_is_current(inputs)))
    data['job_provenance'].update(current_selected_plan=True,
        fresh_inference=not bool(inference.get('stage_reuse')),
        fresh_inference_in_current_pipeline=not bool(inference.get('stage_reuse')),
        inference_stage_reuse=inference.get('stage_reuse'),
        inference_checkpoint_recovery=inference.get('checkpoint_recovery'),
        keypoint_plan_sha256=sha(inputs['keypoint_plan']),
        fresh_registration_in_current_pipeline=current_registration,
        registration_computed_in_current_pipeline=current_registration,
        registration_stage_reuse=inputs.get('stage_reuse'),
        crop_importance_binding=inference.get('crop_importance_binding', prepared.get('crop_importance_binding')),
        detector_event_replay=False, known_change_ids_used=False)
    save(root / 'viewer-data.json', data)
    value.update(root=str(root), viewer_data=str(root / 'viewer-data.json'),
        viewer_manifest=str(root / 'viewer-data.json'),
        verified_changes=str(root / 'results/verified-changes.json'),
        verified_changes_path=str(root / 'results/verified-changes.json'),
        verified_change_count=len(data['verified_changes']), counts=data['counts'],
        inputs_manifest=current_manifest, manifest=current_manifest,
        keypoint_plan=inputs['keypoint_plan'], registration=str(root / 'registration.json'),
        registration_computed_in_current_pipeline=current_registration,
        crop_importance_binding=inference.get('crop_importance_binding', prepared.get('crop_importance_binding')),
        producer=PRODUCER, detector_event_replay=False)
    save(out / 'result.json', value)


def dispatch(job):
    """Outer engine owns DAG, deadlines, fresh stage directory and GPU lease."""
    action = job['action']
    if action not in {'rgb_inputs', 'rgb_prepare', 'rgb_infer', 'rgb_finalize'}:
        return False
    if job['config'].get('changes', {}).get('keypoint_nearby') is not True:
        return False
    validate_options(job['config'])
    threads = job['config'].get('runtime', {}).get('cpu_threads', 1)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    # Existing third-party geometry helpers may request several worker threads;
    # cap this isolated stage process's actual available CPUs as well.
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:threads]))
    started = time.time()
    try:
        globals()[action](job)
    except BaseException as error:
        save(Path(job['output']) / 'paper-change-stage.json', dict(action=action, status='failed',
            error=str(error), seconds=time.time()-started, producer=PRODUCER, config_sha256=digest(job['config'])))
        raise
    save(Path(job['output']) / 'paper-change-stage.json', dict(action=action, status='complete',
        seconds=time.time()-started, producer=PRODUCER, config_sha256=digest(job['config']),
        source_sha256=sha(__file__), cpu_threads=threads, detector_event_replay=False))
    return True
