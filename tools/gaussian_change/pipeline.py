"""One-command mesh + Gaussian change pipeline. No reconstruction, training or planning."""
import argparse
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from tools.mesh_simsac.core import write, sha, camera_transform, transform, read
from .inputs import load_manifest
from .registration import load_mesh, register, assign
from .fusion import fuse


def run(manifest, output, runtime, *, validate_only=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new/empty output directory; inputs and previous results are immutable')
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stages = []
    def progress(stage, **details):
        state = {'state': 'running', 'stage': stage, 'elapsed_s': time.time()-started, **details}
        write(output/'status.json', state)
        print(stage, details, flush=True)
    try:
        progress('input_validation')
        document, sides, settings, hashes = load_manifest(manifest)
        input_receipt = {'schema': document['schema'], 'hashes': hashes, 'units': document['units'],
                         'settings': settings, 'camera_counts': [len(s['views']) for s in sides],
                         'input_files_modified': False, 'prior_detections_consumed': False}
        write(output/'inputs.json', input_receipt)
        if validate_only:
            write(output/'status.json', {'state': 'validated', 'gpu_executed': False, 'stage': 'input_validation'})
            return output/'inputs.json'
        # Administrator-controlled runtime contains model/runtime paths; never input checkpoints.
        for key in ['simsac_source', 'simsac_checkpoint', 'person_checkpoint']:
            if not Path(runtime[key]).exists():
                raise FileNotFoundError(runtime[key])
        if runtime.get('person_deps'):
            sys.path.insert(0, runtime['person_deps'])
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for gsplat and the supplied SimSaC backend')
        from .plyio import load as load_gaussian
        from .render import Renderer, pano
        from tools.mesh_simsac.model import SimSaC
        from ultralytics import YOLO
        import cv2
        progress('mesh_sampling')
        meshes = [load_mesh(s, settings['samples'], 20261027+i) for i, s in enumerate(sides)]
        registration = register(meshes, settings, progress)
        registration['input_meshes'] = [{k: m[k] for k in ['vertices', 'triangles', 'structure_source']} for m in meshes]
        write(output/'registration.json', registration)
        stages += ['global_registration', 'regional_correction']
        h = registration['height_u']; centers = registration['centers']
        global_after = np.array(registration['global_after_to_before'])
        regions = registration['regions']
        matrices = [np.eye(4)[None], np.array([r['matrix_after_to_common'] for r in regions])]
        progress('gaussian_loading')
        renderers = []
        for epoch, side in enumerate(sides):
            gaussian = load_gaussian(side['gaussian'])
            bridge = side['world_from_gaussian']
            labels = assign(transform(transform(gaussian['xyz'], bridge), global_after), centers) if epoch else np.zeros(len(gaussian['xyz']), 'i4')
            renderers.append(Renderer(gaussian, labels, np.array([m @ bridge for m in matrices[epoch]])))
        views = []
        for epoch, side in enumerate(sides):
            rows = side['views']
            limit = settings['max_views_per_capture']
            if limit is not None and len(rows) > limit:
                rows = [rows[i] for i in np.unique(np.linspace(0, len(rows)-1, limit).astype(int))]
            for row in rows:
                pose = row['pose']
                zone = int(assign(transform(pose[None, :3, 3], global_after) if epoch else pose[None, :3, 3], centers)[0])
                common = camera_transform(pose, matrices[epoch][zone if epoch else 0])
                views.append({'id': row['id'], 'source_epoch': epoch, 'source_index': row['source_index'],
                              'camera_to_common_cv': common.tolist(), 'camera_region': regions[zone]['id']})
        write(output/'cameras.json', {'views': views, 'same_pose_for_both_epochs': True,
                                     'existing_change_candidates_used_for_view_selection': False,
                                     'subset_run': settings['max_views_per_capture'] is not None})
        progress('model_loading')
        model = SimSaC(runtime['simsac_source'], runtime['simsac_checkpoint'])
        model.audit.update(input_source='Fresh same-pose Gaussian RGB renders', gaussian_inputs_used=True,
                           depth_source='Gaussian expected radial depth, not sensor depth',
                           inference_mode=settings['inference_mode'],full_panorama_resized_for_model=settings['inference_mode']=='panorama',
                           panorama_size=[settings['width'],settings['height']],cube_face_size=settings['cube_size'])
        detector = YOLO(runtime['person_checkpoint'])
        model.audit['person_checkpoint_sha256'] = sha(runtime['person_checkpoint'])
        write(output/'model.json', model.audit)
        all_samples = []; observations = []; control = None
        supported_regions = np.array([r['supported'] for r in regions])
        for index, view in enumerate(views):
            progress('render_and_infer', completed=index, total=len(views), view_id=view['id'])
            pose = np.array(view['camera_to_common_cv'])
            directory = output/'renders'/f'V{index+1:05d}'
            directory.mkdir(parents=True)
            directional_audit=None;tile_control=None
            if settings['inference_mode']=='perspective_tiles':
                from .directional import render_pair
                pair,probabilities,directional_audit,tile_control=render_pair(renderers,pose,detector,model,settings['width'],settings['height'],settings['cube_size'],directory,
                    settings['perspective_tile_size'],settings['perspective_tile_overlap'],control is None,
                    lambda face:progress('render_and_infer',completed=index,total=len(views),view_id=view['id'],perspective_face=face+1))
            else:
                pair = [pano(r, pose, detector, settings['width'], settings['height'], settings['cube_size']) for r in renderers]
                probabilities = [model.predict(pair[e]['rgb'], pair[1-e]['rgb']) for e in [0, 1]]
            valid = np.ones((settings['height'], settings['width']), bool)
            for phase, p in zip(['before', 'after'], pair):
                cv2.imwrite(str(directory/(phase+'.png')), p['rgb'][..., ::-1])
                np.savez_compressed(directory/(phase+'-geometry.npz'), depth=p['depth'], alpha=p['alpha'], person=p['person'], camera_to_common_cv=pose)
                valid &= np.isfinite(p['depth']) & (p['depth'] > .001*h) & (p['alpha'] >= settings['alpha_threshold'])
            people = pair[0]['person'] | pair[1]['person']
            if control is None:
                cp = tile_control.pop('probabilities') if tile_control else model.predict(pair[0]['rgb'], pair[0]['rgb'])
                control = {'pixels_over_threshold': int((cp > settings['probability_threshold']).sum()),
                           'max_probability': float(cp.max()), 'compared_image_sha256': sha(directory/'before.png'),
                           'input_kind':'native perspective tile' if tile_control else 'whole panorama',
                           'tile':tile_control}
                write(output/'self-control.json', control)
            record = {'id': view['id'], 'directory': directory.relative_to(output).as_posix(),
                      'valid_overlap_pixels': int(valid.sum()), 'person_union_pixels': int(people.sum()), 'phases': [],
                      'inference_mode':settings['inference_mode'],'tiles_per_phase':directional_audit['tiles_per_phase'] if directional_audit else 1}
            for epoch, p in enumerate(pair):
                raw = probabilities[epoch] > settings['probability_threshold']
                eligible=raw&valid&~people
                points=pose[:3,3]+(p['rays'][eligible]@pose[:3,:3].T)*p['depth'][eligible,None]
                supported=supported_regions[assign(points,centers)] if len(points) else np.zeros(0,bool)
                mask=eligible.copy();mask[eligible]=supported;q=points[supported]
                # Bound fusion memory; persisted masks/depth retain every original selected pixel.
                stride = max(1, int(np.ceil(len(q)/20000)))
                if len(q):
                    all_samples.append({'points': q[::stride], 'phase': epoch, 'view': index})
                cv2.imwrite(str(directory/(['before', 'after'][epoch]+'-change.png')), mask.astype('u1')*255)
                record['phases'].append({'raw_pixels': int(raw.sum()), 'valid_changed_pixels': int((raw & valid).sum()),
                                         'removed_person_pixels': int((raw & valid & people).sum()),
                                         'removed_unsupported_region_pixels': int((~supported).sum()),
                                         'selected_pixels': int(mask.sum()), 'fusion_sample_stride': stride})
            np.savez_compressed(directory/'inference.npz', before_probability=probabilities[0], after_probability=probabilities[1], common_valid=valid, person_union=people)
            record['rgb_sha256'] = {p: sha(directory/(p+'.png')) for p in ['before', 'after']}
            observations.append(record)
        stages += ['same_pose_gaussian_rgb_depth_alpha', 'bidirectional_simsac', 'person_and_observability_filter']
        progress('fusion_3d')
        candidates, points, labels = fuse(all_samples, .04*h, settings['min_component_voxels'])
        np.savez_compressed(output/'changes-3d.npz', points=points, candidate_index=labels)
        result = {'schema': 'gaussian-mesh-changes-v1', 'state': 'complete', 'input_hashes': hashes,
                  'units': document['units'], 'coordinate_frame': 'before.world', 'candidates': candidates,
                  'observations': observations, 'cameras': 'cameras.json', 'registration': 'registration.json',
                  'points': 'changes-3d.npz', 'model': 'model.json', 'gpu_executed': True,
                  'scope': '3D change candidates; lifecycle classification is not inferred from RGB alone',
                  'registration_supported_regions': int(supported_regions.sum()), 'region_count': len(regions),
                  'comparison_support': 'supported' if supported_regions.any() else 'insufficient_registration_support',
                  'limited_view_run': settings['max_views_per_capture'] is not None,
                  'render_depth': 'Gaussian expected radial depth', 'accuracy_ground_truth_evaluated': False}
        result['render_size']=[settings['width'],settings['height']]
        result['inference_mode']=settings['inference_mode']
        write(output/'changes.json', result)
        progress('object_classification')
        from .lifecycle import classify
        events = classify(output, sides, {'model':model,'gaussian_models':[r.model for r in renderers]})
        result['object_events'] = 'object-events.json'
        result['object_event_counts'] = events['counts']
        write(output/'changes.json', result)
        stages.append('automatic_object_classification')
        stages.append('save_3d_changes')
        write(output/'status.json', {'state': 'complete', 'stage': 'save_3d_changes', 'gpu_executed': True,
                                     'stages_executed': stages, 'result': 'changes.json',
                                     'elapsed_s': time.time()-started, 'candidate_count': len(candidates)})
        return output/'changes.json'
    except BaseException as exc:
        write(output/'status.json', {'state': 'failed', 'message': str(exc), 'completed_stages': stages,
                                     'elapsed_s': time.time()-started})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args(argv)
    runtime = read(args.runtime) if args.runtime else {}
    if runtime.get('remote_session') and not args.validate_only:
        from .remote import run_remote
        return run_remote(args.manifest, args.output, runtime)
    print(run(args.manifest, args.output, runtime, validate_only=args.validate_only))


if __name__ == '__main__':
    main()
