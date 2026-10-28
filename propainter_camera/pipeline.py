"""Camera-frame selection -> SAM2/ProPainter -> mask-only ERP -> viewer.

CPU preparation/export runs in the configured environment. Model work is delegated
to launch.py, which leases exactly one idle GPU on the configured server.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rotation(projection):
    yaw, pitch = np.deg2rad([projection['yaw_deg'], projection['pitch_down_deg']])
    return np.array([[np.cos(yaw), -np.sin(yaw)*np.sin(pitch), np.sin(yaw)*np.cos(pitch)],
                     [0, np.cos(pitch), np.sin(pitch)],
                     [-np.sin(yaw), -np.cos(yaw)*np.sin(pitch), np.cos(yaw)*np.cos(pitch)]])


def cache_matches(folder, cfg, clip, source_ids, manifest):
    try:
        protocol = read(folder / 'protocol.json')
        report = read(folder / 'propainter_report.json')
        tracking = read(folder / 'tracking_report.json')
        if report['status'] != 'complete' or tracking['status'] != 'complete':
            return False
        if protocol['video_indices'] != clip['video_indices']:
            return False
        if protocol['projection'] != clip['projection'] or tracking['prompt'] != clip['prompt']:
            return False
        if tracking['dilation_radius_pixels'] != cfg['mask']['dilation_radius_pixels']:
            return False
        if tracking['close_kernel_pixels'] != cfg['mask']['close_kernel_pixels']:
            return False
        if report['official_commit'] != cfg['propainter']['official_commit']:
            return False
        command = report['command']
        for key in ('ref_stride', 'neighbor_length', 'subvideo_length', 'mask_dilation', 'raft_iter'):
            flag = '--' + key
            actual = int(command[command.index(flag)+1]) if flag in command else {'raft_iter': 20}.get(key)
            if actual != cfg['propainter'][key]:
                return False
        if ('--fp16' in command) != cfg['propainter']['fp16']:
            return False
        for flag, expected in zip(('--width', '--height'), clip['projection']['size']):
            if int(command[command.index(flag)+1]) != expected:
                return False
        for index in range(len(clip['video_indices'])):
            for path in (folder/'frames'/f'{index:05d}.png', folder/'masks'/f'{index:05d}.png',
                         folder/'propainter_results/frames/frames'/f'{index:04d}.png'):
                if not path.is_file():
                    return False
        selected_indices = {clip['video_indices'].index(manifest['frames'][i]['source_frame_index']) for i in source_ids}
        return all(not row['mask_touches_border'] for row in tracking['rows'] if row['frame'] in selected_indices)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def build_plan(cfg, output):
    manifest = read(cfg['source_manifest'])
    poses = np.load(cfg['camera_poses'])
    ids = cfg['selected_camera_ids']
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(manifest['frames']) for i in ids):
        raise ValueError('selected_camera_ids must contain unique valid camera indices.')
    if poses.shape != (len(manifest['frames']), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Camera poses do not match the source manifest.')
    width, height = cfg['erp_resolution']
    if width != height*2 or height < 2:
        raise ValueError('erp_resolution must be a positive 2:1 panorama size.')
    if cfg['propainter']['mask_dilation'] != 0:
        raise ValueError('Use mask.dilation_radius_pixels to expand the mask; mask_dilation must stay 0 so ERP blending uses the exact inference mask.')
    if cfg['propainter']['neighbor_length'] < 2 or cfg['propainter']['ref_stride'] < 1 or cfg['context']['sample_step'] < 1:
        raise ValueError('Invalid temporal sampling parameters.')
    if cfg['mask']['close_kernel_pixels'] < 1 or cfg['mask']['close_kernel_pixels'] % 2 == 0 or cfg['mask']['dilation_radius_pixels'] < 0:
        raise ValueError('Mask close kernel must be positive and odd; dilation radius must be nonnegative.')
    templates = []
    for cache in cfg['reusable_clips']:
        folder = Path(cache)
        protocol, tracking = read(folder/'protocol.json'), read(folder/'tracking_report.json')
        templates.append(dict(id=protocol['id'], video_indices=protocol['video_indices'],
                              projection=protocol['projection'], prompt=tracking['prompt'],
                              cache_dir=str(folder), source_ids=[]))
    clips, targets = {}, []
    for rank, camera_id in enumerate(ids, 1):
        frame = manifest['frames'][camera_id]
        native = frame['source_frame_index']
        candidates = [clip for clip in templates if native in clip['video_indices']]
        if candidates:
            clip = min(candidates, key=lambda item: (len(item['video_indices']), item['id'])).copy()
        else:
            setting = cfg.get('new_clip_settings', {}).get(str(camera_id))
            if setting is None:
                raise ValueError(f'Camera {camera_id} is outside the reviewed clips. Add new_clip_settings.{camera_id} with projection and a person prompt for its first frame.')
            fps, step = manifest['source_fps'], cfg['context']['sample_step']
            before = int(round(cfg['context']['seconds_before'] * fps / step))
            after = int(round(cfg['context']['seconds_after'] * fps / step))
            indices = list(range(native-before*step, native+after*step+1, step))
            indices = [i for i in indices if 0 <= i < manifest['source_frame_count']]
            projection = dict(setting['projection'])
            projection['rotation'] = rotation(projection).tolist()
            clip = dict(id=f'camera_{camera_id:05d}', video_indices=indices, projection=projection,
                        prompt=setting['prompt'], cache_dir=None, source_ids=[])
        clips.setdefault(clip['id'], clip)['source_ids'].append(camera_id)
        targets.append(dict(id=f'source_{camera_id:05d}', rank=rank, camera_frame_id=camera_id,
                            native_video_frame=native, timestamp_s=frame['timestamp_s'],
                            pose=poses[camera_id].tolist(), clip=clip['id']))
    for clip in clips.values():
        projection = clip['projection']
        size = projection['size']
        if len(size) != 2 or any(v < 8 or v % 8 for v in size) or not 0 < projection['horizontal_and_vertical_fov_deg'] < 180:
            raise ValueError('Projection dimensions must be multiples of 8; FOV must be between 0 and 180.')
        clip['frame_count'] = len(clip['video_indices'])
        clip['time_range_s'] = [i/manifest['source_fps'] for i in (clip['video_indices'][0],clip['video_indices'][-1])]
        local = output/'clips'/clip['id']
        cache = Path(clip['cache_dir']) if clip['cache_dir'] else None
        usable = next((folder for folder in [local, cache] if folder is not None and cache_matches(folder, cfg, clip, clip['source_ids'], manifest)), None)
        clip['reused'] = usable is not None
        clip['directory'] = str(usable or local)
    return dict(schema_version=1, pipeline='camera_frame_propainter_erp',
                selection='Explicit captured-camera indices; output pose equals the captured pose.',
                source_video=cfg['source_video'], source_manifest=cfg['source_manifest'],
                source_sha256=cfg['source_sha256'], camera_poses_sha256=sha(cfg['camera_poses']),
                erp_resolution=cfg['erp_resolution'], source_fps=manifest['source_fps'],
                pixel_grid='endpoint ERP', targets=targets, clips=list(clips.values()))


def prepare_new_clips(plan):
    cap = cv2.VideoCapture(plan['source_video'])
    if not cap.isOpened():
        raise RuntimeError('Could not open source video.')
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        for clip in plan['clips']:
            if clip['reused']:
                continue
            folder = Path(clip['directory'])
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Incomplete or incompatible previous clip: {folder}. Choose a new output directory.')
            started = time.perf_counter()
            for sub in ('frames','sam_frames','raw_masks','masks'):
                (folder/sub).mkdir(parents=True, exist_ok=True)
            projection = clip['projection']
            width, height = projection['size']
            tangent = math.tan(math.radians(projection['horizontal_and_vertical_fov_deg']/2))
            xx, yy = np.meshgrid(((np.arange(width)+.5)/width*2-1)*tangent,
                                 ((np.arange(height)+.5)/height*2-1)*tangent)
            rays = np.stack([xx,yy,np.ones_like(xx)],-1) @ np.asarray(projection['rotation']).T
            rays /= np.linalg.norm(rays,axis=-1,keepdims=True)
            mx = ((np.arctan2(rays[...,0],rays[...,2])/(2*np.pi)+.5)*sw-.5).astype('float32')
            my = np.clip((np.arcsin(rays[...,1])/np.pi+.5)*sh-.5,0,sh-1).astype('float32')
            wanted = {native:index for index,native in enumerate(clip['video_indices'])}
            cap.set(cv2.CAP_PROP_POS_FRAMES,clip['video_indices'][0])
            for native in range(clip['video_indices'][0],clip['video_indices'][-1]+1):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f'Could not decode frame {native}.')
                if native not in wanted:
                    continue
                index = wanted[native]
                patch = cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
                if not cv2.imwrite(str(folder/'frames'/f'{index:05d}.png'),patch):
                    raise IOError('Cannot save projected frame.')
                cv2.imwrite(str(folder/'sam_frames'/f'{index:05d}.jpg'),patch,[cv2.IMWRITE_JPEG_QUALITY,98])
            save(folder/'protocol.json',dict(**clip,elapsed_seconds=time.perf_counter()-started))
    finally:
        cap.release()


def export_erp(cfg, plan, output):
    started = time.perf_counter()
    width, height = plan['erp_resolution']
    lon, lat = np.meshgrid((np.arange(width)/(width-1)-.5)*2*np.pi,(np.arange(height)/(height-1)-.5)*np.pi)
    rays = np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
    cap = cv2.VideoCapture(plan['source_video'])
    if not cap.isOpened():
        raise RuntimeError('Could not open source video.')
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ex, ey = np.meshgrid(np.linspace(-.5,sw-.5,width),np.linspace(-.5,sh-.5,height))
    ex, ey = ex.astype('float32'),np.clip(ey,0,sh-1).astype('float32')
    for name in ('images','originals','masks','records'):
        (output/name).mkdir(parents=True,exist_ok=True)
    items = []
    try:
        for target in plan['targets']:
            clip = next(c for c in plan['clips'] if c['id']==target['clip'])
            folder = Path(clip['directory'])
            projection = clip['projection']
            index = clip['video_indices'].index(target['native_video_frame'])
            patch = cv2.imread(str(folder/'propainter_results/frames/frames'/f'{index:04d}.png'))
            mask = cv2.imread(str(folder/'masks'/f'{index:05d}.png'),0)
            pw, ph = projection['size']
            if patch is None or mask is None or patch.shape!=(ph,pw,3) or mask.shape!=(ph,pw) or not mask.any():
                raise ValueError('Invalid restoration patch or mask.')
            if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
                raise ValueError(f'Person mask touches projection edge: {target["id"]}. Adjust projection before exporting.')
            cap.set(cv2.CAP_PROP_POS_FRAMES,target['native_video_frame'])
            ok, native = cap.read()
            if not ok:
                raise RuntimeError('Could not decode selected frame.')
            original = cv2.remap(native,ex,ey,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
            camera = rays @ np.asarray(projection['rotation'])
            z = camera[...,2]
            tangent = math.tan(math.radians(projection['horizontal_and_vertical_fov_deg']/2))
            vx,vy = camera[...,0]/np.maximum(z,1e-6),camera[...,1]/np.maximum(z,1e-6)
            visible = (z>0)&(abs(vx)<=tangent)&(abs(vy)<=tangent)
            mx,my = ((vx/tangent+1)*pw/2-.5).astype('float32'),((vy/tangent+1)*ph/2-.5).astype('float32')
            sampled = cv2.remap(patch,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            region = (cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0)&visible
            feather = cfg['erp_feather_pixels']
            alpha = np.clip(cv2.distanceTransform(region.astype('uint8'),cv2.DIST_L2,3)/feather,0,1) if feather>0 else region.astype('float32')
            completed = np.rint(original*(1-alpha[...,None])+sampled*alpha[...,None]).clip(0,255).astype('uint8')
            if not np.array_equal(original[~region],completed[~region]):
                raise AssertionError('Pixels outside the mask changed.')
            for directory,image in [('images',completed),('originals',original),('masks',region.astype('uint8')*255)]:
                if not cv2.imwrite(str(output/directory/(target['id']+'.png')),image):
                    raise IOError('Could not save ERP image.')
            dest = output/'records'/clip['id']
            dest.mkdir(exist_ok=True)
            for name in ('protocol.json','tracking_report.json','propainter_report.json','propagation_report.json'):
                if (folder/name).is_file():
                    shutil.copyfile(folder/name,dest/name)
            focus = rays[region].mean(axis=0)
            focus /= np.linalg.norm(focus)
            items.append(dict(**target,file='images/'+target['id']+'.png',
                              original='originals/'+target['id']+'.png',width=width,height=height,
                              focus=dict(yaw=float(np.arctan2(focus[0],focus[2])),pitch=float(np.arcsin(focus[1]))),
                              edited_fraction=float(region.mean()),outside_mask_pixels_unchanged=True,
                              output_sha256=sha(output/'images'/(target['id']+'.png')),
                              patch_sha256=sha(folder/'propainter_results/frames/frames'/f'{index:04d}.png'),
                              reused_model_result=clip['reused']))
    finally:
        cap.release()
    save(output/'records/selection.json',dict(items=items,selection=plan['selection'],pixel_grid=plan['pixel_grid']))
    (output/'data.js').write_text('window.CAMERA_DATA='+json.dumps(items,ensure_ascii=False)+';\n')
    for name in ('index.html','viewer.js'):
        shutil.copyfile(ROOT/'viewer'/name,output/name)
    return items,time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--reuse-only',action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    output = (args.output or Path(cfg['output_dir'])).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'run_complete.json').exists():
        raise RuntimeError('This output already contains a completed run. Use --output with a new directory.')
    started = time.perf_counter()
    plan = build_plan(cfg,output)
    save(output/'plan.json',plan)
    save(output/'effective_config.json',cfg)
    print(json.dumps({'targets':[t['camera_frame_id'] for t in plan['targets']],
                      'reused_clips':[c['id'] for c in plan['clips'] if c['reused']],
                      'new_clips':[c['id'] for c in plan['clips'] if not c['reused']]}),flush=True)
    if args.plan_only:
        return
    if args.reuse_only and any(not c['reused'] for c in plan['clips']):
        raise RuntimeError('--reuse-only requested but matching cached outputs are missing.')
    if sha(plan['source_video']) != cfg['source_sha256']:
        raise RuntimeError('Source video hash differs from the reviewed input.')
    prepare_new_clips(plan)
    if any(not c['reused'] for c in plan['clips']):
        subprocess.run(['python3',str(ROOT/'launch.py'),str(output)],check=True)
    items, export_seconds = export_erp(cfg,plan,output)
    fresh = [c for c in plan['clips'] if not c['reused']]
    sam_seconds = sum(read(Path(c['directory'])/'tracking_report.json')['elapsed_seconds'] for c in fresh)
    pp_seconds = sum(read(Path(c['directory'])/'propainter_report.json')['elapsed_seconds'] for c in fresh)
    save(output/'run_complete.json',dict(completed=True,pipeline=plan['pipeline'],
          target_count=len(items),camera_ids=cfg['selected_camera_ids'],output_resolution=cfg['erp_resolution'],
          new_sam_seconds=sam_seconds,new_propainter_seconds=pp_seconds,erp_and_viewer_seconds=export_seconds,
          total_seconds=time.perf_counter()-started,reused_clips=len(plan['clips'])-len(fresh),
          new_clips=len(fresh),completed_unix=time.time()))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    main()
