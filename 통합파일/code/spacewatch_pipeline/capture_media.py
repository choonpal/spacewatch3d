"""Source-bound real ERP frames. No neural renderer, pose snapping or upscaling."""
from pathlib import Path
import time
import numpy as np
from .io import read,save,result,sha
from .geometry import context,similarity

def original_mode(config):
    return config.get('media', {}).get('mode') == 'original_erp'

def frame_samples(positions,spacing):
    points=np.asarray(positions,float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Capture positions must be finite Nx3')
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Capture spacing must be positive')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    targets=np.arange(0.,arc[-1],spacing)
    chosen={0,len(points)-1}
    for distance in targets:
        i=int(np.searchsorted(arc,distance,side='left'))
        choices=[j for j in [i-1,i] if 0<=j<len(points)]
        chosen.add(min(choices,key=lambda j:(abs(arc[j]-distance),j)))
    return sorted(chosen),arc

def trajectory(job):
    if job['config'].get('media',{}).get('sampling')=='quality_on_existing_path':
        from .existing_path import trajectory as from_existing
        return from_existing(job)
    c,out,deps,prefix=context(job)
    source=deps[prefix+'.reconstruct'];root=Path(source['source_run'])
    pose_path=root/'result/camera_poses.json';manifest_path=root/'input/manifest.json'
    poses=read(pose_path);manifest=read(manifest_path);frames=manifest['frames']
    if len(poses)!=len(frames) or not frames:raise ValueError('Source frame/pose count mismatch')
    video=Path(c['video']).resolve()
    if sha(video)!=manifest['source_sha256']:raise ValueError('Original video identity mismatch')
    tf,scale=similarity(read(source['coordinates'])['source_to_mesh'])
    rows=[]
    temporal=job['config'].get('media',{}).get('sampling')=='quality_temporal'
    if temporal and manifest.get('frame_sampling')!='quality_temporal':
        raise ValueError('Temporal-quality mode requires its own reconstructed frame manifest, not old poses')
    for i,(p,f) in enumerate(zip(poses,frames)):
        if p['frame']!=i or abs(p['timestamp_s']-f['timestamp_s'])>1e-6:
            raise ValueError('Reconstruction pose does not match this source frame time')
        pose=np.asarray(p['camera_to_world'],float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
            raise ValueError('Invalid captured pose')
        matrix=np.eye(4);matrix[:3,:3]=tf[:3,:3]@pose[:3,:3]/scale
        matrix[:3,3]=tf[:3,:3]@pose[:3,3]+tf[:3,3]
        if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(matrix[:3,:3])<.9999:
            raise ValueError('Captured pose orientation must remain rigid')
        index=f.get('source_frame_index');stamp=f.get('timestamp_s')
        if type(index) is not int or index<0 or not np.isfinite(stamp) or stamp<0:
            raise ValueError('Exact source frame index and timestamp required')
        if rows and (index<=rows[-1]['source_frame_index'] or stamp<=rows[-1]['timestamp_s']):
            raise ValueError('Capture frame order must be strictly chronological')
        rows.append(dict(pose_index=i,source_frame_index=index,timestamp_s=stamp,
                         reference_from_camera=matrix.tolist()))
        if temporal:
            from fractions import Fraction
            if type(f.get('pts')) is not int or not f.get('time_base'):
                raise ValueError('Source PTS/time_base required for each new candidate')
            if abs(float(Fraction(f['pts'])*Fraction(f['time_base']))-stamp)>1e-12:
                raise ValueError('Timestamp does not match source PTS')
            rows[-1].update(pts=f['pts'],time_base=f['time_base'])
    from .planning_policy import effective_planning_config
    options=effective_planning_config(job['config']);spacing=options.get('candidate_spacing_m',.3)
    if temporal:
        positions=np.array([np.asarray(r['reference_from_camera'])[:3,3] for r in rows])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
        chosen=list(range(len(rows)));spacing=None
    else:
        chosen,arc=frame_samples([np.asarray(r['reference_from_camera'])[:3,3] for r in rows],spacing)
    for i,row in enumerate(rows):row['arc_m']=float(arc[i])
    width,height=manifest['source_width'],manifest['source_height']
    if type(width) is not int or type(height) is not int or width!=2*height:
        raise ValueError('Original source must be stitched 2:1 ERP')
    value=dict(schema='spacewatch.captured_trajectory.v1',capture_id=prefix,frame_id=prefix+'_mesh',
        video=str(video),video_sha256=manifest['source_sha256'],source_fps=manifest['source_fps'],
        source_size=[width,height],pose_path=str(pose_path),pose_sha256=sha(pose_path),
        coordinates=source['coordinates'],coordinates_sha256=sha(source['coordinates']),
        source_manifest=str(manifest_path),source_manifest_sha256=sha(manifest_path),
        frames=rows,sampled_pose_indices=chosen,spacing_m=spacing,length_m=float(arc[-1]),
        sampling='quality-eligible temporal candidates; own reconstructed poses; no arc resampling' if temporal else
                 'nearest actual frame to fixed cumulative-arc targets; endpoints retained; no interpolated camera',
        candidate_interval_s=manifest.get('candidate_interval_s') if temporal else None,
        quality_status='UNVERIFIED' if temporal else None,
        physical_safety_verified=False,metric_scale_verified=source['metric_scale_verified'])
    save(out/'trajectory.json',value)
    result(out,trajectory=str(out/'trajectory.json'),frames=len(rows),candidates=len(chosen),
        length_m=float(arc[-1]),spacing_m=spacing,media_origin='original_video_frame',
        external_artifacts=[str(video),str(pose_path),str(manifest_path),source['coordinates']])

def render(job):
    import cv2
    from .video_frames import FrameReader
    c,out,deps,prefix=context(job);start=time.monotonic()
    trajectory_path=deps[prefix+'.trajectory']['trajectory'];data=read(trajectory_path)
    plan=read(deps[prefix+'.plan']['plan']);requests=read(deps[prefix+'.plan']['requests'])['requests']
    video=Path(c['video']).resolve()
    if str(video)!=data['video'] or sha(video)!=data['video_sha256']:raise ValueError('Video changed since capture sampling')
    bindings={v['viewpoint_id']:v['source_frame'] for v in plan['viewpoints']}
    frames={r['pose_index']:r for r in data['frames']};rows=[]
    stream=cv2.VideoCapture(str(video));reader=FrameReader(stream)
    if not stream.isOpened():raise ValueError('Cannot open original video')
    images=out/'images';images.mkdir()
    try:
        for req in sorted(requests,key=lambda r:bindings[r['viewpoint_id']]['source_frame_index']):
            binding=bindings[req['viewpoint_id']];frame=frames[binding['pose_index']]
            if (binding['source_frame_index']!=frame['source_frame_index'] or
                binding['timestamp_s']!=frame['timestamp_s'] or
                req['reference_from_camera']!=frame['reference_from_camera']):
                raise ValueError('Selected camera/source image binding mismatch; no nearest-frame substitution')
            image=reader.read(frame['source_frame_index'])
            if [image.shape[1],image.shape[0]]!=data['source_size']:raise ValueError('Decoded original video size changed')
            path=images/(req['viewpoint_id']+'.png')
            if not cv2.imwrite(str(path),image):raise IOError('Original frame PNG export failed')
            # Verify decoded RGB pixels survive storage exactly; no image enhancement.
            decoded=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if not np.array_equal(image,decoded):raise ValueError('Original frame pixel preservation failed')
            thumbnail=images/(req['viewpoint_id']+'.thumbnail.jpg')
            preview=cv2.resize(image,(480,240),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(thumbnail),preview,[cv2.IMWRITE_JPEG_QUALITY,88]):raise IOError('Thumbnail export failed')
            rows.append(dict(viewpoint_id=req['viewpoint_id'],reference_from_camera=req['reference_from_camera'],
                image_ref=str(path),render_status='succeeded',media_origin='original_video_frame',
                thumbnail_ref=str(thumbnail),thumbnail_scope='UI listing only; selected panorama uses untouched native pixels',
                source_frame=binding,source_video_sha256=data['video_sha256'],source_size=data['source_size'],
                source_rgb_pixels_sha256=__import__('hashlib').sha256(image[:,:,::-1].copy().tobytes()).hexdigest(),
                alpha_coverage=None,nearest_training_camera_distance_m=None))
    finally:stream.release()
    save(out/'renders.json',dict(results=rows,media_origin='original_video_frame',
        camera_frame_binding_exact=data.get('source_image_pose_exact',True),
        pose_origin=data.get('pose_origin','recorded_camera'),source_frame_identity_exact=True,
        pixel_preservation='lossless PNG of decoder output; no resize or synthesis',
        rendering_seconds=time.monotonic()-start))
    result(out,renders=str(out/'renders.json'),images=len(rows),media_origin='original_video_frame',
           source_resolution=data['source_size'],seconds=time.monotonic()-start,
           external_artifacts=[str(video),trajectory_path])
