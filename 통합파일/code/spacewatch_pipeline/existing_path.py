"""Per-bin sharpest source frames on an existing path; no front-end quality gate."""
import csv
from fractions import Fraction
from pathlib import Path
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .io import read,save,result,sha
from .geometry import context,similarity


def quality_samples(pool, metrics, interval, start, end):
    """One sharpest actual frame per occupied temporal bin.

    Camera-anchor availability and distance from the bin centre are NOT ranks.
    Observation utility is deliberately left to the unchanged final selector.
    """
    interval=Fraction(str(interval))
    if interval<=0:raise ValueError('Candidate interval must be positive')
    bins={};unposed=[]
    for row in pool:
        if not start<=row['timestamp_s']<=end:
            unposed.append(row);continue
        metric=metrics[row['source_frame_index']]
        sharpness=float(metric['sharpness'])
        if not math.isfinite(sharpness) or sharpness<0:raise ValueError('Invalid quality score')
        key=math.floor(Fraction(row['pts'])*Fraction(row['time_base'])/interval)
        bins.setdefault(key,[]).append(dict(row,quality_sharpness=sharpness,temporal_bin=key))
    rows=[min(values,key=lambda r:(-r['quality_sharpness'],r['source_frame_index']))
          for _,values in sorted(bins.items())]
    return rows,unposed


def interpolate(anchors,chosen):
    times=np.array([a['timestamp_s'] for a in anchors])
    if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Chronological recorded anchors required')
    target=np.array([r['timestamp_s'] for r in chosen])
    if not len(target) or target.min()<times[0] or target.max()>times[-1]:
        raise ValueError('Never extrapolate outside the recorded camera path')
    matrices=np.array([a['reference_from_camera'] for a in anchors])
    rotations=Slerp(times,Rotation.from_matrix(matrices[:,:3,:3]))(target).as_matrix()
    positions=np.column_stack([np.interp(target,times,matrices[:,k,3]) for k in range(3)])
    by_frame={a['source_frame_index']:i for i,a in enumerate(anchors)}
    rows=[]
    for index,(row,rotation,position) in enumerate(zip(chosen,rotations,positions)):
        if row['source_frame_index'] in by_frame:
            a=b=by_frame[row['source_frame_index']];matrix=matrices[a].copy();alpha=0.
            origin='recorded_anchor'
        else:
            b=int(np.searchsorted(times,row['timestamp_s'],side='right'));a=b-1
            alpha=float((row['timestamp_s']-times[a])/(times[b]-times[a]))
            matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=position
            origin='interpolated_existing_path'
        rows.append(dict(row,pose_index=index,reference_from_camera=matrix.tolist(),
            pose_origin=origin,anchor_pose_indices=[a,b],
            anchor_frame_indices=[anchors[a]['source_frame_index'],anchors[b]['source_frame_index']],
            interpolation_alpha=alpha,pose_independently_estimated=False))
    return rows


def path_between(data,a,b):
    frames=data['frames'];first,last=frames[a],frames[b]
    knots=list(frames[a:b+1])
    for row in data.get('recorded_frames',[]):
        if first['timestamp_s']<row['timestamp_s']<last['timestamp_s']:knots.append(row)
    knots.sort(key=lambda r:r['timestamp_s'])
    return [np.asarray(r['reference_from_camera'])[:3,3].tolist() for r in knots]


def trajectory(job):
    started=time.monotonic();c,out,deps,prefix=context(job)
    source=deps[prefix+'.reconstruct'];root=Path(source['source_run'])
    pose_path=root/'result/camera_poses.json';manifest_path=root/'input/manifest.json'
    manifest=read(manifest_path);poses=read(pose_path)
    video=Path(c['video']).resolve();opt=job['config']['media']
    q=Path(opt['quality_source'])
    # Quality pixels were already measured. This stage only reads metadata and
    # must not require an image decoder/vision model in the geometry runtime.
    summary=read(q/'summary.json');quality_state=read(q/'state.json')
    if quality_state['status']!='completed' or quality_state['decoded_frames']!=summary['source_frames']:
        raise ValueError('Incomplete full-video sharpness measurements')
    if sha(video)!=summary['source']['sha256']:
        raise ValueError('Sharpness measurements belong to a different input video')
    if summary['source']['sha256']!=manifest['source_sha256']:
        raise ValueError('Quality frames and existing point cloud depict different video')
    if len(poses)!=len(manifest['frames']):raise ValueError('Existing frame/pose count mismatch')
    with (q/'all_frame_metrics.csv').open() as stream:
        metrics={int(r['source_frame_index']):r for r in csv.DictReader(stream)}
    stamps={i:dict(source_frame_index=i,pts=int(r['pts']),time_base=r['time_base'],timestamp_s=float(r['timestamp_s']))
            for i,r in metrics.items()}
    if sorted(stamps)!=list(range(summary['source_frames'])):
        raise ValueError('Quality must be measured on the whole source before sampling')
    tf,scale=similarity(read(source['coordinates'])['source_to_mesh'])
    anchors=[]
    for i,(p,f) in enumerate(zip(poses,manifest['frames'])):
        if p['frame']!=i or abs(p['timestamp_s']-f['timestamp_s'])>1e-6:
            raise ValueError('Existing camera does not belong to its recorded frame')
        raw=np.asarray(p['camera_to_world'],float);matrix=np.eye(4)
        matrix[:3,:3]=tf[:3,:3]@raw[:3,:3]/scale
        matrix[:3,3]=tf[:3,:3]@raw[:3,3]+tf[:3,3]
        anchors.append(dict(stamps[f['source_frame_index']],reference_from_camera=matrix.tolist()))
    # All source frames participate. Never read the old eligibility pool or
    # exposure/relative-sharpness pass flags; only sharpness ranks within a bin.
    pool=list(stamps.values());interval=Fraction(str(opt.get('candidate_interval_s',.25)))
    if interval <= 0:raise ValueError('Candidate interval must be positive')
    chosen,unposed=quality_samples(pool,metrics,interval,anchors[0]['timestamp_s'],anchors[-1]['timestamp_s'])
    if not chosen:raise ValueError('No source frames on the recorded path')
    rows=interpolate(anchors,chosen)
    positions=np.array([np.asarray(a['reference_from_camera'])[:3,3] for a in anchors])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    for row in rows:
        a,b=row['anchor_pose_indices'];alpha=row['interpolation_alpha']
        row['arc_m']=float(arc[a]*(1-alpha)+arc[b]*alpha)
    gaps=np.diff([r['timestamp_s'] for r in rows])
    value=dict(schema='spacewatch.captured_trajectory.v1',capture_id=prefix,frame_id=prefix+'_mesh',
        video=str(video),video_sha256=manifest['source_sha256'],source_fps=manifest['source_fps'],
        source_size=[manifest['source_width'],manifest['source_height']],
        pose_path=str(pose_path),pose_sha256=sha(pose_path),coordinates=source['coordinates'],
        coordinates_sha256=sha(source['coordinates']),source_manifest=str(manifest_path),
        source_manifest_sha256=sha(manifest_path),frames=rows,recorded_frames=anchors,
        sampled_pose_indices=list(range(len(rows))),spacing_m=None,length_m=float(arc[-1]),
        sampling='All source frames; maximum sharpness per temporal bin; recorded XYZ path',
        front_quality_filter=False,sharpness_measurement_retained=True,
        sampling_interval_kind='temporal_bins_not_exact_frame_timestamps',
        quality_rank='sharpness_desc_then_source_index',anchor_or_bin_center_priority=False,
        candidate_interval_s=float(interval),quality_status='UNVERIFIED',
        pose_origin='existing_path_interpolation',pose_accuracy_verified=False,
        new_reconstruction_or_relocalization=False,source_image_pose_exact=False,
        physical_safety_verified=False,metric_scale_verified=source['metric_scale_verified'],
        recorded_pose_count=len(anchors),recorded_candidates=sum(r['pose_origin']=='recorded_anchor' for r in rows),
        interpolated_candidates=sum(r['pose_origin']=='interpolated_existing_path' for r in rows),
        max_gap_s=float(gaps.max()) if len(gaps) else 0.,
        source_frames_without_bracketing_pose=len(unposed),
        unsupported_end_frames=unposed,seconds=time.monotonic()-started)
    save(out/'trajectory.json',value)
    result(out,trajectory=str(out/'trajectory.json'),frames=len(rows),candidates=len(rows),
        length_m=float(arc[-1]),spacing_m=None,media_origin='original_video_frame',
        pose_origin=value['pose_origin'],seconds=value['seconds'],external_artifacts=[str(video),
            str(pose_path),str(manifest_path),source['coordinates'],*[str(q/n) for n in
            ['summary.json','state.json','all_frame_metrics.csv']]])
