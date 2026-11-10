"""Automatically recenter detected objects and sample neighboring source frames.

Optional label filtering is an experiment selection, not a shape/size prior.
With no --labels, every detected category supplies view directions.
"""
import argparse
import copy
from pathlib import Path
import sys
import cv2
import numpy as np
from florence_core import read, save, sha


def centered_direction(box, spec, basis, size=512, fov=100):
    x,y=(np.asarray(box)[:2]+np.asarray(box)[2:])/2
    forward,right,down=basis(spec)
    s=np.tan(np.deg2rad(fov/2))
    d=forward+(2*(x+.5)/size-1)*s*right+(2*(y+.5)/size-1)*s*down
    return d/np.linalg.norm(d)


def run(a):
    sys.path.insert(0,a.vendor)
    from scene_discovery.common import basis,perspective_map
    root,out=Path(a.baseline),Path(a.output)
    prepared=read(root/'prepared.json')
    scene=copy.deepcopy(next(s for s in prepared['scenes'] if s['id']==a.scene))
    manifest=read(Path(scene['source_run'])/'input/manifest.json')
    if sha(a.video)!=manifest['source_sha256']:raise ValueError('Video differs from reconstruction')
    labels=set(a.labels.split(',')) if a.labels else None
    directions={}
    allowed=set(getattr(a,'observation_keys',[]) or [])
    for path in sorted((root/'od'/a.scene).glob('*.json')):
        r=read(path);frame=r['view']['frame'];kept=directions.setdefault(frame,[])
        for detection_index,(label,box) in enumerate(zip(r['result']['parsed']['labels'],r['result']['parsed']['bboxes'])):
            if allowed and f"{r['view']['key']}/{detection_index:03d}_region" not in allowed:continue
            if labels is not None and label not in labels:continue
            if len(kept)>=getattr(a,'max_directions',10000):continue
            d=centered_direction(box,r['view']['view'],basis,fov=r['view']['view'].get('fov',100))
            if any(np.rad2deg(np.arccos(np.clip(d@v['direction'],-1,1)))<a.angular_spacing for v in kept):continue
            kept.append(dict(direction=d,label=label,source_key=r['view']['key'],box=box))
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened():raise IOError(a.video)
    rows=[];seeds=[]
    for seed,views in sorted(directions.items()):
        if not views:continue
        targets=sorted(set(np.clip(seed+np.array([-a.frame_offset,0,a.frame_offset]),0,len(manifest['frames'])-1).tolist()))
        for frame in targets:
            source=manifest['frames'][frame]
            cap.set(cv2.CAP_PROP_POS_FRAMES,source['source_frame_index'])
            ok,bgr=cap.read()
            if not ok:raise IOError('Could not decode source frame')
            h,w=bgr.shape[:2];wrapped=np.concatenate([bgr[:,-1:],bgr,bgr[:,:1]],1)
            for j,v in enumerate(views):
                d=v['direction'];spec=dict(name=f'focus{seed:05d}_{j:02d}',
                    yaw=float(np.rad2deg(np.arctan2(d[0],d[2]))),pitch=float(np.rad2deg(np.arcsin(d[1]))),fov=a.fov)
                mx,my=perspective_map(w,h,spec,size=512,fov=a.fov)
                image=cv2.remap(wrapped,mx+1,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                key=f'{a.scene}/{frame:05d}_{spec["name"]}'
                path=out/'prepared'/(key+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(path)
                rows.append(dict(key=key,scene=a.scene,frame=frame,view=spec,path=str(path),sha256=sha(path),
                    source_sha256=source['sha256'],source_path=str(Path(scene['source_run'])/'input'/source['file']),
                    timestamp_s=source['timestamp_s']))
        seeds.append(dict(frame=seed,neighbors=targets,directions=[dict(**{k:v for k,v in x.items() if k!='direction'},direction=x['direction'].tolist()) for x in views]))
    cap.release()
    
    scene['views']=rows;scene['frames']=sorted(set(r['frame'] for r in rows))
    save(out/'prepared.json',dict(scenes=[scene],method='detected-box-centered perspective views; neighboring reconstruction frames'))
    save(out/'refinement_plan.json',dict(source=read(root/'inference_summary.json'),selected_labels=a.labels or 'all',max_directions_per_seed=getattr(a,'max_directions',10000),
        frame_offset=a.frame_offset,angular_spacing=a.angular_spacing,fov=a.fov,seeds=seeds,views=len(rows)))
    print('PREPARED',len(rows),'views',scene['frames'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--vendor',required=True);p.add_argument('--video',required=True)
    p.add_argument('--scene',default='dorm2');p.add_argument('--labels',default='')
    p.add_argument('--frame-offset',type=int,default=2);p.add_argument('--angular-spacing',type=float,default=15)
    p.add_argument('--fov',type=float,default=90)
    run(p.parse_args())
