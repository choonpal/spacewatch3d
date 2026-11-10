"""Measure every ERP frame, without eligibility, exposure or relative-blur gates.

The four tangent views and Laplacian variance are the existing metric, unchanged.
No images are exported. Candidate supply is handled by existing_path separately.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def tangent_maps(width=1024, height=512, face=192):
    axis=(np.arange(face,dtype=np.float32)+.5)*2/face-1
    x,y=np.meshgrid(axis,-axis)
    maps=[]
    for yaw in [0,math.pi/2,math.pi,3*math.pi/2]:
        dx=math.cos(yaw)*x+math.sin(yaw)
        dz=-math.sin(yaw)*x+math.cos(yaw)
        lon=np.arctan2(dx,dz)
        lat=np.arctan2(y,np.sqrt(dx*dx+dz*dz))
        u=((lon/(2*math.pi)+.5)*width-.5)%width
        v=(.5-lat/math.pi)*height-.5
        maps.append((u.astype(np.float32),v.astype(np.float32)))
    return maps


def measure(gray,maps):
    values=[]
    for u,v in maps:
        face=cv2.remap(gray,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        values.append(float(cv2.Laplacian(face,cv2.CV_32F,ksize=3)[2:-2,2:-2].var()))
    return float(np.median(values))


def scan(video,out,deadline=600):
    import av
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();count=0
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    state=dict(status='running',decoded_frames=0,front_quality_filter=False)
    save('state.json',state)
    cv2.setNumThreads(4);maps=tangent_maps()
    try:
        with av.open(str(video)) as container,(out/'all_frame_metrics.csv').open('x',newline='') as stream:
            source=container.streams.video[0]
            source.codec_context.thread_count=4;source.thread_type='AUTO'
            if source.width!=2*source.height:raise ValueError('Expected 2:1 ERP video')
            writer=csv.DictWriter(stream,fieldnames=['source_frame_index','pts','time_base','timestamp_s','sharpness'])
            writer.writeheader();last=None;tick=started
            for index,frame in enumerate(container.decode(source)):
                if frame.pts is None:raise ValueError('Missing source PTS')
                if time.monotonic()-started>deadline:raise TimeoutError('Finite scan deadline exceeded')
                stamp=float(frame.pts*frame.time_base)
                if last is not None and stamp<=last:raise ValueError('Nonchronological source PTS')
                last=stamp
                gray=frame.reformat(width=1024,height=512,format='gray').to_ndarray()
                writer.writerow(dict(source_frame_index=index,pts=int(frame.pts),time_base=str(frame.time_base),
                                     timestamp_s=stamp,sharpness=measure(gray,maps)))
                count=index+1
                if time.monotonic()-tick>10:
                    state.update(decoded_frames=count);save('state.json',state);tick=time.monotonic()
            if not count or (source.frames and count!=source.frames):raise ValueError('Incomplete source decode')
            meta=dict(width=source.width,height=source.height,average_rate=str(source.average_rate),declared_frames=source.frames)
        h=hashlib.sha256()
        with Path(video).open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        summary=dict(source=dict(path=str(Path(video).resolve()),sha256=h.hexdigest(),**meta),source_frames=count,
                     seconds=time.monotonic()-started,front_quality_filter=False,
                     policy='Measure every frame; no eligible pool, exposure gate or relative-sharpness threshold')
        save('summary.json',summary)
        state.update(status='completed',decoded_frames=count,elapsed_seconds=summary['seconds']);save('state.json',state)
        return summary
    except BaseException as error:
        state.update(status='failed',decoded_frames=count,error=str(error));save('state.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--deadline',type=float,default=600);a=p.parse_args()
    print(json.dumps(scan(Path(a.video),Path(a.output),a.deadline)))
