"""Uniform whole-video sampling of reconstruction frames, with exact pixel hashes."""
from pathlib import Path
import cv2
import numpy as np
from .common import VIEWS, read, save, sha, perspective_map


def run(config, output):
    output=Path(output); scenes=[]
    for capture in config['captures']:
        source=Path(capture['source_run']); manifest=read(source/'input/manifest.json')
        n=len(manifest['frames'])
        video=capture.get('video')
        cap=None
        if video:
            if sha(video)!=manifest['source_sha256']: raise ValueError('Video differs from reconstruction')
            cap=cv2.VideoCapture(video)
            if not cap.isOpened(): raise IOError(video)
        mode=config.get('sampling','uniform')
        if mode=='all':selected=np.arange(n)
        elif mode=='stride':selected=np.arange(0,n,config.get('frame_stride',1))
        elif mode=='uniform':selected=np.unique(np.rint(np.linspace(0,n-1,min(n,config.get('discovery_frames',12)))).astype(int))
        else:raise ValueError('Unknown sampling mode: '+mode)
        dest=output/'prepared'/capture['id']; rows=[]
        for i in selected:
            row=manifest['frames'][i]; path=source/'input'/row['file']
            if sha(path)!=row['sha256']: raise ValueError('Source image hash mismatch: '+str(path))
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_index'])
                ok,bgr=cap.read()
                if not ok:raise IOError('Could not decode source frame')
            else:bgr=cv2.imread(str(path))
            h,w=bgr.shape[:2]
            wrapped=np.concatenate([bgr[:,-1:],bgr,bgr[:,:1]],1)
            for view in VIEWS:
                mx,my=perspective_map(w,h,view,config.get('view_size',512),config.get('view_fov',100))
                image=cv2.remap(wrapped,mx+1,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                dest.mkdir(parents=True,exist_ok=True)
                target=dest/f'{i:05d}_{view["name"]}.jpg'
                if not cv2.imwrite(str(target),image,[cv2.IMWRITE_JPEG_QUALITY,95]): raise IOError(target)
                rows.append(dict(key=f'{capture["id"]}/{i:05d}_{view["name"]}',scene=capture['id'],frame=int(i),
                                 view=dict(view,fov=config.get('view_fov',100)),path=str(target),sha256=sha(target),source_sha256=row['sha256'],
                                 source_path=str(path),timestamp_s=row['timestamp_s']))
        if cap:cap.release()
        scenes.append(dict(id=capture['id'],source_run=str(source),coordinates=capture['coordinates'],source_index=capture.get('source_index'),
                           frames=[int(i) for i in selected],views=rows,source_video_sha256=manifest['source_sha256'],
                           source_manifest_sha256=sha(source/'input/manifest.json')))
    save(output/'prepared.json',dict(scenes=scenes,method=config.get('sampling','uniform')+' frame sampling; six overlapping perspective views'))


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])
