"""Automatically obtain nearby views around grounded objects for size measurement."""
import copy
from pathlib import Path
import numpy as np
from .common import read,save,sha,perspective_map,basis


def object_view(row,detection,size,fov):
    """Rotate a perspective view toward the detected object, preserving ERP rays."""
    box=np.array(detection['box']);center=(box[:2]+box[2:])/2
    scale=np.tan(np.deg2rad(fov/2));forward,right,down=basis(row['view'])
    xy=(2*(center+.5)/size-1)*scale
    direction=forward+xy[0]*right+xy[1]*down;direction/=np.linalg.norm(direction)
    span=max(box[2:]-box[:2])/size
    crop_fov=float(np.clip(np.rad2deg(2*np.arctan(span*scale*1.6)),35,110))
    return dict(name=f'target{row["frame"]}_{row["view"]["name"]}',yaw=float(np.rad2deg(np.arctan2(direction[0],direction[2]))),
                pitch=float(np.rad2deg(np.arcsin(direction[1]))),fov=crop_fov)


def run(config,output):
    import cv2
    output=Path(output);prepared=read(output/'prepared.json');catalog=read(output/'classes.json')
    eligible={c['name'] for c in catalog if c['instance_eligible']};captures={c['id']:c for c in config['captures']}
    refined=copy.deepcopy(prepared);new_scenes=[]
    for scene in refined['scenes']:
        capture=captures[scene['id']];source=Path(capture['source_run']);manifest=read(source/'input/manifest.json')
        rows=scene['views'].copy();choices={}
        for row in rows:
            detections=read(output/'grounded'/f'{row["key"]}.json')['detections']
            good=[d for d in detections if d['name'] in eligible and not d['border_touch'] and d['stability']>=.85]
            if good:
                best=max(good,key=lambda d:d['dino_score']*d['sam_score'])
                score=best['dino_score']*best['sam_score']
                choices.setdefault(row['frame'],[]).append((score,row,best))
        selected=[]
        for frame,views in sorted(choices.items()):
            for score,row,detection in sorted(views,key=lambda x:(-x[0],x[1]['key']))[:config.get('refinement_views_per_anchor',2)]:
                view=object_view(row,detection,config.get('view_size',512),config.get('view_fov',100))
                for offset in config.get('refinement_offsets',[-2,0,2]):
                    fi=frame+offset
                    if 0<=fi<len(manifest['frames']):selected.append((fi,view))
        existing={r['key'] for r in rows};new=[];cap=cv2.VideoCapture(capture['video']) if capture.get('video') else None
        for frame,view in sorted(selected,key=lambda x:(x[0],x[1]['name'])):
            key=f'{scene["id"]}/{frame:05d}_{view["name"]}'
            if key in existing:continue
            existing.add(key);record=manifest['frames'][frame];source_image=source/'input'/record['file']
            if sha(source_image)!=record['sha256']:raise ValueError('Source image changed')
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES,record['source_frame_index']);ok,image=cap.read()
                if not ok:raise IOError('Could not decode refinement frame')
            else:image=cv2.imread(str(source_image))
            h,w=image.shape[:2];wrapped=np.concatenate([image[:,-1:],image,image[:,:1]],1)
            mx,my=perspective_map(w,h,view,config.get('view_size',512),view['fov'])
            image=cv2.remap(wrapped,mx+1,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            path=output/'prepared_refinement'/f'{key}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(path)
            new.append(dict(key=key,scene=scene['id'],frame=frame,view=view,path=str(path),sha256=sha(path),
                            source_sha256=record['sha256'],source_path=str(source_image),timestamp_s=record['timestamp_s'],
                            selection='automatic temporal neighbor of grounded object'))
        if cap:cap.release()
        scene['views']+=new;scene['frames']=sorted(set(r['frame'] for r in scene['views']))
        new_scenes.append(dict(scene,views=new))
    from .ground import run as ground
    summary=ground(config,output,prepared_override=dict(scenes=new_scenes),names_override=[c['name'] for c in catalog])
    save(output/'refined.json',refined);save(output/'refinement_summary.json',summary)


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])
