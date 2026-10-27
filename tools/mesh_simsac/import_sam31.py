"""Adapt saved SAM video masks to ERP evidence without running segmentation again."""
import argparse
import json
from pathlib import Path
import numpy as np


def export(source, output, captures=('dorm2', 'dorm3')):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    labels = sorted({o['label'] for scene in captures for p in (source/'results'/scene).glob('*/objects.json')
                     for o in json.loads(p.read_text()).values()})
    catalog = {label: i+1 for i, label in enumerate(labels)}
    index = {'labels': labels, 'captures': {}}
    h, w = 512, 1024
    yy, xx = np.mgrid[:h, :w]
    lon=((xx+.5)/w-.5)*2*np.pi;lat=(.5-(yy+.5)/h)*np.pi
    rays=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
    for scene in captures:
        folder=output/scene;folder.mkdir()
        manifest=json.loads((source/'inputs'/scene/'manifest.json').read_text())
        report=json.loads((source/'delivery'/scene/'report.json').read_text())
        mesh_ids={(t['view'],t['local_id']):o['instance_id'] for o in report['instance_records'] for t in o['tracks']}
        mappings={};objects={}
        for view in manifest['views']:
            name=view['name'];yaw,pitch=np.deg2rad([view['yaw_deg'],view['pitch_deg']])
            forward=np.array([np.cos(pitch)*np.sin(yaw),np.sin(pitch),np.cos(pitch)*np.cos(yaw)])
            right=np.array([np.cos(yaw),0,-np.sin(yaw)]);down=np.cross(right,forward)
            z=rays@forward;s=np.tan(np.deg2rad(view['fov_deg']/2))
            x=(rays@right)/np.maximum(z,1e-10)/s*256+255.5
            y=(rays@down)/np.maximum(z,1e-10)/s*256+255.5
            valid=(z>0)&(x>=-.5)&(x<511.5)&(y>=-.5)&(y<511.5)
            mappings[name]=(valid,np.clip(np.rint(y),0,511).astype(int),np.clip(np.rint(x),0,511).astype(int),z*z)
            objects[name]=json.loads((source/'results'/scene/name/'objects.json').read_text())
        frames=[]
        for f in manifest['frames'][::3]:
            classes=np.zeros((h,w),'u2');instances=np.zeros((h,w),'u4');person=np.zeros((h,w),bool);best=np.zeros((h,w),'f4')
            for view,(valid,y,x,quality) in mappings.items():
                with np.load(source/'results'/scene/view/f'{f["local_frame"]:05d}.npz') as data:
                    masks=np.unpackbits(data['masks_packed'],axis=1,count=512*512).reshape(tuple(data['shape'])).astype(bool)
                    for oid,mask,score in zip(data['ids'],masks,data['scores']):
                        label=objects[view][str(int(oid))]['label'];selected=valid&mask[y,x]
                        if label=='person':person|=selected;continue
                        win=selected&(quality*float(score)>best)
                        classes[win]=catalog[label];instances[win]=mesh_ids.get((view,int(oid)),0);best[win]=quality[win]*float(score)
            classes[person]=catalog['person'];instances[person]=0
            name=f'{f["map_frame"]}.npz'
            np.savez_compressed(folder/name,class_id=classes,instance_id=instances,person=person)
            frames.append(dict(frame_id=str(f['map_frame']),file=scene+'/'+name,rgb_sha256=f['sha256']))
        index['captures'][scene]=frames
    (output/'index.json').write_text(json.dumps(index,indent=2))
    return index


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.source,a.output)
