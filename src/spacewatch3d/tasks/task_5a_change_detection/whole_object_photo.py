"""Render complete original-photo instance masks, never a mesh-fragment mask.

Uniform scaling and translation place the whole cutout at its spatial hint.
It is a 2D photo overlay, not a recovered 3D object or a physical-ID match.
"""
import argparse
import copy
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from .dorm_surface import decode, panorama_box, read, write
from .object_photo_tour import camera_matrix

ALIASES = {'cabinet':{'refrigerator'}, 'table':{'dining table'}, 'swivel chair':{'chair'}}


def project(points, camera, width=1036, height=518):
    p = (points-camera[:3,3])@camera[:3,:3]
    radius = np.linalg.norm(p,axis=1)
    return np.c_[(np.arctan2(p[:,0],p[:,2])/(2*np.pi)+.5)*width-.5,
                 (np.arcsin(np.clip(p[:,1]/np.maximum(radius,1e-9),-1,1))/np.pi+.5)*height-.5]


def warp_whole_cutout(rgb, mask, source_camera, target_camera, anchor):
    """Place an intact 2D cutout without mesh clipping or perspective slicing.

    Source RGB and alpha receive the same uniform scale and translation.
    Spatial hints locate the cutout; they do not reshape its visible outline.
    """
    height,width = mask.shape
    yy,xx = np.mgrid[:height,:width]
    source_uv=project(np.asarray(anchor)[None,:],source_camera,width,height)[0]
    target_uv=project(np.asarray(anchor)[None,:],target_camera,width,height)[0]
    source_distance=np.linalg.norm(anchor-source_camera[:3,3])
    target_distance=np.linalg.norm(anchor-target_camera[:3,3])
    scale=source_distance/max(target_distance,1e-6)
    dx=(xx-target_uv[0]+width/2)%width-width/2
    uv=np.stack([source_uv[0]+dx/scale,source_uv[1]+(yy-target_uv[1])/scale],axis=-1).astype('f4')
    valid=(uv[:,:,1]>=0)&(uv[:,:,1]<=height-1)
    alpha=np.asarray(mask,dtype='f4')
    premultiplied=np.dstack([np.asarray(rgb,dtype='f4')*alpha[:,:,None],alpha])
    result=cv2.remap(premultiplied,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    result[~valid]=0
    rgba=np.zeros((height,width,4),dtype='uint8')
    rgba[:,:,:3]=np.clip(np.divide(result[:,:,:3],result[:,:,3:4],out=np.zeros_like(result[:,:,:3]),where=result[:,:,3:4]>1e-5),0,255).astype('uint8')
    rgba[:,:,3]=np.rint(np.clip(result[:,:,3],0,1)*255).astype('uint8')
    return rgba


def overlap_mask(mask, box, padding=9):
    height,width=mask.shape
    x0,y0,x1,y1=box
    xs=(np.arange(width)>=0) & (((np.arange(width)-x0+padding)%width)<x1-x0+2*padding)
    ys=(np.arange(height)>=max(0,y0-padding))&(np.arange(height)<=min(height-1,y1+padding))
    intersection=int(np.count_nonzero(mask&ys[:,None]&xs[None,:]))
    return intersection/max(1,min(int(mask.sum()),int(xs.sum()*ys.sum())))


def publish_whole_cutouts(source, data, output):
    context=read(source/'alignment-context.json')
    poses=[{p['frame']:p for p in read(source/f'new-inputs/poses-{e}.json')['poses']} for e in [0,1]]
    detections=read(source/'image-lifecycle-newmesh-v1/detections.json')['frames']
    tracks={t['id']:t for t in read(source/'image-lifecycle-newmesh-v1/analysis.json')['tracks']}
    mask_cache={};photo_cache={};audit={}
    for item in data['events']:
        geometry=[]
        for mid in item['meshes']:
            geometry.append(decode((output/data['object_meshes'][mid]['file']).read_bytes())[0] if mid else None)
        available=[p for p in geometry if p is not None]
        anchor=np.median(np.concatenate(available),axis=0)
        anchors=[np.median(p,axis=0) if p is not None else anchor for p in geometry]
        pairs=([item['photo_review']['views']] if item.get('photo_review') else [])+item['views']
        candidates=[[],[]]
        cls=item.get('object_class',item['class_name']);allowed=ALIASES.get(cls,{cls})
        for e in [0,1]:
            # An expected empty location is not an observed before object.
            if item['category']=='instance' and not any(pair[e] and pair[e].get('observed') for pair in pairs):continue
            hint=geometry[e] if geometry[e] is not None else np.concatenate(available)
            if len(hint)>256:hint=hint[np.linspace(0,len(hint)-1,256).astype(int)]
            exact_ids={o['id'] for key in item.get('trigger_tracks',[]) if key in tracks and tracks[key]['epoch']==e for o in tracks[key]['observations']}
            for frame in detections:
                if frame['epoch']!=e or frame['frame'] not in poses[e]:continue
                photo=source/f'object-comparison-v1/photos/Dorm{e+2}-{frame["frame"]:05d}.jpg'
                if not photo.exists():continue
                camera=camera_matrix(context,poses[e][frame['frame']],e,item['zone'])
                distance=np.linalg.norm(anchors[e]-camera[:3,3])
                if not .10*data['height_u']<distance<5*data['height_u']:continue
                box=panorama_box(project(hint,camera))
                if box is None:continue
                for ob in frame['objects']:
                    if ob['class_name'] not in allowed:continue
                    if exact_ids and ob['id'] not in exact_ids:continue
                    path=source/ob['mask_url']
                    if not path.exists():path=source.parent/ob['mask_url']
                    if not path.exists():continue
                    if ob['id'] not in mask_cache:mask_cache[ob['id']]=np.asarray(Image.open(path))>0
                    mask=mask_cache[ob['id']];overlap=overlap_mask(mask,box)
                    if overlap<.08 or mask.sum()<30:continue
                    candidates[e].append({'detection':ob,'camera':camera,'photo':photo,'mask':mask,
                                          'frame':frame['frame'],'overlap':overlap,'distance':distance,
                                          'apparent_area':float(mask.sum()*distance**2)})
        item_audit=[]
        pair_quality={}
        for index,pair in enumerate(pairs):
            person_occlusion=[]
            for target_epoch,view in enumerate(pair):
                if not view:continue
                target=camera_matrix(context,poses[target_epoch][view['frame']],target_epoch,item['zone'])
                view['ghosts']=[];combined=np.zeros((518,1036),bool)
                for e in [0,1]:
                    if not candidates[e]:continue
                    target_direction=anchors[e]-target[:3,3];target_direction/=np.linalg.norm(target_direction)
                    physical_area=max(c['apparent_area'] for c in candidates[e])
                    def score(c):
                        direction=anchors[e]-c['camera'][:3,3];direction/=np.linalg.norm(direction)
                        cosine=float(direction@target_direction)
                        same=c['frame']==view['frame'] and e==target_epoch
                        return (3*cosine + .8*c['apparent_area']/physical_area
                                + .4*c['overlap'] + .2*c['detection']['confidence'] + .25*same)
                    chosen=max(candidates[e],key=score)
                    name=f'Dorm{e+2}-{chosen["frame"]:05d}.jpg'
                    if name not in photo_cache:photo_cache[name]=np.asarray(Image.open(chosen['photo']).convert('RGB'))
                    if not (output/name).exists():shutil.copy2(chosen['photo'],output/name)
                    rgba=warp_whole_cutout(photo_cache[name],chosen['mask'],chosen['camera'],target,anchors[e])
                    mask=rgba[:,:,3]>0;combined|=mask
                    edge=mask&~binary_erosion(mask)
                    rgba[edge,:3]=[255,173,115] if e==0 else [120,202,255]
                    filename=f'ghost-{item["id"]}-{e}-{target_epoch}-{index:02d}.png'
                    Image.fromarray(rgba).save(output/filename)
                    detail={'epoch':e,'file':filename,'visible_pixels':int(mask.sum()),
                            'source_frames':[chosen['frame']],'source_detection_id':chosen['detection']['id'],
                            'source_mask_pixels':int(chosen['mask'].sum()),'rendering':'whole_original_photo_mask',
                            'mesh_clipping_used':False,'placement':'intact 2D cutout; uniform scale and translation from spatial hints; approximate alignment'}
                    view['ghosts'].append(detail);item_audit.append(detail)
                if combined.any():
                    yy,xx=np.nonzero(combined)
                    view['change_bbox_px']=view.get('change_bbox_px',view['bbox_px'])
                    view['bbox_px']=panorama_box(np.c_[xx,yy])
                    people=np.zeros_like(combined)
                    target_detections=next((f['objects'] for f in detections if f['epoch']==target_epoch and f['frame']==view['frame']),[])
                    for ob in target_detections:
                        if ob['class_name']!='person':continue
                        path=source/ob['mask_url']
                        if not path.exists():path=source.parent/ob['mask_url']
                        if path.exists():people|=np.asarray(Image.open(path))>0
                    person_occlusion.append(float((combined&people).sum()/combined.sum()))
            pair_quality[id(pair)]=max(person_occlusion,default=1.)
        # Choose an existing pair with less photographer occlusion as the first
        # tour stop; frame files, poses and timestamps stay paired as supplied.
        if not item.get('photo_review'):
            item['views'].sort(key=lambda pair:pair_quality.get(id(pair),1.))
        item['ghost_representation']='complete_visible_original_photo_object'
        audit[item['id']]=item_audit
        print('WHOLE PHOTO',item['id'],[len(c) for c in candidates],flush=True)
    data['ghost_representation']='whole_original_photo_masks'
    data['protocol']['ghosts_clipped_to_mesh']=False
    data['protocol']['ghosts_are_approximate_2d_photo_placements']=True
    return audit


def build(source, previous, output):
    if output.exists():raise ValueError('Choose a new output directory')
    shutil.copytree(previous,output)
    data=copy.deepcopy(read(output/'tour.json'))
    audit=publish_whole_cutouts(source,data,output)
    write(output/'tour.json',data)
    write(output/'whole-object-photo-audit.json',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['source','previous','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.source,a.previous,a.output)
