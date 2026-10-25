"""Run the unchanged pretrained SimSaC on actual registered Gaussian renders."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch
from prepare import FrozenPartition

ROOT=Path('/workspace/gaussian_change_20261025')
PAIR_ROOT=Path('/workspace/pano_change_models_20261025')
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rays_for(width,height):
    y,x=np.mgrid[:height,:width];lo=((x+.5)/width-.5)*2*np.pi;la=((y+.5)/height-.5)*np.pi
    return np.stack([np.sin(lo)*np.cos(la),np.sin(la),np.cos(lo)*np.cos(la)],-1)

def clusters(before,after,zone,height):
    arrays=[before,after];points=np.concatenate(arrays,0)
    if not len(points):return[]
    phases=np.r_[np.zeros(len(before),dtype='u1'),np.ones(len(after),dtype='u1')]
    voxel=np.floor(points/(.06*height)).astype('i4')
    unique,inverse=np.unique(voxel,axis=0,return_inverse=True)
    edges=cKDTree(unique.astype(float)).query_pairs(np.sqrt(3)+1e-8,output_type='ndarray')
    if len(edges):
        adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(unique),len(unique)))
        _,labels=connected_components(adjacency,directed=False)
    else:labels=np.arange(len(unique))
    groups=labels[inverse];result=[]
    for group in np.unique(groups):
        mask=groups==group;p=points[mask]
        bounds=np.stack([p.min(0),p.max(0)]);pad=.02*height
        bounds[0]-=pad;bounds[1]+=pad
        result.append(dict(method='simsac',zone=zone,bounds=bounds.tolist(),center=bounds.mean(0).tolist(),before_pixels=int((phases[mask]==0).sum()),after_pixels=int((phases[mask]==1).sum()),localization='Gaussian rendered expected depth; connected 0.06h voxels; 0.02h box padding',verified_object_change=False))
    result.sort(key=lambda r:r['before_pixels']+r['after_pixels'],reverse=True)
    for number,r in enumerate(result,1):r.update(id=f'SimSaC-{zone}-{number:03d}',title=f'{zone} SimSaC 후보 {number}')
    assert sum(r['before_pixels'] for r in result)==len(before)
    assert sum(r['after_pixels'] for r in result)==len(after)
    return result

def person_mask(model,directory,phase,width,height,rays):
    # Existing policy: segmentation OR detection box with 10% margin each side.
    specs={'front':([0,0,1],[0,-1,0]),'right':([1,0,0],[0,-1,0]),'back':([0,0,-1],[0,-1,0]),'left':([-1,0,0],[0,-1,0]),'up':([0,-1,0],[0,0,-1]),'down':([0,1,0],[0,0,1])}
    union=np.zeros((height,width),bool);detections=[]
    for face,(f,up) in specs.items():
        image=cv2.imread(str(directory/(phase+'-'+face+'.jpg')));h,w=image.shape[:2]
        result=model.predict(image,conf=.25,classes=[0],imgsz=640,device=0,verbose=False)[0]
        mask=np.zeros((h,w),bool)
        if result.masks is not None:
            for raw in result.masks.data.cpu().numpy():mask|=cv2.resize(raw,(w,h),interpolation=cv2.INTER_NEAREST)>.5
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x0,y0,x1,y1=map(float,box);mx,my=(x1-x0)*.1,(y1-y0)*.1
                mask[max(0,int(np.floor(y0-my))):min(h,int(np.ceil(y1+my))),max(0,int(np.floor(x0-mx))):min(w,int(np.ceil(x1+mx)))]=True
                detections.append(dict(face=face,xyxy=box.tolist()))
        down=-np.array(up,float);forward=np.array(f,float);right=np.cross(down,forward);basis=np.column_stack([right,down,forward]);d=rays@basis
        focal=.5*w/np.tan(np.deg2rad(100)/2);z=np.maximum(d[...,2],1e-8)
        mx=(focal*d[...,0]/z+w/2-.5).astype('f4');my=(focal*d[...,1]/z+h/2-.5).astype('f4')
        valid=(d[...,2]>0)&(mx>=0)&(mx<w)&(my>=0)&(my<h)
        projected=cv2.remap(mask.astype('u1'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        union|=valid&projected
    return union,detections

def main():
    for _ in range(600):
        status=ROOT/'render-status.json'
        state=read(status) if status.exists() else {}
        if state.get('state')=='error':raise RuntimeError('Gaussian rendering failed: '+state.get('traceback',''))
        if state.get('state')=='complete':break
        time.sleep(1)
    else:raise TimeoutError('Gaussian rendering did not complete')
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    protocol=read(ROOT/'protocol.json');out=ROOT/'published';data=read(out/'gaussian-data.json')
    (out/'predictions').mkdir(exist_ok=True)
    save(ROOT/'inference-status.json',dict(state='loading'))
    sys.path.insert(0,'/workspace/dorm_registration_benchmark_20261019/image-lifecycle-v4/deps')
    from ultralytics import YOLO
    sys.path.insert(0,str(PAIR_ROOT))
    from run_pair_models import SimSaC
    setup=read(PAIR_ROOT/'setup-status.json');assert setup['stage']=='complete'
    model=SimSaC(setup)
    assert model.audit['checkpoint_sha256']=='55f28690989603b5b01723d255bbc63157bacc704e635100ae517b0ad4d1f99a'
    detector_path=Path('/workspace/dorm_registration_benchmark_20261019/image-lifecycle-v4/yolo11l-seg.pt')
    detector=YOLO(str(detector_path))
    save(out/'simsac-model.json',{**model.audit,'input_source':'New registered Gaussian render RGB; no original-photo probability reused','person_checkpoint_sha256':sha(detector_path),'person_policy':'Six rendered cube views, YOLO11l-seg person masks OR boxes with 10% margin on each side; union across both epochs'})
    classifier=FrozenPartition({k:v for k,v in np.load(ROOT/'partition.npz').items()})
    W,H=protocol['decisions']['render_size'];rays=rays_for(W,H);height=protocol['height_u'];start=time.time()
    result=dict(schema='gaussian-simsac-candidates-v1',zones={},boxes={'raw':[],'filtered':[]},self_controls={},threshold=.5,alpha_threshold=.85,source='gaussian',frame_hashes={})
    for number,zone in enumerate(protocol['zones']):
        zid=zone['id'];directory=out/'renders'/zid;pose=np.array(zone['camera_to_common_cv']);rgb={};geom={};points={};people={};detected={}
        for phase in ['before','after']:
            record=data['renders'][zid][phase];path=out/record['image'];assert sha(path)==record['image_sha256']
            rgb[phase]=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            geometry_path=out/record['geometry'];assert sha(geometry_path)==record['geometry_sha256'];geom[phase]=dict(np.load(geometry_path))
            points[phase]=pose[:3,3]+(rays@pose[:3,:3].T)*geom[phase]['depth'][...,None]
            people[phase],detected[phase]=person_mask(detector,directory,phase,W,H,rays)
            result['frame_hashes'][record['image']]=record['image_sha256']
        person=people['before']|people['after']
        visible=np.ones((H,W),bool)
        for phase in ['before','after']:
            g=geom[phase];visible&=(g['alpha']>=.85)&np.isfinite(g['depth'])&(g['depth']>.02*height)&(g['depth']<40*height)
        stats={};selected={'raw':{},'filtered':{}}
        for phase,other in [('before','after'),('after','before')]:
            probability=model.predict(rgb[phase],rgb[other]);assert np.isfinite(probability).all()
            path=out/'predictions'/(zid+'-'+phase+'.npz');np.savez_compressed(path,prob=probability.astype('f4'))
            raw=probability>.5
            finite=np.isfinite(points[phase]).all(-1);roi=np.zeros((H,W),bool)
            roi[finite]=classifier.assign(points[phase][finite])==int(zid[1:])
            unfiltered=raw&roi&visible;filtered=unfiltered&~person
            assert not (filtered&person).any()
            values=dict(raw_pixels=int(raw.sum()),outside_region_pixels=int((raw&~roi).sum()),unobserved_pixels=int((raw&roi&~visible).sum()),removed_person_pixels=int((unfiltered&person).sum()),filtered_pixels=int(filtered.sum()),observable_without_person_filter=int(unfiltered.sum()),probability_file=str(path.relative_to(out)),probability_sha256=sha(path))
            assert values['raw_pixels']==sum(values[k] for k in ['outside_region_pixels','unobserved_pixels','removed_person_pixels','filtered_pixels'])
            stats[phase]=values;selected['raw'][phase]=points[phase][unfiltered];selected['filtered'][phase]=points[phase][filtered]
            rgba=np.zeros((H,W,4),'u1');rgba[...,0]=raw*255;rgba[...,1]=person*255;rgba[...,2]=(roi&visible)*255;rgba[...,3]=255
            cv2.imwrite(str(directory/(phase+'-candidates.png')),cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA))
        control=model.predict(rgb['before'],rgb['before'])
        result['self_controls'][zid]=dict(candidate_pixels=int((control>.5).sum()),max_probability=float(control.max()))
        zone_boxes={}
        for mode in ['raw','filtered']:
            boxes=clusters(selected[mode]['before'],selected[mode]['after'],zid,height)
            zone_boxes[mode]=len(boxes);result['boxes'][mode].extend(boxes)
        result['zones'][zid]=dict(stats=stats,boxes=zone_boxes,person_pixels=int(person.sum()),person_detections=detected,valid_overlap_pixels=int(visible.sum()),same_camera_pose=True)
        save(ROOT/'inference-status.json',dict(state='running',zone=zid,completed=number+1,total=13,seconds=time.time()-start))
        print('SIMSAC',zid,{s:stats[s]['filtered_pixels'] for s in stats},zone_boxes,flush=True)
    save(out/'simsac-candidates.json',result)
    save(out/'gaussian-validation.json',dict(distinct_source_videos=True,distinct_checkpoints=True,source_gaussians=[s['gaussians'] for s in data['scenes']],exported_gaussians={p:sum(a['gaussians'] for a in data['models'][p]) for p in ['before','after']},native_to_mesh_camera_checks=data['camera_checks'],registration_and_decision_hashes_unchanged=protocol['source_hashes'],new_input_inference=True,identical_registered_camera_poses=True,person_candidate_overlap_pixels=0,all_candidate_pixel_decompositions_exact=True,self_controls=result['self_controls'],model_checkpoint_sha256=model.audit['checkpoint_sha256'],display_is_true_gaussian_splats=True,gaussian_depth_is_expected_depth_not_measured_surface=True,ground_truth_accuracy_claimed=False))
    save(ROOT/'inference-status.json',dict(state='complete',seconds=time.time()-start,zones=13,boxes={k:len(v) for k,v in result['boxes'].items()},self_candidate_pixels=sum(x['candidate_pixels'] for x in result['self_controls'].values())))
    print('SIMSAC COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        save(ROOT/'inference-status.json',dict(state='error',traceback=traceback.format_exc()));raise
