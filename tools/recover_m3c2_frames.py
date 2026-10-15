"""New RGB-D keyframes for independent geometry events; never read instance IDs.

Temporal optical flow anchors new original RGB frames to an existing estimated
RGB camera via mesh ray intersections and PnP. Depth cameras are separately fit
to the mesh. Both remain estimates, not provided calibration/ground truth.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recover_rescan_keyframes import depth_records, depth_cloud, render


def track_camera(anchor_image, image, xyz, mesh_depth, anchor):
    a = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.erode((np.isfinite(mesh_depth)*255).astype('u1'), np.ones((5,5),'u1'))
    corners = cv2.goodFeaturesToTrack(a, 1400, .008, 8, mask=mask)
    if corners is None:
        return None
    moved, okay, error = cv2.calcOpticalFlowPyrLK(a,b,corners,None,winSize=(31,31),maxLevel=4)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(b,a,moved,None,winSize=(31,31),maxLevel=4)
    p, q = corners[:,0], moved[:,0]
    valid = (okay.ravel()>0)&(reverse.ravel()>0)&(error.ravel()<24)&(np.linalg.norm(back[:,0]-p,axis=1)<1.)
    valid &= (q[:,0]>8)&(q[:,0]<640)&(q[:,1]>8)&(q[:,1]<476)
    pp = np.rint(p[valid]).astype(int); q = q[valid]
    obj = xyz[pp[:,1],pp[:,0]].astype('f8')
    finite = np.isfinite(obj).all(1); obj,q = obj[finite],q[finite].astype('f8')
    if len(obj)<24:
        return None
    focal = anchor['focal_px']
    K = np.array([[focal,0,323.5],[0,focal,241.5],[0,0,1.]])
    okay, rv, tv, inliers = cv2.solvePnPRansac(obj,q,K,None,iterationsCount=1500,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not okay or inliers is None or len(inliers)<24:
        return None
    ids=inliers.ravel(); rv,tv=cv2.solvePnPRefineLM(obj[ids],q[ids],K,None,rv,tv)
    error = np.linalg.norm(cv2.projectPoints(obj[ids],rv,tv,K,None)[0][:,0]-q[ids],axis=1)
    w2c=np.eye(4);w2c[:3,:3]=cv2.Rodrigues(rv)[0];w2c[:3,3]=tv.ravel();pose=np.linalg.inv(w2c)
    offset=np.linalg.norm(pose[:3,3]-np.asarray(anchor['camera_to_world'])[:3,3])
    spread=np.ptp(q[ids],axis=0)
    if np.median(error)>2.5 or min(spread)<80 or offset>.8 or not .15<pose[1,3]<2.4:
        return None
    return {'camera_to_world':pose.tolist(),'focal_px':focal,'status':'estimated_rgb_tracked',
            'rgb_inlier_count':len(ids),'rgb_match_count':len(obj),'reprojection_median_px':float(np.median(error)),
            'inlier_spread_px':spread.tolist(),'anchor_frame_id':anchor['id'],'anchor_offset_m':float(offset),
            'pose_source':'original RGB optical flow + mesh-ray PnP; estimated'}


def depth_camera(depth, mesh, seed):
    source=depth_cloud(depth)
    target=mesh.sample_points_uniformly(65000)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.10,max_nn=35))
    reg=o3d.pipelines.registration
    pose=np.array(seed)
    for threshold in (.22,.10,.045):
        fit=reg.registration_icp(source,target,threshold,pose,reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=threshold*.7)),reg.ICPConvergenceCriteria(max_iteration=45))
        pose=fit.transformation
    offset=np.linalg.norm(pose[:3,3]-np.asarray(seed)[:3,3])
    if fit.fitness<.62 or fit.inlier_rmse>.035 or offset>.35:
        return None
    return {'depth_camera_to_world':pose.tolist(),'depth_fitness':fit.fitness,'depth_rmse_m':fit.inlier_rmse,
            'rgb_depth_camera_offset_m':float(offset)}


def recover(data, review_dir, output):
    cv2.setRNGSeed(20261015);cv2.setNumThreads(4);o3d.utility.random.seed(20261015)
    output.mkdir(parents=True,exist_ok=True);cache=output/'frames-cache';cache.mkdir(exist_ok=True)
    review=json.loads((review_dir/'review.json').read_text(encoding='utf8'))
    existing=json.loads((review_dir/'changes.json').read_text(encoding='utf8'))
    used={link['frame_id'] for event in existing['regions'] for links in event['keyframes'] for link in links}
    decoder=Path(__file__).resolve().parents[1]/'third_party/rescan-depth/decode.exe'
    meshes=[o3d.io.read_triangle_mesh(str(review_dir/f'cache/{e}-depth8.ply')) for e in (0,1)]
    frames=[]
    for epoch,source_epoch in enumerate(review['source_epochs']):
        anchors=[f for f in review['keyframes'] if f['epoch']==epoch and f['status']=='estimated_rgb_checked']
        raw,records,times=depth_records(data/f"{review['scene']}_{source_epoch}.depth")
        targets={}
        for anchor in anchors:
            for delta in (-30,-15,15,30):
                index=anchor['source_frame_index']+delta;name=f'frame-{epoch}-{index:06d}'
                if 0<=index<len(records) and name not in used:
                    if index not in targets or abs(delta)<abs(index-targets[index]['source_frame_index']):targets[index]=anchor
        ray_cache={}
        for anchor in anchors:
            _,xyz,mesh_depth=render(meshes[epoch],np.array(anchor['camera_to_world']),anchor['focal_px'])
            ray_cache[anchor['id']]=(cv2.imread(str(review_dir/(anchor['id']+'.jpg'))),xyz,mesh_depth)
        video=cv2.VideoCapture(str(data/f"{review['scene']}_{source_epoch}.h264"));index=0
        while video.grab():
            if index in targets:
                success,image=video.retrieve()
                if not success:raise ValueError('Original RGB decode failed')
                image=cv2.resize(image,(648,484));anchor=targets[index];name=f'frame-{epoch}-{index:06d}'
                meta={'id':name,'epoch':epoch,'source_frame_index':index,'source_capture':source_epoch,
                      'source_timestamp':float(times[1,index]),'depth_timestamp':float(times[0,index]),
                      'time_s':float(times[1,index]-times[1,0]),'width':648,'height':484,
                      'url':'/m3c2-independent/'+name+'.jpg','status':'unlocalized','new_frame':True}
                frame_path=output/(name+'.jpg');cv2.imwrite(str(frame_path),image,[cv2.IMWRITE_JPEG_QUALITY,93])
                tracked=track_camera(*[ray_cache[anchor['id']][0],image,ray_cache[anchor['id']][1],ray_cache[anchor['id']][2],anchor])
                if tracked:
                    meta.update(tracked)
                    pos,size=records[index];compressed=cache/(name+'.depth-record');decoded=cache/(name+'.raw')
                    compressed.write_bytes(raw[pos:pos+size])
                    subprocess.run([str(decoder),str(compressed),str(decoded),'307200'],check=True,capture_output=True)
                    depth=np.fromfile(decoded,dtype='<u2').reshape(480,640)
                    depth_fit=depth_camera(depth,meshes[epoch],meta['camera_to_world'])
                    if depth_fit:meta.update(depth_fit)
                    meta['depth_path']=str(decoded.resolve())
                frames.append(meta)
                print(name,meta['status'],'RGB inliers',meta.get('rgb_inlier_count',0),'depth',round(meta.get('depth_fitness',0),3),flush=True)
                (output/'cameras.json').write_text(json.dumps({'scene':review['scene'],'source_epochs':review['source_epochs'],
                    'frames':frames,'excluded_frame_ids':sorted(used),'labels_consumed':False},ensure_ascii=False,indent=2),encoding='utf8')
            index+=1
            if targets and index>max(targets):break
        video.release()
    print('New checked RGB frames:',sum(f['status']=='estimated_rgb_tracked' for f in frames),'of',len(frames),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--review-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a new output directory')
    recover(args.data,args.review_dir,args.output)
