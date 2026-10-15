"""Reconstruct two actual temporal windows of Bonn RGB-D, without labels.

Official Bonn / RTABMap format-12 pose correction is S @ GT @ S @ M.
M contains a near-uniform scale. Its polar rotation is applied to the camera,
its mean singular value to depth, avoiding a nonrigid TSDF extrinsic matrix.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

INTRINSICS = [542.822841, 542.576870, 315.593520, 237.756098]
DISTORTION = [.039903, -.099343, -.000730, -.000144, 0.]
S = np.array([[-1.,0,0],[0,0,1.],[0,1.,0]])
M = np.array([[1.0157,.1828,-.2389,.0113],[.0009,-.8431,-.6413,-.0098],
              [-.3009,.6147,-.8085,.0111],[0,0,0,1.]])
u,singular,vt = np.linalg.svd(M[:3,:3])
M_ROTATION = u @ vt
DEPTH_CALIBRATION_SCALE = float(singular.mean())


def read_index(path):
    return [(float(parts[0]),parts[1]) for line in Path(path).read_text().splitlines()
            if (parts:=line.split()) and not line.startswith('#')]


def interpolate_pose(gt, timestamp, max_gap=.10):
    right=int(np.searchsorted(gt[:,0],timestamp))
    if right==0 or right==len(gt):raise ValueError('Image timestamp outside ground-truth coverage')
    left=right-1;times=gt[[left,right],0]
    if times[1]-times[0]>max_gap:raise ValueError('Ground-truth interpolation gap too large')
    ratio=(timestamp-times[0])/(times[1]-times[0])
    pose=np.eye(4);pose[:3,:3]=Slerp(times-times[0],Rotation.from_quat(gt[[left,right],4:8]))([timestamp-times[0]]).as_matrix()[0]
    pose[:3,3]=gt[left,1:4]*(1-ratio)+gt[right,1:4]*ratio
    corrected=np.eye(4)
    corrected[:3,:3]=S@pose[:3,:3]@S@M_ROTATION
    corrected[:3,3]=S@pose[:3,3]+S@pose[:3,:3]@S@M[:3,3]
    return corrected


def camera_points(depth, step=4):
    y,x=np.mgrid[0:depth.shape[0]:step,0:depth.shape[1]:step]
    z=depth[::step,::step]/1000.
    fx,fy,cx,cy=INTRINSICS
    return np.stack([(x-cx)*z/fx,(y-cy)*z/fy,z],axis=-1), (z>.3)&(z<4.5)


def prepare(data, output, windows=((1.,4.5),(11.5,16.3)), stride=5):
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output directory')
    prepared=output/'prepared';frames_dir=output/'frames';cache=prepared/'cache'
    cache.mkdir(parents=True);frames_dir.mkdir()
    rgb=read_index(data/'rgb.txt');depth=read_index(data/'depth.txt');gt=np.loadtxt(data/'groundtruth.txt')
    origin_time=rgb[0][0];depth_times=np.array([t for t,p in depth])
    fx,fy,cx,cy=INTRINSICS;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    map_x,map_y=cv2.initUndistortRectifyMap(K,np.array(DISTORTION),None,K,(640,480),cv2.CV_32FC1)
    rows=[];images={};depths={}
    for epoch,(start,end) in enumerate(windows):
        for index,(timestamp,rgb_path) in enumerate(rgb):
            if not start<=timestamp-origin_time<=end or index%stride:continue
            nearest=int(np.argmin(np.abs(depth_times-timestamp)))
            if abs(depth_times[nearest]-timestamp)>.025:continue
            try:pose=interpolate_pose(gt,timestamp)
            except ValueError:continue
            color=cv2.imread(str(data/rgb_path));raw=cv2.imread(str(data/depth[nearest][1]),cv2.IMREAD_UNCHANGED)
            if color is None or raw is None or raw.dtype!=np.uint16:raise ValueError('Invalid original RGB-D frame')
            calibrated=np.rint(raw.astype(float)/5000*DEPTH_CALIBRATION_SCALE*1000).astype('<u2')
            measured=cv2.remap(calibrated,map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            name=f'frame-{epoch}-{index:06d}';raw_path=(frames_dir/(name+'.raw')).resolve();measured.tofile(raw_path)
            cv2.imwrite(str(frames_dir/(name+'.jpg')),color,[cv2.IMWRITE_JPEG_QUALITY,96])
            image=cv2.cvtColor(cv2.remap(color,map_x,map_y,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            row={'id':name,'epoch':epoch,'source_frame_index':index,'source_timestamp':timestamp,
                'time_s':timestamp-origin_time,'source_capture':epoch,'width':640,'height':480,
                'url':'/m3c2-independent/'+name+'.jpg','status':'dataset_pose','new_frame':True,
                'camera_to_world':pose.tolist(),'depth_camera_to_world':pose.tolist(),
                'depth_path':str(raw_path),'depth_intrinsics':INTRINSICS,'depth_size':[640,480],
                'depth_units_per_meter':1000,'rgb_distortion':DISTORTION,
                'pose_source':'Bonn groundtruth + official coordinate/calibration correction',
                'rgb_sharpness':float(cv2.Laplacian(cv2.cvtColor(color,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()),
                'original_rgb_file':rgb_path,'original_rgb_sha256':hashlib.sha256((data/rgb_path).read_bytes()).hexdigest()}
            rows.append(row);images[name]=image;depths[name]=measured
    if any(sum(r['epoch']==e for r in rows)<6 for e in (0,1)):raise ValueError('Too few supported RGB-D frames')
    # One common rigid display frame. Lower-image plane only sets orientation;
    # it never masks floor points or participates in change classification.
    reference=next(r for r in reversed(rows) if r['epoch']==1)
    p,valid=camera_points(depths[reference['id']]);valid[:int(len(valid)*.65)]=False
    points=p[valid];pose=np.array(reference['camera_to_world']);points=points@pose[:3,:3].T+pose[:3,3]
    plane_cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.025)
    plane,inliers=plane_cloud.segment_plane(.02,3,1000)
    up=np.array(plane[:3]);d=plane[3]
    if up@pose[:3,3]+d<0:up=-up;d=-d
    right=pose[:3,0]-up*(up@pose[:3,0]);right/=np.linalg.norm(right)
    rotation=np.vstack([right,np.cross(up,right),up]);floor_origin=pose[:3,3]-up*(up@pose[:3,3]+d)
    common=np.eye(4);common[:3,:3]=rotation;common[:3,3]=-rotation@floor_origin
    for row in rows:
        camera=common@np.array(row['camera_to_world'])
        row['camera_to_world']=camera.tolist();row['depth_camera_to_world']=camera.tolist()
    intrinsic=o3d.camera.PinholeCameraIntrinsic(640,480,fx,fy,cx,cy)
    for epoch in (0,1):
        selected=[r for r in rows if r['epoch']==epoch]
        volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.015,sdf_trunc=.06,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for row in selected:
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(images[row['id']]),
                o3d.geometry.Image(depths[row['id']]),depth_scale=1000,depth_trunc=4.5,convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intrinsic,np.linalg.inv(np.array(row['camera_to_world'])))
        mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices();mesh.compute_vertex_normals()
        if not len(mesh.triangles):raise ValueError('TSDF reconstruction is empty')
        o3d.io.write_triangle_mesh(str(cache/f'{epoch}-depth8.ply'),mesh)
        print('Epoch',epoch,'frames',len(selected),'triangles',len(mesh.triangles),flush=True)
    clean_frames=[{k:v for k,v in r.items() if k not in ('depth_path','depth_camera_to_world')} for r in rows]
    metadata={'scene':'Bonn box removal','source_epochs':[0,1],'dataset_id':'bonn-removal','dataset':'Bonn RGB-D Dynamic Dataset',
        'sequence':'rgbd_bonn_removing_nonobstructing_box','windows_s':windows,
        'capture_relation':'Two temporal windows of one real sequence; not separate repeat surveys.',
        'source_page':'https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/',
        'calibration':{'K':K.tolist(),'distortion':DISTORTION,'original_depth_units_per_meter':5000,
            'depth_scale_correction':DEPTH_CALIBRATION_SCALE,'marker_calibration_matrix':M.tolist(),
            'camera_correction':'S @ GT @ S @ M, uniform scale separated into depth',
            'common_display_transform':common.tolist(),'display_floor_plane_inliers':len(inliers)}}
    (prepared/'review.json').write_text(json.dumps({**metadata,'keyframes':[]},indent=2),encoding='utf-8')
    (prepared/'changes.json').write_text(json.dumps({**metadata,'kind':'rgbd_reconstruction_pair','regions':[],
        'keyframes':clean_frames},indent=2),encoding='utf-8')
    (frames_dir/'cameras.json').write_text(json.dumps({**metadata,'frames':rows},indent=2),encoding='utf-8')
    (output/'source.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Prepared',len(rows),'original RGB-D frames; depth calibration scale',DEPTH_CALIBRATION_SCALE,flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.output)
