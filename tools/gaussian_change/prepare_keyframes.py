"""Use saved Task 5b tour poses, without consulting change candidates.

The original 13 registration regions remain frozen. Rendering views are separate
from those regions: either epoch's requested camera becomes one common camera
for both Gaussian maps. No nearest opposite-epoch keyframe is substituted.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from prepare import FrozenPartition, epoch_to_common, read, save, sha, transform, membership


def transform_camera(camera, reference_to_common):
    """Transform a CV camera by a similarity, retaining a unit rotation."""
    camera=np.asarray(camera,dtype=float)
    m=np.asarray(reference_to_common,dtype=float)
    scale=np.cbrt(np.linalg.det(m[:3,:3]))
    rotation=m[:3,:3]/scale
    if scale<=0 or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8):
        raise ValueError('Camera transform is not a proper similarity')
    if camera.shape!=(4,4) or not np.allclose(camera[3],[0,0,0,1]):
        raise ValueError('Invalid source camera')
    if not np.allclose(camera[:3,:3].T@camera[:3,:3],np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(camera[:3,:3]),1):
        raise ValueError('Source camera rotation is not right handed')
    result=np.eye(4)
    result[:3,:3]=rotation@camera[:3,:3]
    result[:3,3]=transform(camera[:3,3],m)
    return result


def build(base_protocol,source_root,output,*,source_sets=None,view_ids=None,remote_root='/workspace/gaussian_keyframes_20261026_v2'):
    base=read(base_protocol)
    protocol=copy.deepcopy(base)
    classifier=FrozenPartition(dict(np.load(base_protocol.parent/'partition.npz')))
    source_sets=source_sets or [('P2','dorm2',source_root/'production/dorm2'),
                 ('P3','dorm3',source_root/'production/dorm3'),
                 ('S3','dorm3',source_root/'task5b')]
    views=[];sources=[];checks={}
    for prefix,capture,folder in source_sets:
        epoch=['dorm2','dorm3'].index(capture)
        plan_folder=folder/'plan' if (folder/'plan/render_requests.json').exists() else folder
        request_path=plan_folder/'render_requests.json'
        plan_path=plan_folder/'tour_plan.json'
        requests=read(request_path);plan=read(plan_path);scene=read(folder/'scene.json')
        if scene['capture_id']!=capture or scene['coordinates']['up_axis']!='+Z':
            raise ValueError('Wrong capture or coordinate convention')
        data_root=base_protocol.parent/'source/data'/capture
        original=np.load(data_root/'original_poses_glb.npy')
        mesh_to_glb=np.load(data_root/'mesh_to_glb.npy')
        native=transform(original[:,:3,3],np.linalg.inv(mesh_to_glb))
        positions=np.asarray(scene['camera']['positions_m'])
        if positions.shape!=native.shape:
            raise ValueError('Planner and Gaussian capture camera counts differ')
        residual=np.linalg.norm(positions-native,axis=1)
        if residual.max()>1e-7:
            raise ValueError('Planner cameras do not match the Gaussian native frame')
        checks[prefix]={'capture':capture,'cameras':len(native),'max_native_position_residual':float(residual.max()),
                        'planner_to_gaussian':mesh_to_glb.tolist(),
                        'max_gaussian_position_residual':float(np.linalg.norm(transform(positions,mesh_to_glb)-original[:,:3,3],axis=1).max())}
        viewpoint_poses={v['viewpoint_id']:v['reference_from_camera'] for v in plan['viewpoints']}
        if requests['route_id']!=plan['route_id']:
            raise ValueError('Render requests and tour plan have different routes')
        for number,request in enumerate(requests['requests'],1):
            if request['capture_id']!=capture or request['projection']!='equirectangular':
                raise ValueError('Unsupported render request')
            pose=np.asarray(request['reference_from_camera'],dtype=float)
            np.testing.assert_allclose(pose,viewpoint_poses[request['viewpoint_id']],atol=1e-12)
            # Requests live in the Z-up mesh frame. Gaussian training cameras
            # live in the converted Y-up GLB frame: both bridges are necessary.
            native_to_mesh=np.asarray(base['scenes'][epoch]['native_to_mesh'])@mesh_to_glb
            old_position=transform(pose[:3,3],native_to_mesh)
            region=int(membership(base,old_position[None],epoch,classifier)[0])
            zid=f'R{region:02d}'
            native_to_common=epoch_to_common(base['alignment_context'],epoch,zid)@native_to_mesh
            common=transform_camera(pose,native_to_common)
            recovered=np.linalg.solve(native_to_common,np.r_[common[:3,3],1])[:3]
            np.testing.assert_allclose(recovered,pose[:3,3],atol=1e-10)
            view_id=view_ids[request['viewpoint_id']] if view_ids else f'{prefix}-{number:03d}'
            views.append({'id':view_id,'label':f'{"통계 실험" if prefix=="S3" else "기준 계획"} {capture} · {number:02d} · {zid}',
                          'anchor_epoch':epoch,'capture_id':capture,'viewpoint_id':request['viewpoint_id'],
                          'plan_group':prefix,'registration_zone':zid,'registration_supported':bool(base['alignment_context']['methods']['structure']['errors']['regions'][zid]['supported']),
                          'camera_to_common_cv':common.tolist(),'source_reference_from_camera':pose.tolist(),
                          'source_reference_to_gaussian':mesh_to_glb.tolist(),
                          'source_width':request['width'],'source_height':request['height'],
                          'projection':'equirectangular','media_origin':request.get('media_origin'),
                          'target':(common[:3,3]+common[:3,2]*base['height_u']).tolist(),
                          'reset_from_pose':True,'camera_frame':None,'camera_timestamp_s':None,
                          'source_request_index':number-1,'existing_change_candidates_used':False})
        sources.append({'group':prefix,'capture':capture,'render_requests':str(request_path),'render_requests_sha256':sha(request_path),
                        'tour_plan':str(plan_path),'tour_plan_sha256':sha(plan_path),'scene_sha256':sha(folder/'scene.json'),
                        'route_id':plan['route_id'],'count':len(requests['requests']),
                        'planning_status':plan.get('planning_status'),'publication_ready':plan.get('publication_ready'),
                        'quality_status':plan.get('quality_status'),'failure_codes':[f['code'] for f in plan.get('failures',[])]})
    protocol.update(schema='registered-gaussian-tour-keyframes-v1',remote_root=remote_root,
                    views=views,keyframe_sources=sources,keyframe_camera_checks=checks,baseline_boxes=[])
    protocol['decisions'].update(view_selection='All saved Task 5b render requests from the three recorded plans; no change-candidate selection',
                                 existing_change_candidates_used_for_view_selection=False,
                                 same_pose_renders_at_each_requested_keyframe=True,
                                 source_keyframes_publication_ready=False,
                                 query_resolution=[1024,512],source_pose_preserved=True)
    output.mkdir(parents=True,exist_ok=True)
    save(output/'protocol.json',protocol)
    (output/'partition.npz').write_bytes((base_protocol.parent/'partition.npz').read_bytes())
    save(output/'selection-audit.json',{'sources':sources,'camera_checks':checks,'views':len(views),
                                      'candidates_used':False,'registered_before_rendering':True,
                                      'registration_unsupported_zones':[z for z,e in base['alignment_context']['methods']['structure']['errors']['regions'].items() if not e['supported']]})
    print(json.dumps({'views':len(views),'sources':sources,'camera_checks':checks},ensure_ascii=False,indent=2))
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-protocol',type=Path,default=Path('outputs/gaussian-change-20261025/protocol.json'))
    p.add_argument('--source-root',type=Path,default=Path('outputs/gaussian-keyframes-20261026/source'))
    p.add_argument('--output',type=Path,default=Path('outputs/gaussian-keyframes-20261026-v2'))
    a=p.parse_args();build(a.base_protocol,a.source_root,a.output)
