"""Publish receipted results for the repository's AUTO TOUR 360 app."""
import os
import re
import shutil
import colorsys
from pathlib import Path
import numpy as np
from plyfile import PlyData
from .io import read,save,result,digest,verify
from .geometry import similarity,transform

def mesh_bytes(vertices,faces,rgb,labels,catalog,path,instance_ids=None):
    triangles=np.asarray(vertices[faces],dtype='<f4')
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    palette=np.full((256,3),145,np.uint8)
    for row in catalog:palette[row['id']]=row['rgb']
    with Path(path).open('wb') as stream:
        np.array([0x53575031,len(faces)*3,1,0],dtype='<u4').tofile(stream)
        triangles.tofile(stream)
        np.repeat(normals,3,axis=0).astype('<f4').tofile(stream)
        np.asarray(rgb[faces],np.uint8).tofile(stream)
        if instance_ids is None:
            np.repeat(palette[labels],3,axis=0).tofile(stream)
        else:
            palette=np.array([[145,145,145]]+[[int(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.72,.95)]
                for i in range(1,int(instance_ids.max(initial=0))+1)],np.uint8)
            np.repeat(palette[instance_ids],3,axis=0).tofile(stream)

def point_bytes(source,coordinates,path,limit=150000):
    cloud=PlyData.read(source,mmap='r')['vertex'].data
    indices=np.linspace(0,len(cloud)-1,min(limit,len(cloud)),dtype=np.int64)
    tf,_=similarity(read(coordinates)['source_to_mesh'])
    xyz=transform(np.column_stack([cloud[k][indices] for k in 'xyz']),tf).astype('<f4')
    rgb=np.column_stack([cloud[k][indices] for k in ['red','green','blue']]).astype(np.uint8)
    with Path(path).open('wb') as stream:
        np.array([0x53575031,len(indices),2,0],dtype='<u4').tofile(stream)
        xyz.tofile(stream);rgb.tofile(stream)
    return dict(display_points=len(indices),source_points=len(cloud),sampling='uniform source-row display sample')

def viewer(job):
    out=Path(job['output']);run=Path(job['run_root']);deps=job['dependencies']
    bundle=out/'public';bundle.mkdir(exist_ok=True)
    native=job['config'].get('media',{}).get('mode')=='original_erp'
    receipts={}
    for complete in (run/'stages').glob('*/complete.json'):
        for record in read(complete)['artifacts']:receipts[record['path']]=record
    checked=set()
    def trusted(path):
        path=Path(path).resolve();key=str(path)
        if key not in receipts:raise ValueError('Viewer input has no completed-stage receipt: '+key)
        if key not in checked:verify(receipts[key]);checked.add(key)
        return path
    assets=[]
    def link(source,name):
        source=trusted(source);target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():target.unlink()
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
        assets.append(name);return '/pipeline-assets/'+name
    captures=[]
    for capture in job['config']['captures']:
        prefix=capture['id'];folder=bundle/prefix;folder.mkdir(exist_ok=True)
        geometry=deps[prefix+'.semantic'];source=deps[prefix+'.reconstruct'];planning=deps[prefix+'.plan']
        instance_result=deps.get(prefix+'.instances')
        arrays=instance_result['arrays'] if instance_result else geometry['arrays']
        with np.load(trusted(arrays)) as data:
            v,f,rgb,labels=[data[k] for k in ['vertices','faces','rgb','face_class_id']]
            ids=data['face_instance_id'] if 'face_instance_id' in data.files else None
        catalog=read(trusted(geometry['vocabulary']))
        mesh_bytes(v,f,rgb,labels,catalog,folder/'mesh.bin');assets.append(prefix+'/mesh.bin')
        objects=[];instance_url=None
        if ids is not None:
            mesh_bytes(v,f,rgb,labels,catalog,folder/'instances.bin',ids);assets.append(prefix+'/instances.bin')
            instance_url='/pipeline-assets/'+prefix+'/instances.bin'
            for iid in np.unique(ids):
                if not iid:continue
                faces=np.flatnonzero(ids==iid);points=v[np.unique(f[faces])];cid=int(labels[faces[0]])
                objects.append(dict(id=int(iid),class_id=cid,name=catalog[cid]['name'],faces=len(faces),
                                    center=((points.min(0)+points.max(0))/2).tolist(),bounds=[points.min(0).tolist(),points.max(0).tolist()]))
        points=point_bytes(trusted(source['point_cloud']),trusted(source['coordinates']),folder/'points.bin')
        assets.append(prefix+'/points.bin')
        plan=read(trusted(planning['plan']));renders=read(trusted(deps[prefix+'.render']['renders']))['results']
        by_id={row['viewpoint_id']:row for row in renders};views=[]
        for view in plan['viewpoints']:
            row=by_id.get(view['viewpoint_id'])
            if not row or row['render_status']!='succeeded' or row['media_origin']!=('original_video_frame' if native else '3DGS_novel_view'):
                raise ValueError('Missing genuine 3DGS image for '+view['viewpoint_id'])
            if row['reference_from_camera']!=view['reference_from_camera']:raise ValueError('Viewer image pose mismatch')
            identifier=view['viewpoint_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+',identifier):raise ValueError('Invalid viewpoint ID')
            image=link(row['image_ref'],prefix+'/images/'+identifier+('.png' if native else '.jpg'))
            views.append(dict(id=identifier,image=image,position=np.asarray(row['reference_from_camera'])[:3,3].tolist(),
                              pose=row['reference_from_camera'],component=view.get('component_id'),roles=view['roles'],
                              alpha_coverage=row['alpha_coverage'],nearest_camera_m=row['nearest_training_camera_distance_m']))
            if native and row.get('thumbnail_ref'):
                views[-1]['thumbnail']=link(row['thumbnail_ref'],prefix+'/images/'+identifier+'.thumbnail.jpg')
            if native:views[-1]['source_frame']=row.get('source_frame')
        downloads={name:link(geometry[key],prefix+'/'+name+Path(geometry[key]).suffix)
                   for name,key in [('semantic_mesh','semantic_mesh'),('rgb_mesh','rgb_mesh'),('labeled_mesh','semantic_ply')]}
        downloads['point_cloud']=link(source['point_cloud'],prefix+'/point_cloud.ply')
        downloads['plan']=link(planning['plan'],prefix+'/tour_plan.json')
        if instance_result:
            downloads['instance_mesh']=link(instance_result['instance_mesh'],prefix+'/mesh_instances.glb')
            downloads['instance_ply']=link(instance_result['instance_ply'],prefix+'/mesh_instances.ply')
            if instance_result.get('regions'):
                downloads['all_regions']=link(instance_result['regions'],prefix+'/regions.json')
                policy_name='target_policy' if instance_result.get('planning_profile') else 'size_assumptions'
                downloads[policy_name]=link(instance_result['assumptions'],prefix+'/'+policy_name+'.json')
                downloads['all_arrays']=link(instance_result['arrays'],prefix+'/mesh_arrays.npz')
        training=(dict(steps=None,selected_step=None,validation_psnr=None) if native else deps[prefix+'.gs_train'])
        scene=read(trusted(deps[prefix+'.scene']['scene']))
        captures.append(dict(id=prefix,frame_id=plan['reference_frame_id'],views=views,
            tour_order=(plan['captured_navigation']['tour_order'] if native else [row['viewpoint_id'] for row in plan['tour_visits']]),
            links=plan['captured_navigation']['links'] if native else plan['tour_links'],
            mesh='/pipeline-assets/'+prefix+'/mesh.bin',points='/pipeline-assets/'+prefix+'/points.bin',
            bounds=[v.min(0).tolist(),v.max(0).tolist()],floor_height=scene['floor']['height_m'],
            faces=len(f),vertices=len(v),classes=catalog,point_preview=points,downloads=downloads,
            planning_status=plan['planning_status'],input_coverage_status=plan.get('input_coverage_status'),
            object_observed_count=plan['coverage_report']['object_observation_requirements']['observed_object_count'],
            object_input_count=plan['coverage_report']['object_observation_requirements']['input_object_count'],
            require_all_objects=plan['provenance']['config']['require_all_objects'],
            whole_input_unmet=sorted({t for r in plan['coverage_report'].get('input_coverage',{}).get('runs',[]) for t in r['unmet_targets']}),
            failures=[row['code'] for row in plan['failures']],
            coverage=plan['coverage_report']['balanced_space_coverage'],
            whole_surface_coverage_min=min(r['coverage_report']['coverage_total'] for r in plan['coverage_report']['independent_validation']['runs']),
            required_candidate_count=sum(plan['provenance']['config']['require_all_objects'] or o.get('must_see',False) for o in plan['object_importances']),
            observation_target_count=len(plan['object_importances']),candidate_only_observation=True,steps=training['steps'],
            selected_step=training['selected_step'],validation_psnr=training['validation_psnr'],
            metric_scale_verified=source['metric_scale_verified'],quality_validated=False,
            instance_mesh=instance_url,objects=objects,instance_summary=instance_result['summary'] if instance_result else None))
        if native:
            trajectory=read(trusted(deps[prefix+'.trajectory']['trajectory']))
            captures[-1].update(media_origin='original_video_frame',source_size=trajectory['source_size'],
                capture_candidates=len(trajectory['frames']),pose_origin=trajectory.get('pose_origin'),
                source_image_pose_exact=trajectory.get('source_image_pose_exact',False),
                captured_path=[np.asarray(f['reference_from_camera'])[:3,3].tolist() for f in trajectory.get('recorded_frames',trajectory['frames'])])
    changes=None
    if 'pair.changes' in deps:
        change_path=trusted(deps['pair.changes']['changes']);tour=read(change_path)
        supported=tour.get('kind')=='dorm_change_tour'
        download=link(change_path,'change_results.json')
        if supported:
            (bundle/'changes').mkdir(exist_ok=True)
            # Keep the exact upstream map/photo contract, with its original frames.
            for path in change_path.parent.iterdir():
                if path.is_file() and path.suffix in ['.json','.bin','.jpg','.png']:
                    link(path,'changes/'+path.name)
            count=len(tour['events']);deferred=tour['summary']['record_dispositions']['deferred']
        else:
            count=deps['pair.changes']['candidates'];deferred=deps['pair.changes']['deferred']
        surface=tour.get('scope')=='surface_regions_not_object_events'
        comparison=None
        rgb_backend=deps['pair.changes'].get('backend')=='mesh_rgb_simsac'
        if rgb_backend:
            root=trusted(deps['pair.changes']['viewer_manifest']).parent
            selected=[root/n for n in ['index.html','viewer-data.json','validation.json','results/events.json','events-viewer.js','events-viewer.css','photo-presentation.js']]
            selected += [p for folder in ['events','preview'] for p in (root/folder).rglob('*') if p.is_file()]
            for p in selected:
                link(p,'mesh-rgb/'+p.relative_to(root).as_posix())
            comparison=dict(url='/pipeline-assets/mesh-rgb/index.html',quality_status='UNVERIFIED')
        if surface:
            from .change_viewer import publish_surface_comparison
            comparison=publish_surface_comparison(tour,job,bundle,trusted,assets,mesh_bytes)
        changes=dict(viewer_supported=supported or surface or rgb_backend,viewer_kind='mesh_rgb' if rgb_backend else 'surface' if surface else 'dorm' if supported else None,
                     comparison=comparison,url='/pipeline-assets/mesh-rgb/index.html' if rgb_backend else '/pipeline-changes.html' if surface else '/index.html?mode=changes&source=dorm' if supported else None,
                     candidates=count,deferred=deferred,download=download)
    identifier=job.get('bundle_id') or digest(dict(run=str(run),captures=captures))
    discovery_info=None
    if 'discovery' in deps:
        source=deps['discovery']
        discovery_info=dict(classes=source['classes'],size_prior_classes=source['size_prior_classes'],
            vocabulary=link(source['vocabulary'],'discovery/classes.json'),
            size_priors=link(source['size_priors'],'discovery/size_priors.json'),
            source_mode=job['config']['segmentation']['source'],
            model='Provided semantic labels' if job['config']['segmentation']['source']=='provided' else 'Florence-2')
        evidence=[]
        colors={c['name']:c['rgb'] for c in read(source['vocabulary'])}
        from PIL import Image
        for item in source['evidence_sources']:
            root=Path(item['data_root'])
            observations=read(trusted(root/'observations.json'))
            observed={r['key']:r for r in observations}
            summary=read(trusted(root/'inference_summary.json'))
            predictions={}
            for key in summary['prediction_keys']:
                record=read(trusted(root/'predictions'/(key+'.json')))
                predictions.setdefault(record['view']['key'],[]).append(record)
            for group in read(trusted(root/'prepared.json'))['scenes']:
                for row in group['views']:
                    rgba=np.zeros((512,512,4),np.uint8);detections=[]
                    for record in predictions.get(row['key'],[]):
                        mask=np.load(trusted(root/'predictions'/(record['key']+'.npz')))['mask']
                        quality=record['quality'];accepted=quality['usable'] and quality.get('outside_box_fraction',1)<=.1 and record['result']['output_tokens']<summary['max_tokens']+1
                        if accepted:rgba[mask]=[*colors.get(record['label'],[160,160,160]),130]
                        observation=observed.get(record['key'],{})
                        detections.append(dict(name=record['label'],box=record['box'],pixels=int(mask.sum()),
                            mask_valid=accepted,candidate_id=observation.get('candidate_id'),
                            reference_rejection=observation.get('reference_rejection'),quality=quality))
                    mask_name='discovery/masks/'+row['key']+'.png';mask_path=bundle/mask_name;mask_path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rgba).save(mask_path);assets.append(mask_name)
                    evidence.append(dict(scene=row['scene'],frame=row['frame'],timestamp_s=row['timestamp_s'],view=row['view']['name'],
                        image=link(row['path'],'discovery/images/'+row['key']+'.jpg'),detections=detections,
                        mask='/pipeline-assets/'+mask_name,caption='Florence-2 OD → region polygons → reconstructed depth. Candidate IDs are not ground-truth identities.'))
        save(bundle/'discovery/evidence.json',evidence);assets.append('discovery/evidence.json')
        discovery_info['evidence']='/pipeline-assets/discovery/evidence.json'
    manifest=dict(schema_version=1,pipeline_version=job['config'].get('pipeline_version','1.1'),bundle_id=identifier,run_id=run.name,captures=captures,changes=changes,
                  discovery=discovery_info,
                  assets=assets,source_app='spacewatch3d/apps/autotour360',image_origin='original video frames' if native else '3DGS novel views',
                  requested_inputs=job['config'].get('requested_inputs'))
    if native:
        from .survey_support import metadata
        manifest['survey']=metadata(job)
    save(bundle/'manifest.json',manifest)
    service=None
    if job.get('viewer_launch',True):
        from .viewer_service import launch
        service=launch(run,bundle,job['config'].get('viewer',{}).get('port',0))
    result(out,bundle=str(bundle),manifest=str(bundle/'manifest.json'),service=service,
           captures=len(captures),images=sum(len(c['views']) for c in captures),changes=changes)
