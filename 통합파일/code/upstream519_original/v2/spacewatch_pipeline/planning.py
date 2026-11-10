"""Explicit mesh-to-Task5b adapter; upstream planner is called without goal changes."""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import Polygon
import trimesh
from .io import read,save,result
from .early_semantics import filtered_source
from .geometry import context,transform,similarity

def polygon_parts(geometry):
    if geometry.geom_type=='Polygon': return [geometry]
    if hasattr(geometry,'geoms'):
        return [p for g in geometry.geoms for p in polygon_parts(g)]
    return []

def face_components(faces):
    n=len(faces)
    if not n:return []
    vertices=faces.ravel();fi=np.repeat(np.arange(n),3)
    order=np.argsort(vertices,kind='stable');v=vertices[order];ids=fi[order]
    same=v[1:]==v[:-1];a=ids[1:][same];b=ids[:-1][same]
    graph=coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(n,n))
    count,labels=connected_components(graph,directed=False)
    return [np.flatnonzero(labels==i) for i in range(count)]

def scene(job):
    c,out,deps,prefix=context(job)
    if c.get('planning_scene'):
        spec=read(c['planning_scene'])
        if spec['capture_id']!=prefix or spec['coordinates']['frame_id']!=prefix+'_mesh':
            raise ValueError('Explicit planning scene must use this capture and mesh frame')
        result(out,scene=c['planning_scene'],external_artifacts=[c['planning_scene']])
        return
    seg=deps[prefix+'.semantic'];src=filtered_source(deps,prefix)
    data=np.load(seg['arrays']);v,f,labels=data['vertices'],data['faces'],data['face_class_id']
    catalog={r['id']:r for r in read(seg['vocabulary'])}
    triangles=v[f];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    length=np.linalg.norm(cross,axis=1);area=length/2
    normal=cross/np.maximum(length[:,None],1e-12)
    floorids=[k for k,val in catalog.items() if val['name'] in ['floor','tactile paving']]
    floor=(np.isin(labels,floorids))&(abs(normal[:,2])>.8)&(area>1e-10)
    if not floor.any():raise ValueError('No usable semantic floor. Provide corrected segmentation/explicit planning scene.')
    height=float(np.median(triangles[floor].mean(1)[:,2]))
    floor &= abs(triangles.mean(1)[:,2]-height)<job['config'].get('scene',{}).get('floor_tolerance_m',.25)
    polys=[Polygon(t[:,:2]) for t in triangles[floor] if Polygon(t[:,:2]).area>1e-10]
    footprint=shapely.union_all(polys)
    parts=polygon_parts(footprint)
    if not parts:raise ValueError('Empty floor footprint')
    minimum=job['config'].get('scene',{}).get('minimum_object_region_area_m2',.01)
    objects=[];structures=[];small=[]
    meshdir=out/'regions';meshdir.mkdir()
    for label in np.unique(labels):
        selected=np.flatnonzero(labels==label)
        name=catalog.get(int(label),{'name':'unknown'})['name']
        structural=name in ['wall','floor','ceiling','tactile paving','unknown']
        groups=[np.arange(len(selected))] if structural else face_components(f[selected])
        for group_index,group in enumerate(groups):
            faceids=selected[group];oid=f'class_{label}_region_{group_index:04d}'
            path=meshdir/(oid+'.glb')
            used,inv=np.unique(f[faceids],return_inverse=True)
            trimesh.Trimesh(v[used],inv.reshape(-1,3),process=False).export(path)
            item=dict(mesh_uri=str(path),class_name=name,solid=False)
            if structural or float(area[faceids].sum())<minimum:
                structures.append(item)
                if not structural:small.append(dict(region_id=oid,class_name=name,area_m2=float(area[faceids].sum())))
            else:
                item.update(instance_id=oid,class_id=int(label),source_triangle_count=len(faceids),
                            description='Connected semantic mesh region; physical object identity unverified')
                importance=job['config'].get('importance_by_class',{}).get(name)
                if importance is not None:item['importance']=importance
                objects.append(item)
    tf,_=similarity(read(src['coordinates'])['source_to_mesh'])
    poses=np.array([r['camera_to_world'] for r in read(Path(src['source_run'])/'result/camera_poses.json')])
    centers=transform(poses[:,:3,3],tf)
    spec=dict(schema_name='spacewatch3d.planning_scene',schema_version='0.1.0',
              scene_id=job['config'].get('scene_id','spacewatch_scene'),capture_id=prefix,synthetic=False,
              coordinates=dict(frame_id=prefix+'_mesh',units='m',up_axis='+Z'),
              # The complete labeled mesh already contains every floor triangle.
              # Do not add a second flat synthetic floor to coverage geometry.
              floor=dict(height_m=height,include_in_coverage=False,components=[dict(polygon=list(p.exterior.coords),
                         holes=[list(r.coords) for r in p.interiors]) for p in parts]),
              camera=dict(positions_m=centers.tolist()),objects=objects,structures=structures,
              input_assumptions=dict(height_approximate=not src['metric_scale_verified'],floor_estimated=True,
                                     physical_safety_verified=False),
              source_context=dict(semantic_region_identity='unverified',small_regions_as_structure=small,
                                  original_mesh_faces=len(f),source_run=src['source_run'],
                                  metric_scale_verified=src['metric_scale_verified']))
    save(out/'scene.json',spec)
    result(out,scene=str(out/'scene.json'),floor_height_m=height,objects=len(objects),
           small_regions=len(small),floor_components=len(parts),all_faces_preserved=True)

def plan(job):
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(300,repeat=True)
    import sys
    from .io import ROOT
    sys.path.insert(0,str(ROOT/'vendor/native'))
    import objective_executor
    objective_executor.install()
    if job['config'].get('native_visibility',True):
        import native_executor
        native_executor.install()
    from spacewatch3d.tasks.task_5b_keyframes.planner import generate
    from spacewatch3d.tasks.task_5b_keyframes.validation import validate_outputs
    c,out,deps,prefix=context(job)
    scene_path=deps.get(prefix+'.priority',{}).get('scene') or c.get('planning_scene') or deps[prefix+'.scene']['scene']
    config=job['config'].get('planning',{})
    print('TASK5B_PLANNING',scene_path,flush=True)
    target=generate(Path(scene_path),out/'plan',config)
    faulthandler.cancel_dump_traceback_later()
    plan=read(target);requests=target.with_name('render_requests.json')
    if prefix+'.priority' in deps:
        from .change_priority import evaluate
        audit=evaluate(plan,deps[prefix+'.priority']['records'])
        plan['change_priority_report']=audit
        if not audit['all_observed']:
            plan['planning_status']='incomplete'
            reference=target.with_name('reference_route.json')
            if reference.exists():reference.unlink()
            plan['failures'].append(dict(code='change_targets_unobserved',targets=[r['change_id'] for r in audit['targets'] if not r['observed']]))
        save(target,plan)
    validate_outputs(plan,read(requests))
    result(out,plan=str(target),requests=str(requests),planning_status=plan['planning_status'],
           viewpoints=len(plan['viewpoints']),quality_status=plan['quality_status'],
           input_coverage_status=plan['input_coverage_status'],external_artifacts=[scene_path] if c.get('planning_scene') else [])
