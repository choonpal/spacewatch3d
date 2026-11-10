"""Explicit candidate-only object targets; every other triangle stays as geometry."""
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from .io import read,save,result,sha
from .geometry import context,transform,similarity
from .planning import polygon_parts

def export_obj(path,v,f):
    used,remap=np.unique(f,return_inverse=True)
    with path.open('w') as stream:
        np.savetxt(stream,v[used],fmt='v %.17g %.17g %.17g')
        np.savetxt(stream,remap.reshape(-1,3)+1,fmt='f %d %d %d')

def scene(job):
    c,out,deps,prefix=context(job);seg=deps[prefix+'.instances'];src=deps[prefix+'.reconstruct']
    d=np.load(seg['arrays']);v,f,labels=d['vertices'],d['faces'],d['face_class_id'];tri=v[f]
    catalog={r['id']:r for r in read(seg['vocabulary'])};cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1);area=length/2
    normal=cross/np.maximum(length[:,None],1e-12)
    tf,_=similarity(read(src['coordinates'])['source_to_mesh'])
    poses=np.array([r['camera_to_world'] for r in read(Path(src['source_run'])/'result/camera_poses.json')]);centers=transform(poses[:,:3,3],tf)
    floorids=[k for k,r in catalog.items() if r['name'] in ['floor','tactile paving']]
    floor=np.isin(labels,floorids)&(abs(normal[:,2])>.8)&(area>1e-10);method='semantic_floor'
    if not floor.any() and job['config'].get('scene',{}).get('geometry_floor_fallback',False):
        z=tri.mean(1)[:,2];eligible=(abs(normal[:,2])>.9)&(area>1e-10)&(z<np.quantile(centers[:,2],.1)-.2)
        if not eligible.any():raise ValueError('No supported horizontal floor below capture positions')
        unique,inv=np.unique(np.floor(z[eligible]/.1).astype(int),return_inverse=True)
        winner=unique[np.argmax(np.bincount(inv,weights=area[eligible]))]
        floor=eligible&(abs(z-(winner+.5)*.1)<.15);method='geometry_horizontal_area_mode_below_cameras'
    if not floor.any():raise ValueError('No usable planning floor; supply supported floor labels or enable geometry_floor_fallback')
    height=float(np.average(tri[floor].mean(1)[:,2],weights=area[floor]))
    floor &= abs(tri.mean(1)[:,2]-height)<job['config'].get('scene',{}).get('floor_tolerance_m',.25)
    parts=polygon_parts(shapely.union_all([Polygon(t[:,:2]) for t in tri[floor] if Polygon(t[:,:2]).area>1e-10]))
    if not parts:raise ValueError('Empty planning floor footprint')
    folder=out/'regions';folder.mkdir();objects=[];used=np.zeros(len(f),np.uint8)
    for r in read(seg['regions']):
        if r['kind']!='object_candidate':continue
        mask=d['face_region_id']==r['id'];used[mask]+=1;oid='region_'+str(r['id']);path=folder/(oid+'.obj')
        export_obj(path,v,f[mask]);name=catalog[r['class_id']]['name']
        item=dict(instance_id=oid,mesh_uri='regions/'+path.name,class_name=name,class_id=r['class_id'],
            source_triangle_count=int(mask.sum()),solid=False,description='Semantic + assumed class size candidate; physical identity unverified')
        if name in job['config'].get('importance_by_class',{}):item['importance']=job['config']['importance_by_class'][name]
        objects.append(item)
    rest=used==0;structures=[]
    if rest.any():
        export_obj(folder/'remainder.obj',v,f[rest]);structures=[dict(mesh_uri='regions/remainder.obj',solid=False)]
    used[rest]+=1
    if not np.all(used==1):raise ValueError('Scene export omitted or duplicated faces')
    spec=dict(schema_name='spacewatch3d.planning_scene',schema_version='0.1.0',scene_id=job['config'].get('scene_id','spacewatch_scene'),
        capture_id=prefix,synthetic=False,coordinates=dict(frame_id=prefix+'_mesh',units='m',up_axis='+Z'),
        floor=dict(estimation_method=method,height_m=height,include_in_coverage=False,components=[dict(polygon=list(p.exterior.coords),holes=[list(r.coords) for r in p.interiors]) for p in parts]),
        camera=dict(positions_m=centers.tolist()),objects=objects,structures=structures,
        input_assumptions=dict(height_approximate=not src['metric_scale_verified'],floor_estimated=True,physical_safety_verified=False),
        source_context=dict(physical_object_identity_verified=False,metric_scale_verified=src['metric_scale_verified'],original_mesh_faces=len(f),
            segmentation='semantic_size nominal candidates only',segmentation_sha256=sha(seg['arrays']),
            required_objects=len(objects),residual_faces=int(rest.sum()),residual_observation_policy='spatial coverage, not per-object must-see'))
    save(out/'scene.json',spec);result(out,scene=str(out/'scene.json'),floor_height_m=height,objects=len(objects),
        residual_faces=int(rest.sum()),floor_components=len(parts),all_faces_preserved=True)
