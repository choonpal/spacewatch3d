"""Full-mesh size segmentation with explicit candidate/residual separation."""
import colorsys,copy
from pathlib import Path
import numpy as np
from .io import ROOT,read,save,result,sha
from .geometry import context
from .size_geometry import prepare,solve

def assumptions(config):
    opt=config['size_segmentation'];path=Path(opt['assumptions']);path=path if path.is_absolute() else ROOT/path
    spec=read(path);priors=spec['priors'];settings=spec['settings']
    if not priors:raise ValueError('Supply at least one explicit size assumption')
    names=set();ids=set()
    for p in priors:
        cid=p['class_id'];dim=np.asarray(p['dimensions'],float)
        if type(cid)!=int or cid<3 or cid>=255 or cid in ids or p['name'] in names:raise ValueError('Size IDs must be unique 3..254; names unique')
        if dim.shape!=(3,) or not np.isfinite(dim).all() or np.any(dim<=0) or dim[0]>dim[1]:raise ValueError('Sizes must be finite positive [short_horizontal,long_horizontal,height]')
        ids.add(cid);names.add(p['name'])
    for k in ['size_tolerance','noise_allowance','plane_distance','plane_normal_degrees','minimum_plane_area','max_planes','structure_neighbor_angle_degrees','max_gap','candidate_min_faces','candidate_min_two_extent_ratio','random_seed']:
        x=settings[k]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<0:raise ValueError('Invalid size setting: '+k)
    if settings['max_gap']<=0 or settings['candidate_min_faces']<1 or settings['max_planes']<1:raise ValueError('Invalid size support settings')
    for k in ['candidate_min_faces','max_planes','random_seed']:
        if type(settings[k])!=int:raise ValueError('Integer size setting required: '+k)
    if not 0<=settings['candidate_min_two_extent_ratio']<=1:raise ValueError('Extent ratio must be in [0,1]')
    if any(settings[k]>90 for k in ['plane_normal_degrees','structure_neighbor_angle_degrees']):raise ValueError('Normal angles must be in [0,90]')
    for x in [opt.get('factor',1.)]+opt.get('sensitivity_factors',[]):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0:raise ValueError('Size factors must be positive')
    return path,spec

def canonical_labels(labels,catalog,priors,aliases):
    lookup={p['name']:p['class_id'] for p in priors};lookup.update(wall=0,floor=1,ceiling=2)
    # Unmodeled classes retain distinct IDs outside the template catalog.
    unused=iter(i for i in range(3,10000) if i not in set(lookup.values())|{255})
    ids={};inverse={};mapping=[]
    for row in catalog:
        name=aliases.get(row['name'],row['name']);cid=lookup[name] if name in lookup else next(unused)
        ids[row['id']]=cid;inverse[cid]=row['id'];mapping.append(dict(source_id=row['id'],source_name=row['name'],size_name=name,canonical_id=cid,has_size_prior=name in {p['name'] for p in priors}))
    # Two aliases may legitimately share a size, but must not merge distinct source classes.
    canonical=np.full(len(labels),255,np.int32)
    for source,cid in ids.items():canonical[labels==source]=cid
    if len(set(ids.values()))!=len(ids):raise ValueError('Multiple source classes map to one size class; provide distinct size entries instead of merging labels')
    return canonical,inverse,mapping

def export(path,data,ids,kinds,regions):
    import trimesh
    from plyfile import PlyData,PlyElement
    v,f,rgb=[data[k] for k in ['vertices','faces','rgb']]
    vertex=np.empty(len(v),dtype=[(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ['red','green','blue']])
    for j,k in enumerate('xyz'):vertex[k]=v[:,j]
    for j,k in enumerate(['red','green','blue']):vertex[k]=rgb[:,j]
    face=np.empty(len(f),dtype=[('vertex_indices','<i4',(3,)),('class_id','<u2'),('instance_id','<u4'),('region_id','<u4'),('region_kind','u1')])
    face['vertex_indices']=f;face['class_id']=data['face_class_id'];face['instance_id']=ids;face['region_id']=regions;face['region_kind']=kinds
    PlyData([PlyElement.describe(vertex,'vertex'),PlyElement.describe(face,'face')],text=False).write(path/'mesh_instances.ply')
    palette=np.array([[145,145,145]]+[[round(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.72,.95)] for i in range(1,int(ids.max(initial=0))+1)],np.uint8)
    trimesh.Trimesh(v[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),vertex_colors=np.repeat(palette[ids],3,axis=0),process=False).export(path/'mesh_instances.glb')

def instances(job):
    c,out,deps,prefix=context(job);src=deps[prefix+'.semantic'];d=dict(np.load(src['arrays'],allow_pickle=False));catalog=read(src['vocabulary'])
    path,spec=assumptions(job['config']);opt=job['config']['size_segmentation'];factor=opt.get('factor',1.)
    canonical,inverse,mapping=canonical_labels(d['face_class_id'],catalog,spec['priors'],opt.get('class_aliases',{}))
    prepared=prepare(d['vertices'],d['faces'])
    main=solve(d['vertices'],d['faces'],spec['priors'],spec['settings'],semantic=canonical,factor=factor,prepared=prepared)
    for r in main['regions']:
        r['size_class_id']=r['class_id'];r['class_id']=inverse.get(r['class_id'],-1)
    # Florence instance arrays remain provenance, never seeds/IDs for this algorithm.
    for key in ['face_instance_id','instance_observations','instance_agreement']:
        if key in d:d['upstream_'+key]=d.pop(key)
    d.update(face_region_id=main['ids'],face_region_kind=main['kinds'],
        face_instance_id=np.where(main['kinds']==2,main['ids'],0).astype(np.uint32))
    variants={}
    if opt.get('compare_geometry_only',False):
        blind=solve(d['vertices'],d['faces'],spec['priors'],spec['settings'],semantic=None,factor=factor,prepared=prepared)
        d['geometry_size_region_id']=blind['ids'];d['geometry_size_kind']=blind['kinds']
        save(out/'geometry_size_regions.json',blind['regions']);variants['geometry_size']=blind['counts']
    for f in opt.get('sensitivity_factors',[]):
        run=solve(d['vertices'],d['faces'],spec['priors'],spec['settings'],semantic=canonical,factor=f,prepared=prepared)
        name='semantic_size_factor_'+str(f);d[name+'_region_id']=run['ids'];d[name+'_kind']=run['kinds'];variants[name]=run['counts']
        save(out/(name+'_regions.json'),run['regions'])
    np.savez_compressed(out/'mesh_arrays.npz',**d);save(out/'regions.json',main['regions']);save(out/'size_assumptions.json',spec)
    export(out,d,d['face_instance_id'],main['kinds'],main['ids'])
    summary=dict(instances=main['counts'].get('object_candidate',0),candidate_instances=main['counts'].get('object_candidate',0),
        size_references=len(spec['priors']),counts=main['counts'],coverage=main['coverage'],variants=variants,
        geometry_preserved=True,all_faces_have_region_id=True,discarded_faces=0,
        physical_instance_accuracy_measured=False,known_per_physical_instance=False,shape_specific_rules=False,
        method='semantic labels + assumed class size; full-mesh graph grouping',
        required_object_policy='object_candidate_only',residual_policy='retained_for_occlusion_collision_and_spatial_coverage',
        class_mapping=mapping,factor=factor,prior_sha256=sha(path),seed=spec['settings']['random_seed'])
    save(out/'summary.json',summary)
    result(out,arrays=str(out/'mesh_arrays.npz'),regions=str(out/'regions.json'),
        assumptions=str(out/'size_assumptions.json'),instance_mesh=str(out/'mesh_instances.glb'),
        instance_ply=str(out/'mesh_instances.ply'),summary=summary,vocabulary=src['vocabulary'],external_artifacts=[str(path.resolve())])
