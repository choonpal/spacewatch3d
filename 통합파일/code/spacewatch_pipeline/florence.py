"""One segmentation checkpoint: Florence OD/polygons, depth association, face IDs."""
import colorsys
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .io import ROOT,read,save,result,sha,verify
from .geometry import context,similarity,transform,export_semantic

CODE=ROOT/'vendor/florence_only'
VENDOR=ROOT/'vendor/discovery/segmentation_2d3d'

def helpers():
    for p in [CODE,VENDOR]:
        if str(p) not in sys.path:sys.path.insert(0,str(p))

def prepare(job):
    helpers()
    from scene_discovery.prepare import run
    c,out,deps,prefix=context(job);src=deps[prefix+'.reconstruct']
    capture=dict(c,source_run=src['source_run'],coordinates=src['coordinates'])
    manifest=Path(src['source_run'])/'unfiltered_result/correspondence/manifest.json'
    if manifest.exists():
        from geometry import build_index
        from .io import digest
        build_index(dict(source_run=src['source_run'],output=str(out)),
            dict(identity=digest([sha(manifest),sha(Path(src['source_run'])/'component_filter/keep_mask.npy')]),
                 source_frames=src['frames']))
        capture['source_index']=str(out/'source_index')
    run(dict(job['config']['discovery'],captures=[capture]),out)
    model=Path(job['config']['discovery']['models']['florence']['path'])
    model_files=sorted(str(p) for p in model.iterdir() if p.is_file())
    if not (model/'model.safetensors').exists():raise ValueError('Missing pinned Florence weights')
    result(out,data_root=str(out),prepared=str(out/'prepared.json'),
           external_artifacts=model_files+([c['video']] if c.get('video') else []))

def cached_inference(job,refined,out,prepared):
    old=job['config'].get('reuse_florence_run')
    if not old:return None
    old=Path(old);prefix=job['capture']['id'];stage=prefix+('.refine_infer' if refined else '.infer')
    completion=old/'stages'/stage/'complete.json'
    if not completion.exists():return None
    identity=read(old/'identity.json')
    if identity['config']['discovery']!=job['config']['discovery']:raise ValueError('Florence cache settings changed')
    for name in ['infer.py','repair.py','florence_core.py','florence_image.py']:
        key='vendor/florence_only/'+name
        if identity['code'].get(key)!=sha(CODE/name):raise ValueError('Florence cache producing code changed')
    receipt=read(completion);previous=read(receipt['result_path']);root=Path(previous['data_root'])
    before=read(root/'prepared.json')['scenes'][0]['views'];after=read(prepared)['scenes'][0]['views']
    def pixels(rows):return [(r['key'],r['sha256'],r['source_sha256'],r['frame'],r['view']) for r in rows]
    if pixels(before)!=pixels(after):raise ValueError('Florence cache pixels or projection changed')
    external=[]
    # The original prepare receipt pins every model file, not just its directory name.
    for path in [completion,old/'stages'/(prefix+'.prepare')/'complete.json']:
        for artifact in read(path)['artifacts']:
            verify(artifact);external.append(artifact['path'])
        external.append(str(path))
    for folder in ['predictions','od']:
        if (root/folder).exists():shutil.copytree(root/folder,out/folder)
    for name in ['inference_summary.json','repair_summary.json']:
        if (root/name).exists():shutil.copyfile(root/name,out/name)
    return dict(external_artifacts=external,reused_from=str(root),inference_reused=True)

def inference(job,refined=False):
    c,out,deps,prefix=context(job);opts=job['config']['discovery']
    prep=deps[prefix+('.refine_prepare' if refined else '.prepare')]['prepared']
    shutil.copyfile(prep,out/'prepared.json')
    rows=read(prep)['scenes'][0]['views']
    cached=cached_inference(job,refined,out,prep)
    if cached is not None:
        pass
    elif rows:
        python=job['config']['runtime']['florence_python']
        command=[python,str(CODE/'infer.py'),'--discovery',str(out),'--output',str(out),
                 '--model',opts['models']['florence']['path'],'--scene',prefix,'--fresh-od',
                 '--max-tokens',str(opts.get('max_tokens',1024))]
        subprocess.run(command,check=True)
        subprocess.run([python,str(CODE/'repair.py'),'--output',str(out)],check=True)
    else:
        save(out/'inference_summary.json',dict(prediction_keys=[],views=0,detections=0,
            max_tokens=opts.get('max_tokens',1024),model=opts['models']['florence']['path'],
            segmentation_models=['Florence-2'],other_segmentation_models_used=False))
    result(out,data_root=str(out),prepared=str(out/'prepared.json'),views=len(rows),
           source_frames=sorted(set(r['frame'] for r in rows)),summary=read(out/'inference_summary.json'),**(cached or {}))

def infer(job):inference(job)
def refine_infer(job):inference(job,True)

def merge_inputs(roots,out):
    scenes={};keys=[];summaries=[]
    for root in map(Path,roots):
        summary=read(root/'inference_summary.json');summaries.append(summary);keys+=summary['prediction_keys']
        for scene in read(root/'prepared.json')['scenes']:
            if scene['id'] not in scenes:scenes[scene['id']]=dict(scene,views=[])
            scenes[scene['id']]['views']+=scene['views']
        for folder in ['predictions','od']:
            if (root/folder).exists():shutil.copytree(root/folder,out/folder,dirs_exist_ok=True)
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate prediction key')
    for scene in scenes.values():scene['frames']=sorted(set(r['frame'] for r in scene['views']))
    save(out/'prepared.json',dict(scenes=list(scenes.values())))
    save(out/'inference_summary.json',dict(summaries[0],prediction_keys=keys,
        views=sum(s['views'] for s in summaries),detections=sum(s['detections'] for s in summaries),source_summaries=summaries))

def project(job,refined=False):
    helpers()
    import project as projection
    c,out,deps,prefix=context(job)
    sources=[deps[prefix+'.infer']['data_root']]
    if refined:sources.append(deps[prefix+'.refine_infer']['data_root'])
    merge_inputs(sources,out)
    opts=job['config']['discovery']
    projection.run(SimpleNamespace(discovery=str(out),output=str(out),vendor=str(VENDOR),scene=prefix,
        fov=opts.get('view_fov',100),min_points=opts.get('min_points',64),
        overlap=opts.get('association_overlap',.3),dynamic_labels='person'))
    result(out,data_root=str(out),points=str(out/'candidates.npz'),instances=str(out/'instances.json'),
           observations=str(out/'observations.json'),summary=read(out/'projection_summary.json'))

def observe(job):project(job)
def objects(job):project(job,True)

def refine_prepare(job):
    helpers()
    import prepare_refinement
    c,out,deps,prefix=context(job);opts=job['config']['discovery']
    base=deps[prefix+'.observe']['data_root'];observations=read(Path(base)/'observations.json')
    if opts.get('refine',True) and observations:
        video=c.get('video') or str(Path(deps[prefix+'.reconstruct']['source_run'])/'source/video.mp4')
        prepare_refinement.run(SimpleNamespace(baseline=base,output=str(out),vendor=str(VENDOR),video=video,
            scene=prefix,labels='',frame_offset=opts.get('refinement_frame_offset',2),angular_spacing=15.,fov=90.,
            observation_keys=[r['key'] for r in observations],max_directions=opts.get('max_directions_per_frame',6)))
    else:
        prepared=read(Path(base)/'prepared.json');prepared['scenes'][0].update(views=[],frames=[])
        save(out/'prepared.json',prepared)
        save(out/'refinement_plan.json',dict(views=0,reason='disabled_or_no_projected_observations'))
    result(out,data_root=str(out),prepared=str(out/'prepared.json'),plan=read(out/'refinement_plan.json'))

def discovery(job):
    out=Path(job['output']);deps=job['dependencies'];support=defaultdict(set);refs=defaultdict(list)
    evidence=[]
    for capture in job['config']['captures']:
        prefix=capture['id'];root=Path(deps[prefix+'.objects']['data_root'])
        observations=read(root/'observations.json')
        for r in observations:support[r['label']].add((prefix,r['frame']))
        for r in read(root/'instances.json'):
            if r['reference_eligible']:
                good=[observations[i]['dimensions'] for i in r['observation_ids'] if not observations[i]['reference_rejection']]
                refs[r['label']].append(dict(capture=prefix,candidate_id=r['candidate_id'],frames=r['complete_frames'],
                    dimensions=np.median(good,axis=0).tolist(),axis_max_min_ratio=r['size_ratio']))
        evidence.append(dict(capture=prefix,data_root=str(root)))
    names=sorted(k for k,v in support.items() if len(v)>=job['config']['discovery'].get('min_class_frames',2))
    if len(names)>254:raise ValueError('Too many classes for uint8 face class catalogue')
    structural={'wall','floor','ceiling','ground','road','sky'}
    catalog=[dict(id=i,name=name,rgb=[round(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.65,.95)],
        instance_eligible=name not in structural|{'person'},static_mesh=name!='person',
        distinct_frame_support=[dict(capture=a,frame=b) for a,b in sorted(support[name])],
        source='Florence-2 OD + polygon masks + reconstructed depth') for i,name in enumerate(names)]
    priors=dict(units='input_mesh_units',metric_scale_verified=all(deps[c['id']+'.reconstruct']['metric_scale_verified'] for c in job['config']['captures']),
        policy='Separate measured candidates; no universal category size or shape-based splitting',classes={})
    for name in names:
        # Keep separate references rather than equating all chair types/sizes.
        priors['classes'][name]=dict(references=refs[name],families=[dict(dimensions=r['dimensions'],references=[r]) for r in refs[name]])
    save(out/'classes.json',catalog);save(out/'size_priors.json',priors);save(out/'evidence_sources.json',evidence)
    save(out/'class_candidates.json',[dict(name=k,frames=len(v),accepted=k in names) for k,v in sorted(support.items())])
    result(out,data_root=str(out),vocabulary=str(out/'classes.json'),size_priors=str(out/'size_priors.json'),
        classes=names,size_prior_classes=[k for k in names if refs[k]],evidence_sources=evidence,status='complete',
        method='Florence-2 OD -> Florence-2 polygons -> exact depth projection -> distinct-frame association')

def unique_pixel_votes(frame,pixel,candidate):
    """One source pixel per frame; discard overlapping views that disagree."""
    key=np.rec.fromarrays([frame,pixel],names='frame,pixel');order=np.argsort(key,kind='stable')
    if not len(order):return order
    start=np.r_[0,np.flatnonzero(key[order][1:]!=key[order][:-1])+1]
    low=np.minimum.reduceat(candidate[order],start);high=np.maximum.reduceat(candidate[order],start)
    return order[start[low==high]]

def aggregate_votes_reference(face,frame,label,nfaces,unknown,minimum_frames=2,agreement=.6,minimum_pixels=1):
    """Pixel count chooses a within-frame winner; only distinct frames vote."""
    final=np.full(nfaces,unknown,np.int32);counts=np.zeros(nfaces,np.uint32);ratios=np.zeros(nfaces,np.float32)
    if not len(face):return final,counts,ratios
    values=np.column_stack([face,frame,label]);values,n=np.unique(values,axis=0,return_counts=True)
    # Resolve ties by suppressing the ambiguous frame, never by class/ID ordering.
    groups=defaultdict(list)
    for (f,fr,cl),num in zip(values,n):groups[(int(f),int(fr))].append((int(num),int(cl)))
    votes=defaultdict(list)
    for (f,fr),choices in groups.items():
        choices.sort(reverse=True)
        if choices[0][0]<minimum_pixels or (len(choices)>1 and choices[0][0]==choices[1][0]):continue
        votes[f].append(choices[0][1])
    for f,ls in votes.items():
        ids,ns=np.unique(ls,return_counts=True);order=np.argsort(ns);win=order[-1]
        counts[f]=len(ls);ratios[f]=ns[win]/len(ls)
        if ns[win]>=minimum_frames and ratios[f]>=agreement and (len(ns)==1 or ns[win]>ns[order[-2]]):final[f]=ids[win]
    return final,counts,ratios

from .semantic_aggregation import aggregate as aggregate_votes

def semantic(job):
    import open3d as o3d
    c,out,deps,prefix=context(job);mesh=deps[prefix+'.mesh'];source=deps[prefix+'.reconstruct']
    data=dict(np.load(mesh['geometry']));v,f=data['vertices'],data['faces'];nf=len(f)
    discovered=deps['discovery'];catalog=read(discovered['vocabulary']);byname={r['name']:r for r in catalog}
    objects=read(deps[prefix+'.objects']['instances']);lookup={r['candidate_id']:r for r in objects}
    p=np.load(deps[prefix+'.objects']['points']);points=p['points'];ids=p['candidate_id'];frames=p['frame'];pixels=p['pixel']
    take=unique_pixel_votes(frames,pixels,ids)
    take=take[np.array([lookup[int(i)]['label'] in byname and byname[lookup[int(i)]['label']]['static_mesh'] for i in ids[take]],bool)]
    points,ids,frames=points[take],ids[take],frames[take]
    tf,scale=similarity(read(source['coordinates'])['source_to_mesh'])
    poses=np.array([r['camera_to_world'] for r in read(Path(source['source_run'])/'result/camera_poses.json')])
    centers=transform(poses[:,:3,3],tf)
    raycaster=o3d.t.geometry.RaycastingScene(nthreads=4)
    raycaster.add_triangles(o3d.core.Tensor(v.astype('f4')),o3d.core.Tensor(f.astype('u4')))
    hitfaces=[];hitframes=[];hitids=[]
    for frame in np.unique(frames):
        select=frames==frame;pframe=points[select];origin=centers[frame];delta=pframe-origin
        distance=np.linalg.norm(delta,axis=1)
        rays=np.c_[np.broadcast_to(origin,pframe.shape),delta/np.maximum(distance[:,None],1e-12)].astype('f4')
        hit=raycaster.cast_rays(o3d.core.Tensor(rays),nthreads=4);depth=hit['t_hit'].numpy();fi=hit['primitive_ids'].numpy()
        valid=np.isfinite(depth)&(abs(depth-distance)<=.1*scale)&(distance>0)
        hitfaces.extend(fi[valid]);hitframes.extend([frame]*int(valid.sum()));hitids.extend(ids[select][valid])
    hitfaces=np.asarray(hitfaces,int);hitframes=np.asarray(hitframes,int);hitids=np.asarray(hitids,int)
    classes=np.array([byname[lookup[int(i)]['label']]['id'] for i in hitids],int)
    options=job['config'].get('semantic',{});minimum=options.get('minimum_frames',2)
    backend=options.get('aggregation_backend','numpy')
    if backend=='cuda':
        # Geometry env need not contain torch. Child inherits this stage's lease.
        votes=out/'aggregation_input.npz'
        np.savez(votes,face=hitfaces,frame=hitframes,classes=classes,instances=hitids)
        subprocess.run([job['config']['runtime']['reconstruction_python'],'-m','spacewatch_pipeline.semantic_aggregation',
            '--input',str(votes),'--output',str(out/'aggregation_output.npz'),
            '--faces',str(nf),'--minimum-frames',str(minimum)],check=True)
        with np.load(out/'aggregation_output.npz') as a:
            labels,counts,ratios=(a['classes_'+k] for k in ['labels','counts','ratios'])
            instance_ids,icounts,iratios=(a['instances_'+k] for k in ['labels','counts','ratios'])
    else:
        labels,counts,ratios=aggregate_votes(hitfaces,hitframes,classes,nf,255,minimum,backend=backend)
        instance_ids,icounts,iratios=aggregate_votes(hitfaces,hitframes,hitids,nf,0,minimum,backend=backend)
    eligible=np.array([r['id'] for r in catalog if r['instance_eligible']],int)
    instance_ids[~np.isin(labels,eligible)]=0
    for iid in np.unique(instance_ids):
        if iid:instance_ids[(instance_ids==iid)&(labels!=byname[lookup[int(iid)]['label']]['id'])]=0
    data.update(face_class_id=labels.astype('u2'),face_instance_id=instance_ids.astype('u4'),
        observations=counts,vote_agreement=ratios,instance_observations=icounts,instance_agreement=iratios)
    np.savez_compressed(out/'mesh_arrays.npz',**data)
    report=dict(faces=nf,labeled_faces=int((labels!=255).sum()),instance_faces=int((instance_ids>0).sum()),
        candidate_instances=len(objects),mesh_instances=len(np.unique(instance_ids[instance_ids>0])),
        independent_frames=len(np.unique(hitframes)),depth_checked_pixels=len(hitfaces),minimum_frames=minimum,
        geometry_unchanged=True,segmentation_models=['Florence-2'],physical_instance_accuracy_measured=False,
        aggregation_backend=backend)
    save(out/'projection_report.json',report)
    result(out,arrays=str(out/'mesh_arrays.npz'),coordinates=mesh['coordinates'],vocabulary=discovered['vocabulary'],
        report=report,**export_semantic(out,data,catalog))

def instances(job):
    import trimesh
    from plyfile import PlyData,PlyElement
    c,out,deps,prefix=context(job);semantic=deps[prefix+'.semantic'];data=dict(np.load(semantic['arrays']))
    v,f,labels,ids=[data[k] for k in ['vertices','faces','face_class_id','face_instance_id']]
    np.savez_compressed(out/'mesh_arrays.npz',**data)
    vertex=np.empty(len(v),dtype=[(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ['red','green','blue']])
    for j,k in enumerate('xyz'):vertex[k]=v[:,j]
    for j,k in enumerate(['red','green','blue']):vertex[k]=data['rgb'][:,j]
    face=np.empty(len(f),dtype=[('vertex_indices','<i4',(3,)),('class_id','<u2'),('instance_id','<u4')])
    face['vertex_indices']=f;face['class_id']=labels;face['instance_id']=ids
    ply=out/'mesh_instances.ply';PlyData([PlyElement.describe(vertex,'vertex'),PlyElement.describe(face,'face')],text=False).write(str(ply))
    palette=np.array([[145,145,145]]+[[round(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.72,.95)] for i in range(1,int(ids.max(initial=0))+1)],np.uint8)
    trimesh.Trimesh(v[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),vertex_colors=np.repeat(palette[ids],3,axis=0),process=False).export(out/'mesh_instances.glb')
    checks=dict(no_unknown_instances=not bool(ids[labels==255].any()),
        no_cross_class_instances=all(len(np.unique(labels[ids==i]))==1 for i in np.unique(ids) if i),
        ply_ids_exact=np.array_equal(PlyData.read(ply)['face'].data['instance_id'],ids))
    if not all(checks.values()):raise ValueError('Instance consistency failure')
    summary=dict(instances=len(np.unique(ids[ids>0])),candidate_instances=deps[prefix+'.objects']['summary']['candidate_instances'],
        size_references=deps[prefix+'.objects']['summary']['reference_eligible'],checks=checks,geometry_preserved=True,
        shape_specific_rules=False,physical_instance_accuracy_measured=False,method='Florence polygons + multi-frame 3D overlap')
    save(out/'summary.json',summary)
    result(out,arrays=str(out/'mesh_arrays.npz'),instance_mesh=str(out/'mesh_instances.glb'),instance_ply=str(ply),summary=summary,vocabulary=semantic['vocabulary'])
