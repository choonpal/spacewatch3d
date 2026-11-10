"""Project the new vocabulary onto the integrated mesh, then apply measured size priors."""
from pathlib import Path
import sys
import numpy as np
from .common import read,save,sha


def resize_nearest(array,height,width):
    y=np.floor(np.arange(height)*array.shape[0]/height).astype(int)
    x=np.floor(np.arange(width)*array.shape[1]/width).astype(int)
    return array[y[:,None],x[None,:]]


def split_with_priors(vertices,faces,labels,baseline,catalog,priors):
    from segmentation_2d3d.chair_size_prior import split_chair_region,horizontal_box
    names={c['id']:c['name'] for c in catalog};result=baseline.copy();records=[]
    next_id=int(baseline.max(initial=0))+1
    for parent in np.unique(baseline):
        if parent==0:continue
        fi=np.flatnonzero(baseline==parent);name=names[int(labels[fi[0]])]
        families=priors['classes'][name]['families']
        if not families:continue
        p=vertices[np.unique(faces[fi])];box=horizontal_box(p);short,long=box['sides'];height=float(np.ptp(p[:,2]))
        if long<2*short:continue
        compatible=[]
        for family in families:
            d=np.array(family['dimensions']);lo=np.array(family['lower']);hi=np.array(family['upper'])
            if .5*lo[0]<=short<=1.6*hi[1] and .55*lo[2]<=height<=1.6*hi[2]:
                distance=abs(np.log(max(short,1e-8)/np.sqrt(d[0]*d[1])))+abs(np.log(max(height,1e-8)/d[2]))
                compatible.append((distance,family))
        if not compatible:continue
        compatible.sort(key=lambda x:x[0])
        if len(compatible)>1 and compatible[1][0]-compatible[0][0]<.25:
            records.append(dict(parent_id=int(parent),name=name,status='abstained',reason='ambiguous_size_family'));continue
        family=compatible[0][1];lo,hi=family['lower'],family['upper']
        # Footprint-equivalent width, computed from measured dimensions in code.
        lower=float(np.sqrt(lo[0]*lo[1]));upper=float(np.sqrt(hi[0]*hi[1]));width=float(np.sqrt(family['dimensions'][0]*family['dimensions'][1]))
        counts=[];center_partition=None;info=None
        for w in np.linspace(lower,upper,5).tolist()+[width]:
            partition,details=split_chair_region(vertices,faces[fi],w,min_area=.05*w*w)
            counts.append(details['count']);center_partition=partition;info=details
        record=dict(parent_id=int(parent),name=name,width=width,width_interval=[lower,upper],counts=counts,reference_ids=family['references'])
        if len(set(counts))>1:
            record.update(status='abstained',reason='count_changes_with_measured_size_uncertainty')
        elif counts[-1]<=1:
            record.update(status='unchanged',reason=info['reason'])
        else:
            child_ids=[]
            for part in np.unique(center_partition):
                result[fi[center_partition==part]]=next_id;child_ids.append(next_id);next_id+=1
            record.update(status='split',children=child_ids,axis=info['axis'],boundaries=info['boundaries'])
        records.append(record)
    return result,records


def run(config,output):
    import open3d as o3d
    import trimesh
    from segmentation_2d3d.mesh_instances import assign_mesh_ids
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from geometry import source_frames
    output=Path(output);catalog=read(output/'classes.json');priors=read(output/'size_priors.json')
    nc=len(catalog);summaries=[]
    for capture in config['captures']:
        dest=output/'meshes'/capture['id'];dest.mkdir(parents=True,exist_ok=True)
        data=dict(np.load(capture['mesh']));v,f=data['vertices'],data['faces'];nf=len(f)
        coordinates=read(capture['coordinates']);tf=np.array(coordinates['source_to_mesh'])
        scale=float(np.cbrt(abs(np.linalg.det(tf[:3,:3]))))
        poses=np.array([p['camera_to_world'] for p in read(Path(capture['source_run'])/'result/camera_poses.json')])
        caster=o3d.t.geometry.RaycastingScene(nthreads=2)
        caster.add_triangles(o3d.core.Tensor(v.astype('f4')),o3d.core.Tensor(f.astype('u4')))
        vote=np.zeros((nc,nf),np.float32);observations=np.zeros(nf,np.uint32);geometry_observations=np.zeros(nf,np.uint32)
        frames=0;accepted_count=0
        for frame,xyz,valid,_ in source_frames(dict(source_run=capture['source_run'])):
            points=xyz@tf[:3,:3].T+tf[:3,3];origin=poses[frame,:3,3]@tf[:3,:3].T+tf[:3,3]
            delta=points-origin;distance=np.linalg.norm(delta,axis=1)
            rays=np.column_stack([np.broadcast_to(origin,points.shape),delta/np.maximum(distance[:,None],1e-12)]).astype('f4')
            hit=caster.cast_rays(o3d.core.Tensor(rays),nthreads=2)
            depth=hit['t_hit'].numpy();face=hit['primitive_ids'].numpy()
            accepted=np.isfinite(depth)&(distance>0)&(np.abs(depth-distance)<=config.get('depth_tolerance_source_units',.1)*scale)
            pixels=np.flatnonzero(valid)[accepted];face=face[accepted].astype(np.int64)
            geometry_observations[np.unique(face)]+=1
            with np.load(output/'segmentation'/capture['id']/f'{frame:05d}.npz') as seg:
                h,w=valid.shape
                label=resize_nearest(seg['labels'],h,w).ravel()[pixels]
                support=resize_nearest(seg['support_fraction'].astype('f4'),h,w).ravel()[pixels]
                margin=resize_nearest(seg['margin_fraction'].astype('f4'),h,w).ravel()[pixels]
            keep=(support>=config.get('semantic_min_support',.25))&(margin>=config.get('semantic_min_margin',.05))
            face,label,support=face[keep],label[keep],support[keep]
            denom=np.bincount(face,minlength=nf)
            observations[np.flatnonzero(denom)]+=1
            for cid in np.unique(label):
                selected=label==cid
                value=np.bincount(face[selected],weights=support[selected],minlength=nf)
                vote[int(cid)]+=value/np.maximum(denom,1)
            frames+=1;accepted_count+=len(pixels)
            if frame%10==0:print('PROJECT',capture['id'],frame,'accepted',len(pixels),flush=True)
        winner=vote.argmax(0);best=vote[winner,np.arange(nf)];agreement=best/np.maximum(vote.sum(0),1e-9)
        labels=np.where((observations>=2)&(agreement>=.55)&(best>0),winner,255).astype('u2')
        effective=[dict(c,static_mesh=c.get('instance_eligible',False)) for c in catalog]
        baseline,_,_,area=assign_mesh_ids(v,f,labels,effective,min_area=.01*scale*scale)
        ids,splits=split_with_priors(v,f,labels,baseline,catalog,priors)
        arrays=dict(data,face_class_id=labels,face_instance_id=ids,baseline_face_instance_id=baseline,
                    vote_agreement=agreement.astype('f4'),observations=observations,geometric_observations=geometry_observations)
        np.savez_compressed(dest/'mesh_arrays.npz',**arrays)
        save(dest/'splits.json',splits)
        # Export exact face attributes in PLY; GLB is a color visualization.
        from plyfile import PlyData,PlyElement
        vertex=np.empty(len(v),dtype=[(c,'<f4') for c in 'xyz']+[(c,'u1') for c in ['red','green','blue']])
        for j,k in enumerate('xyz'):vertex[k]=v[:,j]
        for j,k in enumerate(['red','green','blue']):vertex[k]=data['rgb'][:,j]
        triangle=np.empty(nf,dtype=[('vertex_indices','<i4',(3,)),('class_id','<u2'),('instance_id','<u4')])
        triangle['vertex_indices']=f;triangle['class_id']=labels;triangle['instance_id']=ids
        PlyData([PlyElement.describe(vertex,'vertex'),PlyElement.describe(triangle,'face')],text=False).write(str(dest/'mesh_instances.ply'))
        palette=np.full((256,3),145,np.uint8)
        for c in catalog:palette[c['id']]=c['rgb']
        import colorsys
        instance_palette=np.array([[145,145,145]]+[[int(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.72,.95)] for i in range(1,int(ids.max(initial=0))+1)],np.uint8)
        for name,colors in [('semantic',palette[labels]),('instances',instance_palette[ids])]:
            trimesh.Trimesh(v[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),vertex_colors=np.repeat(colors,3,axis=0),process=False).export(dest/f'mesh_{name}.glb')
        row=dict(scene=capture['id'],frames=frames,vertices=len(v),faces=nf,accepted_rays=accepted_count,
                 labeled_faces=int((labels!=255).sum()),assigned_faces=int((ids>0).sum()),
                 baseline_instances=int(len(np.unique(baseline[baseline>0]))),instances=int(len(np.unique(ids[ids>0]))),
                 successful_size_splits=sum(r['status']=='split' for r in splits),geometry_sha256=sha(capture['mesh']),
                 vocabulary_sha256=sha(output/'classes.json'),class_face_counts={c['name']:int((labels==c['id']).sum()) for c in catalog})
        save(dest/'summary.json',row);summaries.append(row)
    save(output/'projection_summary.json',summaries)


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])
