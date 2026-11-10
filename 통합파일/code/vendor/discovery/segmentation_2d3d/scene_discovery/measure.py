"""Measure pilot masks in the original reconstructed coordinate system."""
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import read,save,sha,VIEWS,erp_to_view


def dimensions(points):
    p=np.asarray(points,float)
    if len(p)<8 or not np.isfinite(p).all():raise ValueError('Need at least eight finite points')
    best=None
    for angle in np.linspace(0,np.pi/2,91):
        c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        xy=p[:,:2]@rotation
        q=np.quantile(xy,[.01,.99],axis=0);extent=q[1]-q[0]
        cost=np.prod(extent)
        if best is None or cost<best[0]:best=(cost,sorted(extent))
    h=np.diff(np.quantile(p[:,2],[.01,.99]))[0]
    return np.array([*best[1],h])


def clean_points(points):
    """Remove distant mask/depth leakage without assuming a metric object size."""
    p=np.asarray(points,float)
    if len(p)<32:return p,1.
    q=np.quantile(p,[.01,.99],axis=0);diagonal=np.linalg.norm(q[1]-q[0])
    voxel=max(diagonal/60,1e-8)
    _,ids=np.unique(np.floor(p/voxel).astype(np.int64),axis=0,return_index=True)
    sample=p[ids];tree=cKDTree(sample)
    pairs=tree.query_pairs(voxel*3.2,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(sample),len(sample)))
    _,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels);main=counts.argmax()
    _,near=tree.query(p,k=1)
    keep=labels[near]==main
    return p[keep],float(keep.mean())


def overlap(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0])).prod()
    return float(intersection/max(1e-12,min(np.prod(a[1]-a[0]),np.prod(b[1]-b[0]))))


def associate(observations):
    """Complete-link spatial association; simultaneous separate detections never merge."""
    groups=[]
    for i,row in sorted(enumerate(observations),key=lambda x:(x[1]['scene'],x[1]['name'],x[1]['frame'],x[1]['view'],x[1]['index'])):
        eligible=[]
        for gi,group in enumerate(groups):
            others=[observations[j] for j in group]
            if any(o['scene']!=row['scene'] or o['name']!=row['name'] for o in others):continue
            if any((o['frame'],o['view'])==(row['frame'],row['view']) for o in others):continue
            scores=[overlap(o['bounds'],row['bounds']) for o in others]
            if min(scores)>=.30:eligible.append((float(np.mean(scores)),gi))
        if eligible:groups[max(eligible)[1]].append(i)
        else:groups.append([i])
    return groups


def build_priors(instances,catalog,ratio=1.35):
    priors={}
    for cls in catalog:
        refs=[r for r in instances if r['name']==cls['name'] and r['reference_eligible']]
        families=[]
        for ref in sorted(refs,key=lambda r:(r['dimensions'][0]*r['dimensions'][1],r['instance_id'])):
            dims=np.asarray(ref['dimensions']);match=None
            for family in families:
                values=np.array([r['dimensions'] for r in family]+[dims.tolist()])
                if np.all(values.max(0)/np.maximum(values.min(0),1e-8)<=ratio):match=family;break
            if match is None:families.append([ref])
            else:match.append(ref)
        records=[]
        for family in families:
            d=np.array([r['dimensions'] for r in family]);observed=np.array([o for r in family for o in r['observed_dimensions']])
            lo=np.minimum(d.min(0),np.quantile(observed,.1,axis=0));hi=np.maximum(d.max(0),np.quantile(observed,.9,axis=0))
            records.append(dict(dimensions=np.median(d,axis=0).tolist(),lower=lo.tolist(),upper=hi.tolist(),
                                references=[r['instance_id'] for r in family],independent_objects=len(family),
                                width_interval=[float(lo[0]),float(hi[1])],
                                status='measured',confidence='multiple_objects' if len(family)>1 else 'single_object_multiview'))
        priors[cls['name']]=dict(class_id=cls['id'],status='measured' if records else 'abstained',
                                 reason=None if records else 'no_consistent_multiview_reference',families=records)
    return priors


def run(config,output):
    output=Path(output);catalog=read(output/'classes.json');classes={c['name']:c for c in catalog}
    # Reuse the reconstruction's exact pixel-to-point transform, not image-box depth guesses.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from geometry import source_frames
    observations=[];clouds=[];prepared=read(output/('refined.json' if (output/'refined.json').exists() else 'prepared.json'))
    for scene in prepared['scenes']:
        coordinate=read(scene['coordinates']);tf=np.array(coordinate['source_to_mesh'])
        if not np.allclose(coordinate.get('up',[0,0,1]),[0,0,1]):raise ValueError('Size estimator requires a Z-up transform')
        grouped=defaultdict(list)
        for row in scene['views']:grouped[row['frame']].append(row)
        maps={}
        for frame,xyz,valid,_ in source_frames(dict(source_run=scene['source_run']),selected=scene['frames']):
            p=xyz@tf[:3,:3].T+tf[:3,3];pixels=np.flatnonzero(valid)
            for view in grouped[frame]:
                record=read(output/'grounded'/f'{view["key"]}.json')
                with np.load(output/'grounded'/f'{view["key"]}.npz') as z:
                    shape=tuple(z['shape']);masks=np.unpackbits(z['masks_packed'],axis=1,count=shape[1]*shape[2]).reshape(shape).astype(bool)
                key=(valid.shape,view['view']['name'])
                if key not in maps:maps[key]=erp_to_view(valid.shape,view['view'],config.get('view_size',512),view['view'].get('fov',config.get('view_fov',100)))
                visible,index=maps[key]
                for i,(det,mask) in enumerate(zip(record['detections'],masks)):
                    cls=classes.get(det['name'])
                    if not cls or not cls['instance_eligible']:continue
                    eroded=binary_erosion(mask,iterations=1)
                    selected=visible[pixels]&eroded.ravel()[index[pixels]]
                    raw=p[selected]
                    if len(raw)<64:continue
                    cloud,purity=clean_points(raw)
                    if len(cloud)<64:continue
                    dims=dimensions(cloud)
                    if dims.min()<=1e-5:continue
                    q=np.quantile(cloud,[.01,.99],axis=0)
                    reasons=[]
                    if det['border_touch']:reasons.append('truncated_at_view_border')
                    if purity<.75:reasons.append('fragmented_depth_support')
                    if det['stability']<.85:reasons.append('unstable_mask')
                    if det.get('person_boundary_contact',0)>.08:reasons.append('possibly_occluded_by_person')
                    if det['pixels']/(mask.size)>.75:reasons.append('mask_covers_most_of_view')
                    observations.append(dict(scene=scene['id'],frame=frame,view=view['view']['name'],index=i,key=view['key'],
                        name=det['name'],class_id=cls['id'],points=len(cloud),purity=purity,dimensions=dims.tolist(),
                        bounds=q.tolist(),center=np.median(cloud,axis=0).tolist(),reference_rejection=reasons,
                        dino_score=det['dino_score'],sam_score=det['sam_score']))
                    step=max(1,len(cloud)//3000);clouds.append(cloud[::step].astype('f4'))
            print('MEASURE',scene['id'],frame,'observations',len(observations),flush=True)
    instances=[]
    for group in associate(observations):
        members=[observations[i] for i in group];good=[r for r in members if not r['reference_rejection']]
        allpoints=np.concatenate([clouds[i] for i in group]);dims=dimensions(allpoints)
        reasons=[];good_frames=set(r['frame'] for r in good)
        if len(good_frames)<2:reasons.append('fewer_than_two_complete_frames')
        if good:
            ds=np.array([r['dimensions'] for r in good]);ratios=ds.max(0)/np.maximum(ds.min(0),1e-8)
            if ratios.max()>1.4:reasons.append('inconsistent_dimensions_across_views')
        else:ds=np.array([dims])
        iid=len(instances)+1
        path=output/'instance_points'/f'{iid:05d}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,points=allpoints)
        instances.append(dict(instance_id=iid,scene=members[0]['scene'],name=members[0]['name'],class_id=members[0]['class_id'],
                              observation_indices=group,frames=sorted(set(r['frame'] for r in members)),
                              complete_frames=sorted(good_frames),dimensions=np.median(ds,axis=0).tolist(),
                              observed_dimensions=ds.tolist(),union_dimensions=dims.tolist(),
                              bounds=np.quantile(allpoints,[.01,.99],axis=0).tolist(),
                              reference_eligible=not reasons,reference_rejection=reasons,points_path=str(path)))
    priors=build_priors(instances,catalog)
    save(output/'observations.json',observations);save(output/'instances.json',instances)
    save(output/'size_priors.json',dict(units='input_mesh_units',metric_scale_verified=all(read(s['coordinates']).get('metric_scale_verified',False) for s in prepared['scenes']),
                                      dimension_order=['short_horizontal','long_horizontal','height'],classes=priors,
                                      method='SAM2 masks -> exact reconstruction pixels -> robust 3D dimensions; class-specific size families',
                                      bounds_are_statistical_confidence_intervals=False))


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])
