"""Physical-area spatial strata; original occlusion/collision meshes stay intact."""
import numpy as np
from .models import InputError

def partition_signature(scene, config):
    if not config.spatially_stratified_sampling:
        return None
    origin=scene.floor.bounds[:2] if scene.fallback_tile_origin_xy_m is None else scene.fallback_tile_origin_xy_m
    return (float(config.tile_m),tuple(map(float,origin)),scene.fallback_tile_boundary_policy)

def _snap(values, physical, origin, tile_m, policy):
    if policy != 'roundoff_snap':
        return values
    tolerance=8*np.finfo(float).eps*(np.abs(physical)+np.abs(origin)+tile_m)/tile_m
    nearest=np.rint(values)
    return np.where(np.abs(values-nearest)<=tolerance,nearest,values)

def _clip(triangle, low, high):
    polygon=list(triangle)
    for axis,bound,above in ((0,low[0],True),(0,high[0],False),(1,low[1],True),(1,high[1],False)):
        clipped=[]
        if not polygon:
            break
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            ia,ib=(a[axis]>=bound,b[axis]>=bound) if above else (a[axis]<=bound,b[axis]<=bound)
            if ia:
                clipped.append(a)
            if ia != ib:
                p=a+(b-a)*((bound-a[axis])/(b[axis]-a[axis]))
                p[axis]=bound
                clipped.append(p)
        polygon=clipped
    if len(polygon)<3:
        return []
    return [np.array([polygon[0],polygon[i],polygon[i+1]]) for i in range(1,len(polygon)-1)]

def spatialize_strata(strata, scene, config):
    """Clip only grid-crossing residual faces; group ordinary faces in arrays.

    No area cutoff deletes a patch. A vertical face exactly on a tile boundary
    belongs to its positive-side tile, as in the legacy coordinate convention.
    Sampling alone uses these pieces; all ray tests still use original faces.
    """
    signature=partition_signature(scene,config)
    if signature is None:
        return tuple(strata)
    tile_m,origin,policy=signature
    origin=np.asarray(origin)
    declared={rid for rid,_ in scene.regions}
    output=[]
    count=0
    for gi,rid,triangles,areas,total in strata:
        if rid is not None:
            output.append((gi,rid,triangles,areas,total))
            count+=len(triangles)
            continue
        low_xy,high_xy=triangles[:,:,:2].min(axis=1),triangles[:,:,:2].max(axis=1)
        low_coord=_snap((low_xy-origin)/tile_m,low_xy,origin,tile_m,policy)
        high_coord=_snap((high_xy-origin)/tile_m,high_xy,origin,tile_m,policy)
        lower=np.floor(low_coord).astype(np.int64)
        upper=np.maximum(lower,np.ceil(high_coord).astype(np.int64)-1)
        whole=np.all(lower==upper,axis=1)
        buckets={}
        indices=np.flatnonzero(whole)
        if len(indices):
            keys,inverse=np.unique(lower[indices],axis=0,return_inverse=True)
            order=np.argsort(inverse,kind='stable')
            starts=np.r_[0,np.flatnonzero(inverse[order][1:]!=inverse[order][:-1])+1]
            ends=np.r_[starts[1:],len(order)]
            for key,a,b in zip(keys,starts,ends):
                buckets[tuple(key)]=[triangles[indices[order[a:b]]]]
        crossing={}
        for index in np.flatnonzero(~whole):
            for x in range(int(lower[index,0]),int(upper[index,0])+1):
                for y in range(int(lower[index,1]),int(upper[index,1])+1):
                    key=(x,y)
                    low=origin+np.array(key)*tile_m
                    pieces=_clip(triangles[index],low,low+tile_m)
                    if pieces:
                        crossing.setdefault(key,[]).extend(pieces)
        for key,pieces in crossing.items():
            buckets.setdefault(key,[]).append(np.asarray(pieces))
        subtotal=0.
        for key,chunks in sorted(buckets.items()):
            t=np.concatenate(chunks)
            area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
            positive=area>0
            if not positive.any():
                continue
            t,area=t[positive],area[positive]
            count+=len(t)
            if count>config.max_triangles:
                raise InputError('max_triangles exceeded by spatial surface partition; explicitly raise the computation budget')
            name='tile_%d_%d'%key
            while name in declared:
                name='unnamed_'+name
            value=float(area.sum())
            subtotal+=value
            output.append((gi,name,t,area,value))
        if not np.isclose(total,subtotal,rtol=1e-9,atol=1e-10):
            raise InputError('Spatial surface partition failed area conservation')
    if count>config.max_triangles:
        raise InputError('max_triangles exceeded by spatial surface partition')
    return tuple(output)
