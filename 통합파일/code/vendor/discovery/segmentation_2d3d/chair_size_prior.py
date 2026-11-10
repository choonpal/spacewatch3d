"""Experimental size-regularized cuts of elongated, fused chair mesh regions.

This partitions existing triangles; it does not reconstruct missing surfaces or
infer physical object identity. All lengths use the input mesh's coordinate unit.
"""
from itertools import combinations
import numpy as np


def horizontal_box(points):
    """Deterministic minimum-area XY box, searched at 0.25 degree increments."""
    xy=np.asarray(points,dtype=np.float64)[:,:2]
    angles=np.deg2rad(np.arange(0,90,.25))
    candidates=[]
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle)
        p=xy@np.array([[c,-s],[s,c]])
        span=np.ptp(p,axis=0)
        candidates.append((float(np.prod(span)),span,angle))
    _,span,angle=min(candidates,key=lambda x:x[0])
    return {'sides':sorted(span.tolist()),'angle_degrees':float(np.rad2deg(angle)),
            'effective_width':float(np.sqrt(np.prod(span)))}


def split_chair_region(vertices,faces,width,geometry_weight=.65,tolerance=.20,
                       min_ratio=.60,max_ratio=1.40,min_faces=8,min_area=.04):
    """Choose count and transverse cuts by dynamic programming.

    E = sum(((interval_width - width)/(tolerance*width))**2)
        + geometry_weight * sum(normalized shared-vertex cut weights).
    An edge's weight is the geometric mean of its incident triangle areas.
    Candidate cut positions are uniformly spaced at <= width/30.
    A compact region is returned unchanged. Subsets are original face indices
    relative to `faces`; no vertex or face is edited. `width` is an estimated
    effective horizontal footprint, not a measured chair-seat width.
    """
    if not np.isfinite(width) or width<=0:raise ValueError('width must be positive')
    if not 0<min_ratio<1<max_ratio:raise ValueError('invalid size bounds')
    if tolerance<=0 or geometry_weight<0:raise ValueError('invalid objective parameters')
    v=np.asarray(vertices);f=np.asarray(faces)
    if not len(f):return np.empty(0,dtype=np.int32),{'count':0,'reason':'empty'}
    tri=v[f];cent=tri.mean(1).astype(np.float64)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if area.sum()<=0:return np.zeros(len(f),np.int32),{'count':1,'reason':'zero_area'}
    center=np.average(cent,axis=0,weights=area)
    cov=((cent[:,:2]-center[:2])*area[:,None]).T@(cent[:,:2]-center[:2])/area.sum()
    eig,axes=np.linalg.eigh(cov);axis=axes[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    tp=(tri[:,:,:2]-center[:2])@axis;t=(cent[:,:2]-center[:2])@axis
    lo,hi=float(tp.min()),float(tp.max());length=hi-lo
    elongation=float(np.sqrt(eig[-1]/max(eig[0],1e-12)))
    info=dict(width_prior=float(width),axis=axis.tolist(),center=center.tolist(),
              length=length,elongation=elongation,geometry_weight=geometry_weight,
              tolerance=tolerance,bounds_ratio=[min_ratio,max_ratio])
    if length<1.7*width or elongation<2.0:
        return np.zeros(len(f),np.int32),dict(info,count=1,reason='compact_region')
    grid=np.linspace(lo,hi,max(3,int(np.ceil(length/(width/30)))+1))
    incident={}
    for i,face in enumerate(f):
        for vertex in face:incident.setdefault(int(vertex),[]).append(i)
    edges=set()
    for adjacent in incident.values():edges.update(combinations(sorted(set(adjacent)),2))
    edges=np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2)
    cut_cost=np.zeros(len(grid))
    if len(edges):
        left=np.minimum(t[edges[:,0]],t[edges[:,1]]);right=np.maximum(t[edges[:,0]],t[edges[:,1]])
        weights=np.sqrt(area[edges[:,0]]*area[edges[:,1]])
        for j,boundary in enumerate(grid):cut_cost[j]=weights[(left<boundary)&(right>=boundary)].sum()
    scale=float(np.quantile(cut_cost[1:-1],.90))
    if scale>0:cut_cost/=scale
    order=np.argsort(t);st=t[order];cumulative=np.r_[0,np.cumsum(area[order])]
    offsets=np.searchsorted(st,grid,side='left');offsets[0]=0;offsets[-1]=len(t)
    dp=np.full(len(grid),np.inf);prev=np.full(len(grid),-1,dtype=np.int32);dp[0]=0
    for j in range(1,len(grid)):
        for i in range(j):
            span=grid[j]-grid[i]
            if span<min_ratio*width or span>max_ratio*width:continue
            if offsets[j]-offsets[i]<min_faces or cumulative[offsets[j]]-cumulative[offsets[i]]<min_area:continue
            cost=dp[i]+((span-width)/(tolerance*width))**2
            if j<len(grid)-1:cost+=geometry_weight*cut_cost[j]
            if cost<dp[j]:dp[j]=cost;prev[j]=i
    if prev[-1]<0:return np.zeros(len(f),np.int32),dict(info,count=1,reason='no_feasible_partition')
    indices=[len(grid)-1]
    while indices[-1]>0:indices.append(int(prev[indices[-1]]))
    indices=indices[::-1];boundaries=grid[indices]
    labels=np.searchsorted(boundaries[1:-1],t,side='right').astype(np.int32)
    # Explicit equal-interval ablation at this selected count.
    equal=np.linspace(lo,hi,len(boundaries))
    equal_cost=float(np.interp(equal[1:-1],grid,cut_cost).sum())
    info.update(count=len(boundaries)-1,reason='size_regularized_partition',
                boundaries=boundaries.tolist(),interval_widths=np.diff(boundaries).tolist(),
                boundary_cost=float(cut_cost[indices[1:-1]].sum()),equal_interval_boundary_cost=equal_cost,
                objective=float(dp[-1]),candidate_positions=grid.tolist(),candidate_costs=cut_cost.tolist())
    return labels,info
