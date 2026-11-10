"""Full-mesh size-constrained segmentation. Every original face receives a region ID.

Semantic mode uses class labels, never old instance IDs. Blind mode receives only
vertices, faces and a size-template catalog; semantic labels are used AFTERWARD
for a proxy audit, not for segmentation. Candidate size fit is not physical GT.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def geometry(v, f):
    t=v[f].astype(np.float64); c=t.mean(1)
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); norm=np.linalg.norm(cross,axis=1)
    n=cross/np.maximum(norm[:,None],1e-30)
    return c,n,norm/2


def adjacency(f):
    """Triangle edge adjacency, including a chain for non-manifold edges."""
    edges=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    fi=np.repeat(np.arange(len(f)),3); order=np.lexsort((edges[:,1],edges[:,0]))
    same=np.all(edges[order[1:]]==edges[order[:-1]],axis=1)
    pairs=np.sort(np.column_stack((fi[order[1:]][same],fi[order[:-1]][same])),axis=1)
    return np.unique(pairs,axis=0)


def plane_regions(c,n,area,allowed,settings,seed):
    """Area-weighted deterministic RANSAC; all inputs are geometry-only."""
    rng=np.random.default_rng(seed); remaining=allowed.copy(); result=[]
    cosine=np.cos(np.deg2rad(settings['plane_normal_degrees']))
    distance=settings['plane_distance']
    for iteration in range(settings['max_planes']):
        available=np.flatnonzero(remaining & (area>1e-12))
        if len(available)<30 or area[available].sum()<settings['minimum_plane_area']: break
        sample=rng.choice(available,size=min(64,len(available)),replace=False,p=area[available]/area[available].sum())
        cn=c[available]; nn=n[available]; aa=area[available]
        normals=n[sample]; offsets=np.einsum('ij,ij->i',c[sample],normals)
        matches=(np.abs(cn@normals.T-offsets)<=distance)&(np.abs(nn@normals.T)>=cosine)
        support=(matches*aa[:,None]).sum(0); best=int(np.argmax(support))
        if support[best]<settings['minimum_plane_area']: break
        fi=available[matches[:,best]]
        center=np.average(c[fi],axis=0,weights=area[fi]); delta=c[fi]-center
        _,vectors=np.linalg.eigh((delta*area[fi,None]).T@delta/area[fi].sum())
        normal=vectors[:,0]
        fi=available[(np.abs((cn-center)@normal)<=distance)&(np.abs(nn@normal)>=cosine)]
        if area[fi].sum()<settings['minimum_plane_area']:
            fi=available[matches[:,best]];normal=normals[best]
        remaining[fi]=False
        result.append(dict(faces=fi,normal=normal.tolist(),area=float(area[fi].sum()),
                           center=np.average(c[fi],axis=0,weights=area[fi]).tolist()))
    return result


def projected_bounds(v,f):
    """Bounds for 12 yaw orientations, including original triangle vertices."""
    angles=np.deg2rad(np.arange(0,90,7.5))
    xy=np.stack([v[:,0,None]*np.cos(angles)+v[:,1,None]*np.sin(angles),
                 -v[:,0,None]*np.sin(angles)+v[:,1,None]*np.cos(angles)],axis=-1).astype(np.float32)
    low=xy[f].min(1);high=xy[f].max(1)
    z=v[:,2][f];return low,high,z.min(1).astype(np.float32),z.max(1).astype(np.float32)


def solve(v,f,priors,settings,semantic=None,factor=1.,prepared=None):
    """semantic=None is the strict no-face-label comparison."""
    started=time.perf_counter();count=len(f)
    c,n,area=geometry(v,f) if prepared is None else prepared['geometry']
    pairs=adjacency(f) if prepared is None else prepared['pairs']
    bounds=projected_bounds(v,f) if prepared is None else prepared['bounds']
    ids=np.zeros(count,np.uint32);regions=[];plane_faces=np.zeros(count,bool)
    def add(fi,kind,**record):
        assert len(fi) and np.all(ids[fi]==0)
        iid=len(regions)+1;ids[fi]=iid
        p=v[np.unique(f[fi])];t=c[fi]
        row=dict(id=iid,kind=kind,faces=len(fi),area=float(area[fi].sum()),
                 bounds=[p.min(0).tolist(),p.max(0).tolist()],
                 centroid=np.average(t,axis=0,weights=np.maximum(area[fi],1e-30)).tolist(),**record)
        regions.append(row)
    # Semantics only restricts which faces can be structural in the aided condition.
    plane_cache_key='blind' if semantic is None else 'semantic'
    if prepared is not None and plane_cache_key in prepared:
        planes=prepared[plane_cache_key]
    else:
        allowed=np.ones(count,bool) if semantic is None else np.isin(semantic,[0,1,2])
        planes=plane_regions(c,n,area,allowed,settings,settings['random_seed'])
        if prepared is not None: prepared[plane_cache_key]=planes
    for p in planes:
        fi=p['faces'];plane_faces[fi]=True
        cid=-1 if semantic is None else int(np.bincount(semantic[fi].astype(int),weights=area[fi]).argmax())
        add(fi,'structure_plane',class_id=cid,normal=p['normal'])
    # Remaining labeled structure is grouped spatially, never cut into furniture-sized pieces.
    structural=np.zeros(count,bool) if semantic is None else np.isin(semantic,[0,1,2]) & ~plane_faces
    if structural.any():
        valid=structural[pairs].all(1)&(semantic[pairs[:,0]]==semantic[pairs[:,1]])
        valid&=(np.abs(np.einsum('ij,ij->i',n[pairs[:,0]],n[pairs[:,1]]))>=np.cos(np.deg2rad(settings['structure_neighbor_angle_degrees'])))
        pp=pairs[valid];graph=coo_matrix((np.ones(len(pp)),(pp[:,0],pp[:,1])),shape=(count,count))
        _,cc=connected_components(graph,directed=False)
        order=np.flatnonzero(structural);order=order[np.argsort(cc[order],kind='stable')]
        for fi in np.split(order,np.flatnonzero(np.diff(cc[order]))+1):
            add(fi,'structure_residual',class_id=int(semantic[fi[0]]))
    remaining=ids==0; nodes=np.flatnonzero(remaining)
    parents=np.arange(count);sizes=np.ones(count,np.int32)
    low,high,zlo,zhi=[a.copy() for a in bounds]
    dims=np.asarray([p['dimensions'] for p in priors],np.float32)*factor
    caps=dims*(1+settings['size_tolerance'])+settings['noise_allowance']
    class_ids=np.asarray([p['class_id'] for p in priors]);template_map={int(c):i for i,c in enumerate(class_ids)}
    all_templates=np.arange(len(priors))
    def find(x):
        root=int(x)
        while parents[root]!=root:root=int(parents[root])
        while parents[x]!=x:
            following=int(parents[x]);parents[x]=root;x=following
        return root
    def fit(lo,hi,zzlo,zzhi,choices):
        if not len(choices):return np.empty(0,int)
        spans=hi-lo;short=spans.min(1);long=spans.max(1)
        ok=((short[:,None]<=caps[choices,0])&(long[:,None]<=caps[choices,1])).any(0)
        ok&=(zzhi-zzlo<=caps[choices,2])
        return choices[ok]
    valid=remaining[pairs].all(1)
    if semantic is not None:valid&=(semantic[pairs[:,0]]==semantic[pairs[:,1]])
    pp=pairs[valid]
    normal_cost=1-np.abs(np.einsum('ij,ij->i',n[pp[:,0]],n[pp[:,1]]))
    costs=normal_cost+.05*np.linalg.norm(c[pp[:,0]]-c[pp[:,1]],axis=1)
    topology=len(pp)
    # Add short gap candidates only within the allowed semantic class in aided mode.
    if len(nodes)>1:
        distances,nearest=cKDTree(c[nodes]).query(c[nodes],k=min(5,len(nodes)),distance_upper_bound=settings['max_gap'],workers=1)
        ii,jj=np.nonzero(np.isfinite(distances)&(nearest<len(nodes))&(distances>1e-8))
        gp=np.column_stack([nodes[ii],nodes[nearest[ii,jj]]]);keep=gp[:,0]<gp[:,1]
        if semantic is not None:keep&=semantic[gp[:,0]]==semantic[gp[:,1]]
        gp=gp[keep];gapcost=distances[ii,jj][keep]/settings['max_gap']+1.1
        pp=np.concatenate([pp,gp]);costs=np.r_[costs,gapcost]
    gap_merges=0;rejected=0
    for index in np.argsort(costs,kind='stable'):
        a,b=pp[index];ra,rb=find(a),find(b)
        if ra==rb:continue
        choices=all_templates if semantic is None or int(semantic[a]) not in template_map else np.array([template_map[int(semantic[a])]])
        lo=np.minimum(low[ra],low[rb]);hi=np.maximum(high[ra],high[rb]);zl=min(zlo[ra],zlo[rb]);zh=max(zhi[ra],zhi[rb])
        if not len(fit(lo,hi,zl,zh,choices)):
            rejected+=1;continue
        if sizes[ra]<sizes[rb]:ra,rb=rb,ra
        parents[rb]=ra;sizes[ra]+=sizes[rb];low[ra],high[ra],zlo[ra],zhi[ra]=lo,hi,zl,zh
        if index>=topology:gap_merges+=1
    roots=np.array([find(i) for i in nodes]);order=np.argsort(roots,kind='stable');nodes=nodes[order];roots=roots[order]
    for fi in np.split(nodes,np.flatnonzero(np.diff(roots))+1):
        if not len(fi):continue
        root=find(fi[0]);span=high[root]-low[root];best=int(np.argmin(np.prod(span,axis=1)))
        observed=np.array([*sorted(span[best]),zhi[root]-zlo[root]])
        original_class=-1 if semantic is None else int(semantic[fi[0]])
        choices=all_templates if original_class not in template_map else np.array([template_map[original_class]])
        fits=fit(low[root],high[root],zlo[root],zhi[root],choices)
        supported=[]
        for t in fits:
            # A visible surface need only support its two largest physical extents.
            ratio=np.sort(observed)[-2:]/np.sort(dims[t])[-2:]
            if np.all(ratio>=settings['candidate_min_two_extent_ratio']):supported.append(int(t))
        sufficient=len(fi)>=settings['candidate_min_faces'] and area[fi].sum()>=.02
        if semantic is not None and original_class not in template_map:
            kind='unknown';chosen=-1
        elif sufficient and supported:
            kind='object_candidate'
            scores=[np.sum(np.abs(np.log(np.maximum(np.sort(observed)[-2:],1e-6)/np.sort(dims[t])[-2:]))) for t in supported]
            chosen=int(class_ids[supported[int(np.argmin(scores))]])
        else:
            kind='fragment';chosen=original_class
        add(fi,kind,class_id=chosen,observed_dimensions=observed.tolist(),
            compatible_class_ids=[int(class_ids[t]) for t in supported],
            size_fits_any_template=bool(len(fits)),ambiguous_size_match=len(supported)>1,
            source_class_id=original_class)
    assert np.all(ids>0) and sum(r['faces'] for r in regions)==count
    kind_names=['structure_plane','structure_residual','object_candidate','fragment','unknown']
    kind_lookup=np.array([255]+[kind_names.index(r['kind']) for r in regions],np.uint8)
    kinds=kind_lookup[ids]
    counts={kind:sum(r['kind']==kind for r in regions) for kind in kind_names}
    coverage={kind:dict(faces=int((kinds==i).sum()),area=float(area[kinds==i].sum())) for i,kind in enumerate(kind_names)}
    return dict(ids=ids,kinds=kinds,regions=regions,counts=counts,coverage=coverage,
                faces=count,seconds=time.perf_counter()-started,gap_merges=gap_merges,
                rejected_size_merges=rejected,plane_count=len(planes),all_faces_assigned=True,
                semantic_labels_used=semantic is not None,factor=factor)


def prepare(v,f):
    return dict(geometry=geometry(v,f),pairs=adjacency(f),bounds=projected_bounds(v,f))
