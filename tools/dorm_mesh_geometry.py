"""Geometry-only relative Sim(3), surface changes and panorama projection.

Coordinates remain in the reference reconstruction's units. No instance or
semantic fields enter these functions. Structural planes are hypotheses.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def sample_mesh(mesh, count=350000, spacing=.06, seed=20261019):
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    tri = v[f]; area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)/2
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(f), count, p=area/area.sum())
    uv = rng.random((count,2)); u = np.sqrt(uv[:,0])
    bary = np.c_[1-u,u*(1-uv[:,1]),u*uv[:,1]]
    p = np.einsum('ni,nij->nj',bary,tri[ids])
    keep = np.unique(np.floor(p/spacing).astype(np.int64),axis=0,return_index=True)[1]
    mesh.compute_triangle_normals()
    n = np.asarray(mesh.triangle_normals)[ids[keep]]
    return p[keep], n


def extract_structure(points,normals,seed=20261019):
    import open3d as o3d
    o3d.utility.random.seed(seed)
    # Height is only a reconstruction-relative scale, never metres.
    height = float(np.quantile(points[:,1],.97)-np.quantile(points[:,1],.03))
    height = max(height,1.)
    remaining = np.arange(len(points)); candidates=[]
    for _ in range(24):
        if len(remaining)<350:break
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining]))
        equation,local=cloud.segment_plane(distance_threshold=.018*height,ransac_n=3,num_iterations=650)
        ids=remaining[local]; remaining=np.delete(remaining,local)
        if len(ids)<180:continue
        n=np.array(equation[:3]);n/=np.linalg.norm(n)
        ids=ids[np.abs(normals[ids]@n)>.85]
        if len(ids)<150:continue
        p=points[ids];center=np.median(p,axis=0)
        _,_,vt=np.linalg.svd(p-center,full_matrices=False)
        uv=(p-center)@vt[:2].T; span=np.ptp(np.quantile(uv,[.04,.96],axis=0),axis=0)
        vertical=abs(n[1])<.20
        horizontal=abs(n[1])>.96
        if vertical and np.ptp(np.quantile(p[:,1],[.04,.96]))>.42*height and span.min()>.22*height:kind='wall'
        elif horizontal and span.min()>.45*height:kind='horizontal'
        else:continue
        candidates.append({'normal':n,'center':center,'ids':ids,'kind':kind,'span':span,'count':len(ids)})
    horizontal=[p for p in candidates if p['kind']=='horizontal']
    if not horizontal:raise ValueError('No broad geometric floor reference')
    # Reject a high table as a floor: take the lowest broad horizontal surface.
    floor=min(horizontal,key=lambda p:p['center'][1]);floor['kind']='floor'
    ceiling=max(horizontal,key=lambda p:p['center'][1])
    if ceiling is not floor and ceiling['center'][1]-floor['center'][1]>.55*height:
        ceiling['kind']='ceiling';height=float(ceiling['center'][1]-floor['center'][1])
    chosen=[p for p in candidates if p['kind'] in ('floor','ceiling','wall')]
    if len([p for p in chosen if p['kind']=='wall'])<2:raise ValueError('Fewer than two geometric wall references')
    return chosen,height


def yaw_matrix(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def plane_samples(points,planes):
    ids=np.concatenate([p['ids'] for p in planes]); n=np.concatenate([np.tile(p['normal'],(len(p['ids']),1)) for p in planes])
    return points[ids],n


def refine_similarity(source,target,source_normals,target_normals,initial,height,iterations=24,trace=None):
    matrix=initial.copy();tree=cKDTree(target)
    # A finite, fixed evaluation set prevents a step from looking better merely
    # because it discards most correspondences. Corridor planes have a weak
    # tangential direction: a least-squares step there must not be accepted blindly.
    eval_ids=np.arange(0,len(source),max(1,len(source)//6000))
    reverse_ids=np.arange(0,len(target),max(1,len(target)//6000))
    def objective(candidate,radius):
        p=transform(source[eval_ids],candidate)
        d,near=tree.query(p)
        rotation=candidate[:3,:3]/np.cbrt(np.linalg.det(candidate[:3,:3]))
        compatible=np.abs(np.sum((source_normals[eval_ids]@rotation.T)*target_normals[near],axis=1))>.90
        d=np.where(compatible,d,radius)
        reverse=cKDTree(transform(source,candidate)).query(target[reverse_ids])[0]
        return float((np.mean(np.minimum(d/radius,1.)**2)+np.mean(np.minimum(reverse/radius,1.)**2))/2)
    for radius in [.12*height,.065*height,.035*height]:
        for _ in range(iterations):
            p=transform(source,matrix);distance,nearest=tree.query(p,workers=4)
            scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rotation=matrix[:3,:3]/scale
            valid=(distance<radius)&(np.abs(np.einsum('ij,ij->i',source_normals@rotation.T,target_normals[nearest]))>.90)
            if valid.sum()<100:break
            a,b,n=p[valid],target[nearest[valid]],target_normals[nearest[valid]]
            residual=np.einsum('ij,ij->i',n,a-b)
            valid2=np.abs(residual)<=np.quantile(np.abs(residual),.8)
            a,b,n,residual=a[valid2],b[valid2],n[valid2],residual[valid2]
            pivot=np.median(a,axis=0);centered=a-pivot
            design=np.c_[n,np.cross(centered,n)/height,np.einsum('ij,ij->i',n,centered)/height]
            cells=np.floor(a/(.1*height)).astype(int);_,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
            sigma=max(1.4826*np.median(np.abs(residual-np.median(residual))),height*.002)
            weight=np.minimum(1.,1.5*sigma/np.maximum(np.abs(residual),1e-12))/counts[inv]
            step=np.linalg.lstsq(design*np.sqrt(weight[:,None]),-residual*np.sqrt(weight),rcond=.025)[0]
            step[3:]/=height
            step[:3]=np.clip(step[:3],-.04*height,.04*height)
            step[3:6]=np.clip(step[3:6],-.02,.02);step[6]=np.clip(step[6],-.015,.015)
            previous=objective(matrix,radius);accepted=False
            for fraction in [1.,.5,.25,.125,.0625]:
                s=step*fraction
                update=np.eye(4);update[:3,:3]=np.exp(s[6])*Rotation.from_rotvec(s[3:6]).as_matrix()
                update[:3,3]=pivot+s[:3]-update[:3,:3]@pivot
                candidate=update@matrix;cost=objective(candidate,radius)
                if cost<previous-1e-9:
                    matrix=candidate;accepted=True
                    if trace is not None:trace.append({'radius':radius,'before':previous,'after':cost,'fraction':fraction})
                    break
            if not accepted:break
            if np.linalg.norm(step)<1e-6:break
    return matrix


def align_structure(before,after,normals_before,normals_after,planes_before,planes_after,height_before,height_after):
    a,na=plane_samples(before,planes_before);b,nb=plane_samples(after,planes_after)
    # Spatially uniform subsamples prevent tessellation density dominating.
    def thin(p,n):
        ids=np.unique(np.floor(p/(height_before*.035)).astype(int),axis=0,return_index=True)[1]
        return p[ids],n[ids]
    a,na=thin(a,na);b,nb=thin(b,nb)
    tree=cKDTree(a);reverse=cKDTree(b)
    floor_a=next(p for p in planes_before if p['kind']=='floor')
    floor_b=next(p for p in planes_after if p['kind']=='floor')
    walls_a=sorted([p for p in planes_before if p['kind']=='wall'],key=lambda p:-p['count'])[:9]
    walls_b=sorted([p for p in planes_after if p['kind']=='wall'],key=lambda p:-p['count'])[:9]
    base=height_before/height_after;candidates=[]
    angles={round(v,4) for v in np.arange(-np.pi,np.pi,np.pi/12)}
    for first,second in itertools.product(walls_a,walls_b):
        x=np.arctan2(first['normal'][2],first['normal'][0]);y=np.arctan2(second['normal'][2],second['normal'][0])
        for d in [0,np.pi]:angles.add(round(y-x+d,4))
    coarse=b[::max(1,len(b)//900)];coarse_normals=nb[::max(1,len(b)//900)]
    for angle in angles:
        rotation=yaw_matrix(angle)
        pairs=[(x,y) for x,y in itertools.product(walls_a,walls_b) if abs(x['normal']@(rotation@y['normal']))>.97]
        for scale in base*np.array([.70,.85,1.,1.18,1.4]):
            shifts=[np.median(a,axis=0)-scale*rotation@np.median(b,axis=0)]
            for (x,y),(z,w) in itertools.combinations(pairs,2):
                system=np.array([x['normal'][[0,2]],z['normal'][[0,2]]])
                if abs(np.linalg.det(system))<.65 or y is w:continue
                ty=float(floor_a['center'][1]-scale*floor_b['center'][1])
                rhs=np.array([x['normal']@(x['center']-scale*rotation@y['center'])-x['normal'][1]*ty,
                    z['normal']@(z['center']-scale*rotation@w['center'])-z['normal'][1]*ty])
                tx,tz=np.linalg.solve(system,rhs);shifts.append(np.array([tx,ty,tz]))
            for shift in shifts:
                moved=coarse@(scale*rotation).T+shift
                distance,nearest=tree.query(moved)
                compatible=np.abs(np.einsum('ij,ij->i',coarse_normals@rotation.T,na[nearest]))>.9
                hit=(distance<height_before*.075)&compatible
                if hit.sum()<80:continue
                score=float(hit.mean()-.15*np.mean(np.minimum(distance/height_before,.5)))
                matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=shift
                candidates.append((score,matrix))
    if not candidates:raise ValueError('No structural alignment hypothesis')
    candidates.sort(key=lambda pair:-pair[0]);best=[]
    for _,initial in candidates[:16]:
        matrix=refine_similarity(b,a,nb,na,initial,height_before)
        moved=transform(b,matrix);distance,nearest=tree.query(moved)
        rotation=matrix[:3,:3]/np.cbrt(np.linalg.det(matrix[:3,:3]))
        normal=np.abs(np.einsum('ij,ij->i',nb@rotation.T,na[nearest]))>.9
        good=(distance<height_before*.05)&normal
        other= cKDTree(moved).query(a)[0]
        overlap=(good.mean()+(other<height_before*.05).mean())/2
        residual=np.abs(np.einsum('ij,ij->i',na[nearest],moved-a[nearest]))
        error=float(np.median(residual[good])) if good.any() else height_before
        score=overlap-.3*error/height_before
        best.append((score,matrix,float(overlap),error))
    _,matrix,overlap,error=max(best,key=lambda row:row[0])
    return matrix,{'structural_overlap':overlap,'normal_residual_median_units':error,'hypotheses':len(candidates),
        'reference_height_units':height_before,'relative_scale_after_to_before':float(np.cbrt(np.linalg.det(matrix[:3,:3]))),
        'reference_points':[len(a),len(b)]}


def erp_project(points,pose,width,height):
    p=(np.asarray(points)-pose[:3,3])@pose[:3,:3]
    depth=np.linalg.norm(p,axis=1)
    # PanoVGGT convention: longitude atan2(x,z), positive Y down.
    u=(np.arctan2(p[:,0],p[:,2])/(2*np.pi)+.5)*width-.5
    v=(np.arcsin(np.clip(p[:,1]/np.maximum(depth,1e-12),-1,1))/np.pi+.5)*height-.5
    return np.c_[u,v],depth
