"""One event-location -> every panorama projector, with distance-only display.

An RGB billboard is an approximate location annotation, not an object mesh.
There are no keypoint IDs, event IDs, semantic-class rules, or manual placements.
"""
import numpy as np


def box_points(bounds):
    lo,hi=np.asarray(bounds,float)
    return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])


def photo_rays(photo,pose):
    specs={'front':([0,0,1],[0,1,0]),'right':([1,0,0],[0,1,0]),
        'back':([0,0,-1],[0,1,0]),'left':([-1,0,0],[0,1,0]),
        'up':([0,-1,0],[0,0,1]),'down':([0,1,0],[0,0,-1])}
    f,d=specs[photo['face']];rotation=np.column_stack([np.cross(d,f),d,f])
    w,h=photo['width'],photo['height'];focal=(w-1)/(2*np.tan(np.radians(50)))
    K=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1.]])
    x0,y0,x1,y1=photo['bbox'];uv=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1],[(x0+x1)/2,(y0+y1)/2,1]])
    pose=np.asarray(pose,float);rays=uv@np.linalg.inv(K).T@rotation.T@pose[:3,:3].T
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    clipped=min(x0,y0)<1 or x1>=w-1 or y1>=h-1
    return pose[:3,3],rays,.1 if clipped else 1.


def weighted_median(values,weights):
    order=np.argsort(values);s=np.cumsum(np.asarray(weights)[order])
    return np.asarray(values)[order[np.searchsorted(s,s[-1]/2)]]


def rgb_billboard(observations,fallback_depth):
    """Infer one shared RGB footprint from source bearings, never per-keyframe."""
    origins=np.array([o[0] for o in observations]);directions=np.array([o[1][-1] for o in observations]);weights=np.array([o[2] for o in observations])
    projectors=np.eye(3)-directions[:,:,None]*directions[:,None,:]
    A=np.sum(projectors*weights[:,None,None],axis=0)
    b=np.sum(np.einsum('nij,nj->ni',projectors,origins)*weights[:,None],axis=0)
    center=np.linalg.lstsq(A,b,rcond=None)[0]
    depth=np.sum((center-origins)*directions,axis=1)
    residual=np.linalg.norm(np.cross(center-origins,directions),axis=1)
    stable=np.linalg.cond(A)<1000 and (depth>.05).all() and np.median(residual)<max(.15,.15*np.median(depth))
    if not stable:
        index=int(np.argmax(weights));distance=float(fallback_depth(origins[index],directions[index]))
        if not np.isfinite(distance) or distance<=0:raise ValueError('RGB event location unavailable; cannot fabricate a distance')
        center=origins[index]+directions[index]*distance
    normal=np.average(directions,axis=0,weights=weights);normal/=np.linalg.norm(normal)
    right=np.cross([0.,0.,1.],normal)
    if np.linalg.norm(right)<.01:right=np.cross([0.,1.,0.],normal)
    right/=np.linalg.norm(right);up=np.cross(normal,right)
    extents=[];valid_weights=[]
    for (origin,rays,weight) in observations:
        denominator=rays[:4]@normal
        if np.min(np.abs(denominator))<1e-4:continue
        t=((center-origin)@normal)/denominator
        if (t<=0).any():continue
        points=origin+rays[:4]*t[:,None];delta=points-center
        u=delta@right;v=delta@up;extents.append([u.min(),v.min(),u.max(),v.max()]);valid_weights.append(weight)
    if not extents:raise ValueError('RGB event has no usable image footprint')
    # A candidate box covers the union of its observed footprints, not a median
    # footprint that can discard a partially visible side of the same change.
    extents=np.asarray(extents)
    limits=np.r_[extents[:,:2].min(0),extents[:,2:].max(0)]
    a,b,c,d=limits
    points=np.array([center+u*right+v*up for u,v in [(a,b),(c,b),(c,d),(a,d)]])
    return points,dict(method='source_RGB_bearings' if stable else 'source_RGB_background_distance_proxy',
        source_observations=len(observations),ray_residual_m=residual.tolist(),approximate_location=True,object_mesh_required=False)


def project_event(event,world_points,pose,max_distance_m=6.):
    points=np.asarray(world_points,float);pose=np.asarray(pose,float)
    center=points.mean(0);distance=float(np.linalg.norm(center-pose[:3,3]))
    if distance>max_distance_m:return None
    if len(points)==4:
        samples=np.concatenate([points[i]+np.linspace(0,1,24)[:,None]*(points[(i+1)%4]-points[i]) for i in range(4)])
    else:
        samples=np.concatenate([a+np.linspace(0,1,12)[:,None]*(b-a) for i,a in enumerate(points) for b in points[i+1:]])
    xyz=(samples-pose[:3,3])@pose[:3,:3];norm=np.linalg.norm(xyz,axis=1);xyz=xyz[norm>1e-6]
    if len(xyz)<2:return None
    angles=np.degrees(np.c_[np.arctan2(xyz[:,0],xyz[:,2]),-np.arctan2(xyz[:,1],np.hypot(xyz[:,0],xyz[:,2]))])
    mean=(center-pose[:3,3])@pose[:3,:3]
    return dict(id=event['id'],type=event['type'],label=event['label'],status='판단 보류' if event['type']=='unresolved' else '변화 추정',
        angular_samples=np.round(angles,6).tolist(),yaw=float(np.degrees(np.arctan2(mean[0],mean[2]))),
        pitch=float(-np.degrees(np.arctan2(mean[1],np.hypot(mean[0],mean[2])))),distance_m=distance,pixels=0,
        object_geometry_required=False,display_origin='shared_event_location_distance_only')
