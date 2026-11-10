"""Conservative viewer-only wall occlusion; no object mesh is required.

Only hide an annotation when >=95% of its footprint probes meet a wall strictly
before its nearest shared location. Unknown/substantial partial visibility stays.
"""
import numpy as np
from .event_overlay_labels import directions


def wall_occlusion(item, world_points, pose, cast_wall_rays, tolerance_m=.2, blocked_fraction=.95):
    pose=np.asarray(pose,float);points=np.asarray(world_points,float)
    near=float(np.linalg.norm(points-pose[:3,3],axis=1).min())
    angles=np.asarray(item['angular_samples'],float)
    angles=np.vstack([angles,[[item['yaw'],item['pitch']]]])
    # Include interior probes, not just box corners: a doorway through the middle
    # is visible even when the bounding edges fall on walls.
    local=directions(angles);center=directions(np.array([[item['yaw'],item['pitch']]]))[0]
    local=np.concatenate([local,local*.5+center*.5,local*.25+center*.75])
    local/=np.linalg.norm(local,axis=1,keepdims=True)
    world=local@pose[:3,:3].T
    rays=np.c_[np.broadcast_to(pose[:3,3],world.shape),world].astype('f4')
    distance=np.asarray(cast_wall_rays(rays),float)
    blocked=np.isfinite(distance)&(distance>.05)&(distance+tolerance_m<near)
    hidden=bool(blocked.mean()>=blocked_fraction)
    return dict(hidden=hidden,wall_blocked_fraction=float(blocked.mean()),
        probes=len(blocked),nearest_location_m=near,tolerance_m=tolerance_m,required_blocked_fraction=blocked_fraction,
        finite_wall_hits=int(np.isfinite(distance).sum()),
        wall_hit_min_m=float(np.min(distance[np.isfinite(distance)])) if np.isfinite(distance).any() else None)
