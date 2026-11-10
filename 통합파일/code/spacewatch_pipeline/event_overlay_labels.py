"""Use existing per-photo 2D evidence to position RGB-only candidate boxes.

No model call or event decision: missing labels leave the common box unchanged.
"""
import numpy as np
from scipy.ndimage import label,minimum_filter

def directions(angles):
    a=np.radians(angles);y=a[:,0];p=-a[:,1]
    return np.c_[np.sin(y)*np.cos(p),np.sin(p),np.cos(y)*np.cos(p)]

def refine_from_labels(mask,item):
    components,count=label(mask)
    expected=directions(np.array([[item['yaw'],item['pitch']]]))[0]
    prior=directions(np.array(item['angular_samples']));prior_radius=float(np.quantile(np.arccos(np.clip(prior@expected,-1,1)),.9))
    h,w=mask.shape;choices=[]
    for component in range(1,count+1):
        own=components==component;y,x=np.nonzero(own)
        if len(x)<8:continue
        angle=np.c_[((x+.5)/w-.5)*360,-((y+.5)/h-.5)*180];rays=directions(angle)
        center=rays.mean(0);center/=np.linalg.norm(center)
        gap=float(np.arccos(np.clip(center@expected,-1,1)))
        radius=float(np.quantile(np.arccos(np.clip(rays@center,-1,1)),.9))
        if gap>max(np.radians(20),prior_radius+radius):continue
        boundary=own&~minimum_filter(own,size=3,mode=('nearest','wrap'));yy,xx=np.nonzero(boundary)
        ii=np.linspace(0,len(xx)-1,min(256,len(xx)),dtype=int);samples=np.c_[((xx[ii]+.5)/w-.5)*360,-((yy[ii]+.5)/h-.5)*180]
        choices.append((gap,-len(x),dict(angular_samples=samples.tolist(),
            yaw=float(np.degrees(np.arctan2(center[0],center[2]))),
            pitch=float(-np.degrees(np.arctan2(center[1],np.hypot(center[0],center[2])))),pixels=len(x),
            display_origin='shared_event_location_existing_2D_evidence',label_position_refined=True)))
    return min(choices,key=lambda x:x[:2])[2] if choices else None
