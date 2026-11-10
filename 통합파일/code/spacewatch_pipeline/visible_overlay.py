"""Presentation visibility, not change detection or object existence validation."""
import copy
import numpy as np
from scipy.ndimage import label,minimum_filter
from .event_overlay_labels import directions


def unoccluded(target_depth, first_surface_depth, margin_m=.05):
    target=np.asarray(target_depth);front=np.asarray(first_surface_depth)
    return np.isfinite(target)&((~np.isfinite(front))|(target<=front+margin_m))


def photo_evidence_item(item):
    """Keep an already verified current-photo footprint, without refining twice."""
    if not item.get('label_position_refined'):return None
    out=copy.deepcopy(item)
    out['display_origin']='current_photo_2D_evidence_preserved'
    return out


def visible_items(item,mask,mode):
    """Separate visible components so a rectangle cannot bridge hidden regions."""
    mask=np.asarray(mask,bool);h,w=mask.shape;components,count=label(mask);result=[]
    for n in range(1,count+1):
        own=components==n;y,x=np.nonzero(own)
        if len(x)<8:continue # same 2D evidence legibility floor, not detection
        boundary=own&~minimum_filter(own,size=3,mode=('nearest','wrap'))
        yy,xx=np.nonzero(boundary)
        if not len(xx):yy,xx=y,x
        ii=np.linspace(0,len(xx)-1,min(512,len(xx)),dtype=int)
        samples=np.c_[((xx[ii]+.5)/w-.5)*360,-((yy[ii]+.5)/h-.5)*180]
        mean=directions(np.c_[((x+.5)/w-.5)*360,-((y+.5)/h-.5)*180]).mean(0)
        out=copy.deepcopy(item);out.update(angular_samples=samples.round(6).tolist(),
            yaw=float(np.degrees(np.arctan2(mean[0],mean[2]))),
            pitch=float(-np.degrees(np.arctan2(mean[1],np.hypot(mean[0],mean[2])))),
            pixels=len(x),display_origin=mode,visible_component=n)
        result.append(out)
    return result
