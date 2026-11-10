"""Trial-only native executor, with the reference implementation retained."""
from pathlib import Path
import ctypes
import weakref
import numpy as np
from spacewatch3d.tasks.task_5b_keyframes.visibility import BVH
from spacewatch3d.tasks.task_5b_keyframes.models import InputError

ORIGINAL=BVH.blocked_many
PACKED=weakref.WeakKeyDictionary()
LIB=ctypes.CDLL(str(Path(__file__).with_name('rays.so')))
double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
integer=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
LIB.query.argtypes=[ctypes.c_int64,double,integer,integer,double,double,ctypes.c_int64,double,double,byte,byte]
LIB.query.restype=ctypes.c_int

def packed(bvh):
    bounds=[];nodes=[];indices=[]
    def visit(part):
        low,high,ids,left,right=part;n=len(nodes);nodes.append(None);bounds.append(np.r_[low,high])
        if ids is None:
            a=visit(left);b=visit(right);nodes[n]=[a,b,0,0]
        else:
            offset=len(indices);indices.extend(ids);nodes[n]=[-1,-1,offset,len(ids)]
        return n
    visit(bvh.root)
    return (np.ascontiguousarray(bounds,dtype=float),np.ascontiguousarray(nodes,dtype=np.int64),
            np.ascontiguousarray(indices,dtype=np.int64),np.ascontiguousarray(bvh.triangles,dtype=float))

def blocked_many(self,origin,targets):
    origin=np.asarray(origin,dtype=float);targets=np.asarray(targets,dtype=float)
    if (origin.shape!=(3,) or targets.ndim!=2 or targets.shape[1]!=3
            or not np.isfinite(origin).all() or not np.isfinite(targets).all()):
        raise InputError('Occlusion queries require a finite origin 3-vector and target Nx3 array')
    # A single leaf has no traversal overhead to remove; retain the vectorized
    # reference, including its bounded batches and per-query origin cache.
    if self.root[2] is not None:return ORIGINAL(self,origin,targets)
    if self not in PACKED:PACKED[self]=packed(self)
    bounds,nodes,indices,triangles=PACKED[self]
    delta=targets-origin;distances=np.linalg.norm(delta,axis=1)
    directions=np.divide(delta,distances[:,None],out=np.zeros_like(delta),where=distances[:,None]>1e-8)
    ends=distances-1e-6;active=np.asarray(distances>1e-8,dtype=np.uint8);out=np.zeros(len(targets),dtype=np.uint8)
    result=LIB.query(len(nodes),bounds,nodes,indices,triangles,np.ascontiguousarray(origin),len(targets),
                     np.ascontiguousarray(directions),np.ascontiguousarray(ends),active,out)
    if result:raise MemoryError('native BVH stack allocation failed')
    boundary=out==2
    if boundary.any():out[boundary]=ORIGINAL(self,origin,targets[boundary])
    return out.astype(bool)

def install():BVH.blocked_many=blocked_many
def restore():BVH.blocked_many=ORIGINAL
