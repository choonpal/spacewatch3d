"""Analytic sparse Jacobian of translation + SO(3) rotation-vector residuals."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix


def skew(v):
    result = np.zeros((*v.shape[:-1],3,3), dtype=np.float64)
    result[...,0,1] = -v[...,2]; result[...,0,2] = v[...,1]
    result[...,1,0] = v[...,2]; result[...,1,2] = -v[...,0]
    result[...,2,0] = -v[...,1]; result[...,2,1] = v[...,0]
    return result


def jacobian(x, initial, a, b, xa, xb, weights):
    values = x.reshape(-1,6)
    rotations = initial[:,:3,:3].copy()
    rotations[1:] = Rotation.from_rotvec(values[:,3:]).as_matrix()
    vectors = np.zeros((len(initial),3)); vectors[1:] = values[:,3:]
    theta2 = (vectors*vectors).sum(1); theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    aa = np.empty_like(theta); bb = np.empty_like(theta)
    aa[small] = .5-theta2[small]/24+theta2[small]**2/720
    bb[small] = 1/6-theta2[small]/120+theta2[small]**2/5040
    aa[~small] = (1-np.cos(theta[~small]))/theta2[~small]
    bb[~small] = (theta[~small]-np.sin(theta[~small]))/(theta2[~small]*theta[~small])
    cross = skew(vectors)
    right = np.eye(3)-aa[:,None,None]*cross+bb[:,None,None]*(cross@cross)
    rows=[]; cols=[]; data=[]
    for ids, points, sign in [(a,xa,1),(b,xb,-1)]:
        selected = np.flatnonzero(ids != 0); frames = ids[selected]
        block = np.empty((len(selected),3,6))
        block[:,:,:3] = np.eye(3)
        block[:,:,3:] = -rotations[frames]@skew(points[selected])@right[frames]
        block *= (sign*weights[selected])[:,None,None]
        rows.append(np.broadcast_to((3*selected[:,None]+np.arange(3))[...,None],block.shape).ravel())
        cols.append(np.broadcast_to((6*(frames-1)[:,None]+np.arange(6))[:,None,:],block.shape).ravel())
        data.append(block.ravel())
    return coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
                      shape=(3*len(a),len(x))).tocsr()
