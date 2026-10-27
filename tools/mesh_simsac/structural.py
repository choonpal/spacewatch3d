"""Robust structural fitting reused from the mesh registration experiments."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .core import transform

def rigid_icp(source, target, ns, nt, initial, height, point_weight):
    """Trimmed, bidirectional robust point-plane ICP plus weak point anchors."""
    matrix = initial.copy()
    target_tree = cKDTree(target)
    trace = []
    for gate in [.65, .30, .15, .08]:
        for iteration in range(20):
            p = transform(source, matrix)
            sn = ns @ matrix[:3, :3].T
            groups = []
            for a, b, an, bn, tree, reverse in [(p, target, sn, nt, target_tree, False),
                (target, p, nt, sn, cKDTree(p), True)]:
                d, ids = tree.query(a, workers=4)
                valid = (d < gate * height) & (abs(np.sum(an * bn[ids], axis=1)) > .75)
                if valid.sum() < 50:
                    continue
                valid &= d <= np.quantile(d[valid], .85)
                picks = np.flatnonzero(valid)[::max(1, int(valid.sum()) // 5000)]
                if reverse:
                    groups.append((b[ids[picks]], a[picks], an[picks]))
                else:
                    groups.append((a[picks], b[ids[picks]], bn[ids[picks]]))
            if not groups:
                break
            pp, qq, nn = [np.concatenate([g[k] for g in groups]) for k in range(3)]
            center = np.mean(pp, axis=0)
            plane = np.sum((pp - qq) * nn, axis=1)
            robust_scale = max(.025 * height, 1.4826 * np.median(abs(plane - np.median(plane))))
            weight = np.minimum(1., 1.5 * robust_scale / np.maximum(abs(plane), 1e-12))
            category = np.where(abs(nn[:, 1]) > .8, 0, np.where(abs(nn[:, 1]) < .3, 1, 2))
            counts = np.maximum(np.bincount(category, minlength=3), 1)
            weight *= len(pp) / (3 * counts[category])
            jac = np.c_[np.cross(pp - center, nn), nn]
            design = jac * np.sqrt(weight[:, None])
            rhs = -plane * np.sqrt(weight)
            # Weak isotropic anchors constrain drift along a long straight wall.
            if point_weight:
                x = pp - center
                skew = np.zeros((len(pp), 3, 3))
                skew[:, 0, 1] = x[:, 2]; skew[:, 0, 2] = -x[:, 1]
                skew[:, 1, 0] = -x[:, 2]; skew[:, 1, 2] = x[:, 0]
                skew[:, 2, 0] = x[:, 1]; skew[:, 2, 1] = -x[:, 0]
                anchor = np.concatenate([skew, np.broadcast_to(np.eye(3), skew.shape)], axis=2)
                aw = np.sqrt(point_weight * weight)[:, None]
                design = np.r_[design, (anchor * aw[..., None]).reshape(-1, 6)]
                rhs = np.r_[rhs, ((qq - pp) * aw).ravel()]
            update = np.linalg.lstsq(design, rhs, rcond=None)[0]
            factor = min(1., .04 / max(np.linalg.norm(update[:3]), 1e-12),
                         .12 * height / max(np.linalg.norm(update[3:]), 1e-12))
            update *= factor
            delta = np.eye(4); delta[:3, :3] = Rotation.from_rotvec(update[:3]).as_matrix()
            delta[:3, 3] = center + update[3:] - delta[:3, :3] @ center
            matrix = delta @ matrix
            trace.append({'gate_height': gate, 'iteration': iteration, 'pairs': len(pp),
                          'plane_median_u': float(np.median(abs(plane))),
                          'update_norm': float(np.linalg.norm(update))})
            if np.linalg.norm(update) < 2e-6:
                break
    return matrix, trace

STRUCTURE={0,3}
def residuals(a,b,na,nb,ca,cb):
    """Both directions, same semantic class, no distance-based censoring."""
    rows=[]
    for direction,(p,q,n,nq,c,cq) in enumerate([(a,b,na,nb,ca,cb),(b,a,nb,na,cb,ca)]):
        for k in STRUCTURE:
            src=c==k;dst=cq==k
            if src.sum()==0 or dst.sum()==0:continue
            d,ix=cKDTree(q[dst]).query(p[src]);compat=np.abs((n[src]*nq[dst][ix]).sum(1))>.85
            delta=p[src]-q[dst][ix]
            if direction==1:delta=-delta
            rows.append((p[src],d,nq[dst][ix],compat,delta))
    if not rows:return None
    return tuple(np.concatenate([r[i] for r in rows]) for i in range(5))

def error_summary(rows,h,center=None):
    if rows is None:return {'supported':False,'samples':0,'p90_u':None,'normal_rank':0,'normal_compatible_fraction':0}
    p,d,n,compat,delta=rows;scope='zone'
    if center is not None:
        near=np.linalg.norm(p-center,axis=1)<1.2*h
        if near.sum()>=150:p,d,n,compat,delta=p[near],d[near],n[near],compat[near],delta[near];scope='near_object'
    # Include every nearest structural distance in the empirical quantile.
    eig=np.linalg.eigvalsh(n.T@n/max(len(n),1));rank=int((eig>.025).sum())
    supported=len(d)>=150 and compat.mean()>.6 and rank==3
    # Nearest distance to a plane measures one component of a 3-D map offset.
    # Recover the jointly observed structural translation so an xyz shift is
    # not falsely flagged because each individual plane is only .08 away.
    r=np.einsum('ij,ij->i',delta,n);weight=np.ones(len(r));bias=np.zeros(3)
    for _ in range(5):
        bias=np.linalg.lstsq(n*np.sqrt(weight[:,None]),r*np.sqrt(weight),rcond=.025)[0]
        residual=r-n@bias;sigma=max(.003*h,1.4826*np.median(np.abs(residual-np.median(residual))))
        weight=np.minimum(1,1.5*sigma/np.maximum(abs(residual),1e-12))
    scatter=float(np.quantile(np.abs(r-n@bias),.9))
    bound=float(max(.05*h,np.quantile(d,.9),np.linalg.norm(bias)+np.sqrt(3)*scatter))
    return {'supported':bool(supported),'samples':len(d),'p50_u':float(np.median(d)),
            'p90_u':float(np.quantile(d,.9)),'normal_rank':rank,'normal_eigenvalues':eig.tolist(),
            'normal_compatible_fraction':float(compat.mean()),'scope':scope,
            'translation_bias_u':bias.tolist(),'translation_bias_norm_u':float(np.linalg.norm(bias)),
            'plane_scatter_p90_u':scatter,'tolerance_u':bound}
