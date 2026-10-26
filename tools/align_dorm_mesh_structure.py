"""Whole-footprint geometric Sim(3) proposal and guarded structural refinement.

No object/semantic/change labels and no hand-picked cross-epoch photo matches.
The result describes geometric fit, not physical identity of repeated interiors.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial import cKDTree
from dorm_mesh_geometry import transform,yaw_matrix,refine_similarity


def structural_samples(points,normals,height):
    # Conservative geometric bands: avoid horizontal tables and the ceiling.
    floor=(np.abs(normals[:,1])>.88)&(points[:,1]>-.35*height)&(points[:,1]<.24*height)
    walls=(np.abs(normals[:,1])<.25)&(points[:,1]>.18*height)&(points[:,1]<.82*height)
    ids=np.flatnonzero(floor|walls)
    keep=np.unique(np.floor(points[ids]/(.045*height)).astype(int),axis=0,return_index=True)[1]
    ids=ids[keep]
    return points[ids],normals[ids],np.where(floor[ids],0,1)


def smooth_structural_samples(points,height):
    """Local geometric PCA, then tall vertical columns, without class fields."""
    tree=cKDTree(points);normals=np.empty_like(points);planarity=np.empty(len(points))
    for start in range(0,len(points),10000):
        block=points[start:start+10000];distance,ids=tree.query(block,k=24,workers=4)
        neighbors=points[ids];centered=neighbors-neighbors.mean(1)[:,None,:]
        covariance=np.einsum('nki,nkj->nij',centered,centered)/24
        eigen,vectors=np.linalg.eigh(covariance)
        normals[start:start+len(block)]=vectors[:,:,0]
        planarity[start:start+len(block)]=(eigen[:,1]-eigen[:,0])/np.maximum(eigen[:,2],1e-10)
    floor=(np.abs(normals[:,1])>.93)&(points[:,1]>-.3*height)&(points[:,1]<.20*height)&(planarity>.25)
    walls=(np.abs(normals[:,1])<.18)&(points[:,1]>.15*height)&(points[:,1]<.85*height)&(planarity>.35)
    ids=np.flatnonzero(walls);cells=np.floor(points[ids][:,[0,2]]/(.12*height)).astype(int)
    unique,inverse=np.unique(cells,axis=0,return_inverse=True);lower=np.full(len(unique),np.inf);upper=np.full(len(unique),-np.inf)
    np.minimum.at(lower,inverse,points[ids,1]);np.maximum.at(upper,inverse,points[ids,1])
    # A low chair or table cannot supply a full-height vertical reference column.
    walls[ids]&=(upper[inverse]-lower[inverse])>.43*height
    ids=np.flatnonzero(floor|walls)
    keep=np.unique(np.floor(points[ids]/(.045*height)).astype(int),axis=0,return_index=True)[1]
    ids=ids[keep]
    return points[ids],normals[ids],np.where(floor[ids],0,1)


def raster(points,normals,kinds,spacing):
    xz=points[:,[0,2]];lower=np.floor(xz.min(0)/spacing).astype(int)-2
    cells=np.floor(xz/spacing).astype(int)-lower;size=cells.max(0)+3
    images=np.zeros((3,*size),np.float32)
    channels=np.where(kinds==0,0,np.where(np.abs(normals[:,0])>np.abs(normals[:,2]),1,2))
    for c in range(3):
        unique=np.unique(cells[channels==c],axis=0)
        images[c,unique[:,0],unique[:,1]]=1
    return images,lower*spacing


def metrics(source,target,source_normals,target_normals,matrix,height):
    moved=transform(source,matrix);rotation=matrix[:3,:3]/np.cbrt(np.linalg.det(matrix[:3,:3]))
    dn,jn=cKDTree(target).query(moved);dr,jr=cKDTree(moved).query(target)
    compatible=np.abs(np.sum(source_normals@rotation.T*target_normals[jn],axis=1))>.85
    reverse_compatible=np.abs(np.sum(source_normals[jr]@rotation.T*target_normals,axis=1))>.85
    tolerance=.05*height;good=(dn<tolerance)&compatible;reverse=(dr<tolerance)&reverse_compatible
    score=float((np.mean(np.minimum(np.where(compatible,dn,height)/(.14*height),1)**2)+
        np.mean(np.minimum(np.where(reverse_compatible,dr,height)/(.14*height),1)**2))/2)
    return {'overlap':float((good.mean()+reverse.mean())/2),'cost':score,
        'distance_median':float(np.median(np.r_[dn,dr])),'tolerance':tolerance}


def run(root,output):
    if output.exists():raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    metadata=json.loads((root/'run-v6/alignment.json').read_text());height=metadata['quality']['reference_height_units']
    clouds=np.load(root/'inspection/alignment-samples.npz')
    before,nb,kb=structural_samples(clouds['before'],clouds['normal_before'],height)
    after,na,ka=structural_samples(clouds['after'],clouds['normal_after'],height)
    spacing=.12*height;reference,origin_a=raster(before,nb,kb,spacing)
    # Whole-layout scoring includes cross walls, short end sections and the floor;
    # no median-centre translation proposal from arbitrarily selected planes.
    candidates=[]
    for angle in np.r_[np.arange(-.24,.241,.04),np.pi+np.arange(-.24,.241,.04)]:
        rotation=yaw_matrix(angle)
        for scale in np.arange(.82,1.121,.03):
            moved=after@(scale*rotation).T
            moving,origin_b=raster(moved,na@rotation.T,ka,spacing)
            correlations=[fftconvolve(gaussian_filter(reference[c],.6),moving[c,::-1,::-1],mode='full') for c in range(3)]
            correlation=correlations[0]*.5+correlations[1]+correlations[2]*1.5
            maxima=(maximum_filter(correlation,size=5)==correlation)
            flat=np.flatnonzero(maxima);flat=flat[np.argsort(correlation.ravel()[flat])[-3:]]
            for index in flat:
                cell=np.array(np.unravel_index(index,correlation.shape))-(np.array(moving.shape[1:])-1)
                shift=origin_a-origin_b+cell*spacing
                matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=[shift[0],0,shift[1]]
                score=metrics(after[::3],before[::3],na[::3],nb[::3],matrix,height)
                candidates.append((score['cost'],matrix,score))
    candidates.sort(key=lambda row:row[0]);distinct=[]
    for row in candidates:
        if all(np.linalg.norm(row[1][:3,3]-other[1][:3,3])>.6*height or abs(np.cbrt(np.linalg.det(row[1][:3,:3]))-np.cbrt(np.linalg.det(other[1][:3,:3])))>.08 for other in distinct):distinct.append(row)
        if len(distinct)>=8:break
    results=[]
    for _,matrix,initial in distinct:
        trace=[];refined=refine_similarity(after,before,na,nb,matrix,height,trace=trace)
        final=metrics(after,before,na,nb,refined,height)
        print('WHOLE STRUCTURE',initial,'->',final,flush=True)
        results.append({'matrix':refined.tolist(),'initial':initial,'final':final,'trace':trace})
    results.sort(key=lambda r:r['final']['cost'])
    record={'method':'whole footprint FFT proposals; geometric floor/vertical surface bands; guarded robust Sim3 point-to-plane refinement',
        'labels_used':False,'physical_scale_known':False,'reference_height':height,'source_counts':[len(before),len(after)],'hypotheses':results,
        'after_to_before':results[0]['matrix'],'physical_identity_verified':False}
    (output/'alignment.json').write_text(json.dumps(record,indent=2))
    np.savez_compressed(output/'structure.npz',before=before,after=after,nb=nb,na=na,kb=kb,ka=ka)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.root,a.output)
