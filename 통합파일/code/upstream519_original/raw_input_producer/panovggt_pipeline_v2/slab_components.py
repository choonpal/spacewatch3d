"""Exact 26-neighbour occupied-voxel components using bounded-memory slabs."""
import time
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.csgraph import connected_components

def label_slabs(grid,output_path=None,slab_width=64,max_dense_cells=80_000_000,progress=None):
    start=time.perf_counter()
    n=len(grid)
    assert n>0
    lower=np.asarray(grid.min(0),dtype=np.int64)
    shape=np.asarray(grid.max(0),dtype=np.int64)-lower+1
    axis=int(np.argmax(shape))
    axes=[axis]+[a for a in range(3) if a!=axis]
    shape=shape[axes];lower=lower[axes]
    if int(shape[1])*int(shape[2]) > max_dense_cells:
        raise ValueError('Component grid cross-section exceeds configured memory bound')
    width=max(1,min(slab_width,int(max_dense_cells//(shape[1]*shape[2]))))
    slab_ids=((np.asarray(grid[:,axis],dtype=np.int64)-lower[0])//width).astype(np.int32)
    order=np.argsort(slab_ids,kind='stable')
    slab_counts=np.bincount(slab_ids,minlength=int((shape[0]+width-1)//width))
    del slab_ids
    provisional=(np.lib.format.open_memmap(output_path,mode='w+',dtype='<u4',shape=(n,))
                 if output_path else np.empty(n,dtype=np.uint32))
    offset=0;position=0;previous=None;edge_parts=[];occupied_total=0
    for slab,count in enumerate(slab_counts):
        indices=order[position:position+count];position+=count
        if count==0:
            previous=None
            continue
        slab_x=slab*width
        local_shape=(min(width,int(shape[0]-slab_x)),int(shape[1]),int(shape[2]))
        q=np.asarray(grid[indices][:,axes],dtype=np.int64)-lower
        q[:,0]-=slab_x
        flat=np.ravel_multi_index(q.T,local_shape)
        occupancy=np.zeros(local_shape,dtype=bool);occupancy.ravel()[flat]=True
        occupied_total+=int(occupancy.sum())
        local,num=ndimage.label(occupancy,structure=np.ones((3,3,3),dtype=np.uint8),output=np.uint32)
        assert offset+num<2**32
        provisional[indices]=local.ravel()[flat]+offset
        first=np.where(local[0]>0,local[0]+offset,0).astype(np.uint32)
        last=np.where(local[-1]>0,local[-1]+offset,0).astype(np.uint32)
        del local,occupancy,q,flat
        if previous is not None:
            links=[];height,depth=first.shape
            for dy in (-1,0,1):
                y0=max(0,-dy);y1=min(height,height-dy)
                for dz in (-1,0,1):
                    z0=max(0,-dz);z1=min(depth,depth-dz)
                    a=previous[y0:y1,z0:z1]
                    b=first[y0+dy:y1+dy,z0+dz:z1+dz]
                    both=(a>0)&(b>0)
                    if both.any():
                        links.append((a[both].astype(np.uint64)<<32)|b[both].astype(np.uint64))
            if links:edge_parts.append(np.unique(np.concatenate(links)))
        previous=last;offset+=int(num)
        if progress and (slab%5==0 or slab==len(slab_counts)-1):
            progress(f'SLAB {slab+1}/{len(slab_counts)} points={position}/{n} provisional_components={offset} elapsed={time.perf_counter()-start:.1f}s')
    del order
    assert position==n and provisional.min()>0
    edges=np.unique(np.concatenate(edge_parts)) if edge_parts else np.empty(0,dtype=np.uint64)
    a=(edges>>32).astype(np.int64)-1
    b=(edges&np.uint64(0xffffffff)).astype(np.int64)-1
    graph=sparse.coo_matrix((np.ones(len(edges),dtype=np.uint8),(a,b)),shape=(offset,offset)).tocsr()
    total,mapping=connected_components(graph,directed=False,return_labels=True)
    for begin in range(0,n,1_000_000):
        provisional[begin:begin+1_000_000]=mapping[provisional[begin:begin+1_000_000]-1]+1
    if hasattr(provisional,'flush'):provisional.flush()
    return provisional,dict(component_count=int(total),provisional_components=offset,boundary_links=len(edges),
                            occupied_voxels=occupied_total,grid_shape_original=(np.asarray(grid.max(0),dtype=np.int64)-np.asarray(grid.min(0),dtype=np.int64)+1).tolist(),
                            slab_axis=axis,slab_width=width,slabs=len(slab_counts),seconds=time.perf_counter()-start)

def self_check():
    # Compare with one dense-grid label operation, including diagonal crossings,
    # empty slabs, duplicate points, and negative grid coordinates.
    rng=np.random.default_rng(17)
    for width in (1,2,3,5):
        occupied=rng.random((14,9,8))<.08
        occupied[6:8]=False
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                occupied[2,4,4]=True;occupied[3,4+dy,4+dz]=True
        q=np.argwhere(occupied)
        q=np.concatenate([q,q[:11]])
        rng.shuffle(q)
        expected,num=ndimage.label(occupied,structure=np.ones((3,3,3),dtype=np.uint8))
        target=expected[tuple(q.T)]
        actual,report=label_slabs(q-np.array([8,4,5]),slab_width=width)
        pairs=np.unique(np.column_stack([actual,target]),axis=0)
        assert len(pairs)==num==report['component_count']
        assert len(np.unique(pairs[:,0]))==len(np.unique(pairs[:,1]))==num
    return dict(dense_equivalence_cases=4,slab_widths=[1,2,3,5],diagonal_boundaries=True,empty_slabs=True,duplicate_points=True,negative_indices=True)

if __name__=='__main__':print(self_check())
