"""Stream all valid pixels into a globally voxel-averaged cloud on server storage."""
from pathlib import Path
import os, json, time, shutil, tempfile
os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']='2'
import numpy as np
from plyfile import PlyData
from infer import save,digest
from runtime import R,C
M=json.loads((R/'input/manifest.json').read_text())
DTYPE=np.dtype([(n,'<f4') for n in ['x','y','z']]+[(n,'u1') for n in ['red','green','blue']])
def header(n):
    return ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join('property float '+n+'\n' for n in ['x','y','z'])+''.join('property uchar '+n+'\n' for n in ['red','green','blue'])+'end_header\n').encode()

def main():
    start=time.perf_counter();out=R/'unfiltered_result';out.mkdir(exist_ok=True)
    report=json.loads((R/'registration_report.json').read_text());assert report['status'] in ['converged','converged_components']
    with np.load(R/'optimized.npz') as z:poses=z['poses']
    with np.load(R/'prepared.npz') as z:scales=z['scales'];owners=z['owners']
    voxel=C['voxel_normalized'];raw=0;merged=0
    target=out/C['output']['ply_name'];bucket_count=C['output']['voxel_buckets'];step=C['output']['pixel_step']
    assert not target.exists(),'Preserve existing export; inspect before retrying.'
    with tempfile.TemporaryDirectory(prefix='voxel_work_',dir=out) as work:
        work=Path(work);files=[(work/f'{k:02d}.bin').open('wb') for k in range(bucket_count)]
        for wi,indices in enumerate(C['window_indices']):
            with np.load(R/f'windows/window_{wi:03d}.npz') as z:
                pp=z['points'];rgb=z['colors'];vv=z['valid'];rawposes=z['poses']
            for f in np.flatnonzero(owners[:,0]==wi):
                li=owners[f,1];mask=vv[li].copy()
                sample_grid=np.zeros_like(mask);sample_grid[::step,::step]=True;mask &= sample_grid
                p=pp[li][mask].astype(np.float64);color=rgb[li][mask]
                rp=rawposes[li];rot=poses[f,:3,:3]@rp[:3,:3].T
                t=poses[f,:3,3]-scales[wi]*rot@rp[:3,3]
                xyz=(scales[wi]*(p@rot.T)+t).astype(np.float32)
                assert np.isfinite(xyz).all();raw+=len(xyz)
                key=np.floor(xyz/voxel).astype(np.int64)
                bucket=((key[:,0]*73856093)^(key[:,1]*19349663)^(key[:,2]*83492791))%bucket_count
                data=np.empty(len(xyz),DTYPE)
                for k,n in enumerate(['x','y','z']):data[n]=xyz[:,k]
                for k,n in enumerate(['red','green','blue']):data[n]=color[:,k]
                for k,handle in enumerate(files):data[bucket==k].tofile(handle)
            if wi%10==0:print('COLLECT',wi,len(C['window_indices']),raw,flush=True)
        for f in files:f.close()
        body=work/'vertices.bin'
        with body.open('wb') as handle:
            for k in range(bucket_count):
                data=np.fromfile(work/f'{k:02d}.bin',dtype=DTYPE)
                if not len(data):continue
                xyz=np.column_stack([data[n] for n in ['x','y','z']]);rgb=np.column_stack([data[n] for n in ['red','green','blue']])
                keys=np.floor(xyz/voxel).astype(np.int64);_,inv=np.unique(keys,axis=0,return_inverse=True);counts=np.bincount(inv)
                vertices=np.empty(len(counts),DTYPE)
                for j,n in enumerate(['x','y','z']):vertices[n]=np.bincount(inv,weights=xyz[:,j])/counts
                for j,n in enumerate(['red','green','blue']):vertices[n]=np.rint(np.bincount(inv,weights=rgb[:,j])/counts).clip(0,255)
                vertices.tofile(handle);merged+=len(vertices)
                if k%16==0:print('VOXEL',k,bucket_count,merged,flush=True)
        tmp=target.with_suffix('.partial.ply')
        with tmp.open('wb') as handle,body.open('rb') as src:handle.write(header(merged));shutil.copyfileobj(src,handle)
        tmp.replace(target)
    v=PlyData.read(target,mmap='r')['vertex'].data
    assert len(v)==merged
    for k in range(0,merged,1000000):assert all(np.isfinite(v[n][k:k+1000000]).all() for n in ['x','y','z'])
    rot=poses[:,:3,:3]
    assert np.max(abs(np.linalg.det(rot)-1))<1e-6 and np.max(abs(rot@rot.transpose(0,2,1)-np.eye(3)))<1e-6
    ids=np.sort(np.random.default_rng(C['output']['preview_seed']).choice(merged,min(C['output']['preview_points'],merged),replace=False));sample=v[ids]
    with (out/'preview.ply').open('wb') as f:f.write(header(len(sample)));sample.tofile(f)
    xyz=np.column_stack([sample[n] for n in ['x','y','z']]);rgb=np.column_stack([sample[n] for n in ['red','green','blue']])
    np.savez_compressed(out/'viewer_sample.npz',xyz=xyz,rgb=rgb)
    rows=[dict(frame=i,timestamp_s=M['frames'][i]['timestamp_s'],camera_to_world=p.tolist(),source_window=int(owners[i,0]),depth_scale=float(scales[owners[i,0]])) for i,p in enumerate(poses)]
    save(out/'camera_poses.json',rows)
    for path,h in json.loads((R/'source_hashes.json').read_text()).items():assert digest(Path(path))==h
    validation=dict(point_cloud=str(target),points=merged,raw_valid_points=raw,sha256=digest(target),bytes=target.stat().st_size,
                    finite_coordinates=True,valid_camera_rotations=True,source_caches_unchanged=True,
                    voxel_size_normalized=voxel,pixel_step=step,input_frames=len(poses),seconds=time.perf_counter()-start,
                    geometry_approved=False,unverified_bridge_count=report.get('unverified_bridge_count',0),
                    unverified_bridges=report.get('unverified_bridges',[]),units='first input frame median predicted depth = 1; metric scale unknown',
                    limitations=['No explicit long-range loop closure','Pixelwise depth shapes are unchanged','Held-out image consistency is not ground-truth accuracy'])
    save(out/'validation.json',validation);save(out/'settings.json',C)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sel=np.arange(0,len(xyz),max(1,len(xyz)//180000))
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax,(i,j,title) in zip(axes,[(0,2,'Horizontal projection (X/Z)'),(0,1,'Side projection (X/Y)')]):
        ax.scatter(xyz[sel,i],xyz[sel,j],c=rgb[sel]/255,s=.08,rasterized=True)
        ax.plot(poses[:,i,3],poses[:,j,3],color='#e64980',lw=1,alpha=.85)
        for bridge in report.get('unverified_bridges',[]):
            pair=[bridge['a'],bridge['b']]
            ax.plot(poses[pair,i,3],poses[pair,j,3],color='red',lw=3)
            ax.scatter(poses[pair,i,3],poses[pair,j,3],color='red',s=12)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('Normalized monocular units')
        lo,hi=np.quantile(xyz[:,[i,j]],[.005,.995],axis=0);pad=.03*(hi-lo)
        ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    fig.suptitle(f"{C['name']} | {C['sample_interval_s']} s | {C['width']} x {C['height']} | {merged:,} points | red: unverified bridges")
    fig.savefig(out/'overview.png',dpi=180);plt.close(fig)
    save(R/'export_status.json',dict(status='complete',**validation,files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()}))
    print('EXPORT_COMPLETE',json.dumps(validation),flush=True)

if __name__=='__main__':main()
