#!/usr/bin/env python3
"""Real-data CPU checks: export order, shared correspondence, registration."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from spacewatch_pipeline.io import ROOT, read, save, sha


def registration(out):
    from scipy.spatial.transform import Rotation
    source=Path('/workspace/dorm2_pointcloud_v2_20261017')
    os.environ['PANOVGGT_RUN_ROOT']=str(source)
    sys.path.insert(0,str(ROOT/'vendor/panovggt_pipeline_v2/workers'))
    import refine
    with np.load(source/'observations.npz') as z:
        obs=z['train'];obs=obs[(obs[:,0]<20)&(obs[:,1]<20)]
    with np.load(source/'optimized.npz') as z:initial=z['poses'][:20].copy()
    rng=np.random.default_rng(67);initial[1:,:3,3]+=rng.normal(0,.005,(19,3))
    outputs={};report={}
    for analytic in [False,True]:
        refine.O['analytic_jacobian']=analytic
        pose,details=refine.Problem(obs,initial.copy()).solve(max_nfev=80)
        name='analytic' if analytic else 'finite_difference';outputs[name]=pose;report[name]=details
    report['max_pose_difference']=float(np.max(abs(outputs['analytic']-outputs['finite_difference'])))
    np.testing.assert_allclose(outputs['analytic'],outputs['finite_difference'],atol=3e-4,rtol=3e-4)
    if abs(report['analytic']['cost']-report['finite_difference']['cost'])>max(1e-8,report['finite_difference']['cost']*1e-4):
        raise AssertionError('Registration costs differ')
    report['observations']=len(obs);report['frames']=20
    save(out/'registration.json',report);print('REGISTRATION',report,flush=True)


def index_worker(source, output, vendor):
    sys.path.insert(0,str(vendor));from geometry import build_index, source_frames
    from core import voxel_keys,buckets,exact_lookup
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    build_index(dict(source_run=str(source),output=str(output)),dict(identity='v51-real-export-check',source_frames=6))
    result=dict(seconds=time.perf_counter()-start)
    if vendor==ROOT/'vendor/segmentation':
        from spacewatch_pipeline.correspondence import ExportCorrespondence
        shared=ExportCorrespondence(source);index=shared.index()
        arrays=[np.load(r['path'],mmap_mode='r') for r in index['buckets']];pixels=0
        for frame,xyz,mask,_ in source_frames(dict(source_run=str(source))):
            keys=voxel_keys(xyz,index['voxel']);bucket=buckets(keys,len(arrays));ids=np.empty(len(keys),np.uint32)
            for k,base in enumerate(arrays):
                selected=bucket==k;ids[selected]=exact_lookup(base,keys[selected])+index['buckets'][k]['offset']
            actual,actual_mask=shared.frame(frame)
            np.testing.assert_array_equal(actual,ids);np.testing.assert_array_equal(actual_mask,mask);pixels+=len(ids)
        result.update(exact_pixel_row_parity=True,verified_pixels=pixels)
    save(output/'benchmark.json',result)


def export(out):
    from plyfile import PlyData
    source=Path('/workspace/dorm2_pointcloud_v2_20261017')
    config=read(source/'config.json');config['window_indices']=config['window_indices'][:1]
    config['frames']=6;config['output']['pixel_step']=2
    report={}
    for version,root in [('v5',Path('/workspace/spacewatch_pipeline_v5')),('v51',ROOT)]:
        target=out/('export_'+version);target.mkdir();(target/'input').mkdir();(target/'windows').mkdir()
        manifest=read(source/'input/manifest.json');manifest['frames']=manifest['frames'][:6]
        save(target/'input/manifest.json',manifest);save(target/'config.json',config)
        (target/'windows/window_000.npz').symlink_to(source/'windows/window_000.npz')
        (target/'windows/window_000.json').symlink_to(source/'windows/window_000.json')
        with np.load(source/'prepared.npz') as z:
            np.savez(target/'prepared.npz',scales=z['scales'][:1],owners=np.c_[np.zeros(6,int),np.arange(6)])
        with np.load(source/'optimized.npz') as z:np.savez(target/'optimized.npz',poses=z['poses'][:6])
        save(target/'registration_report.json',dict(status='converged'))
        save(target/'source_hashes.json',{str(source/'windows/window_000.npz'):sha(source/'windows/window_000.npz')})
        env={**os.environ,'PANOVGGT_RUN_ROOT':str(target),'PYTHONPATH':str(ROOT)}
        start=time.perf_counter()
        with (out/(version+'_export.log')).open('w') as log:
            subprocess.run([sys.executable,str(root/'vendor/panovggt_pipeline_v2/workers/export.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cloud=target/'unfiltered_result'/config['output']['ply_name'];n=len(PlyData.read(cloud,mmap='r')['vertex'].data)
        report[version]=dict(seconds=time.perf_counter()-start,sha256=sha(cloud),points=n)
        (target/'result').mkdir();(target/'result'/config['output']['ply_name']).symlink_to(cloud)
        (target/'component_filter').mkdir();np.save(target/'component_filter/keep_mask.npy',np.ones(n,bool))
        save(target/'component_filter/report.json',dict(source=str(cloud),output=str(target/'result'/config['output']['ply_name'])))
        with (out/(version+'_index.log')).open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(out), '--index-source',str(target),
                '--index-vendor',str(root/'vendor/segmentation')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report[version]['index']=read(target/'index_test/benchmark.json')
    if report['v5']['sha256']!=report['v51']['sha256']:raise AssertionError('Export PLY bytes changed')
    report['exact_ply_parity']=True;save(out/'export.json',report);print('EXPORT',report,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',choices=['registration','export'])
    p.add_argument('--index-source');p.add_argument('--index-vendor');args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.index_source:index_worker(Path(args.index_source),Path(args.index_source)/'index_test',Path(args.index_vendor));return
    globals()[args.worker](out)


if __name__=='__main__':main()
