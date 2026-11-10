#!/usr/bin/env python3
"""Compare completed v5/v5.1 scientific outputs, excluding path provenance."""
import argparse
from pathlib import Path
import numpy as np
from spacewatch_pipeline.io import read, save, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();old=read(args.baseline/'results.json');new=read(args.run/'results.json')
    state=read(args.run/'status.json')
    if state['status']!='complete':raise ValueError('Pipeline has not completed')
    comparisons=[]
    for capture in read(args.run/'config.json')['captures']:
        name=capture['id'];meshes={}
        for stage in ['semantic','instances']:
            with np.load(old[name+'.'+stage]['arrays']) as a,np.load(new[name+'.'+stage]['arrays']) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                meshes[stage]=dict(all_arrays_equal=True,faces=len(a['faces']))
        a=read(old[name+'.plan']['plan']);b=read(new[name+'.plan']['plan'])
        keys=['planning_status','input_coverage_status','quality_status','publication_ready',
              'viewpoints','tour_visits','tour_links','coverage_report','unobservable_items','failures','warnings']
        differences=[k for k in keys if a[k]!=b[k]]
        if differences:raise AssertionError('Plan fields changed: '+name+' '+str(differences))
        ar=read(old[name+'.render']['renders'])['results'];br=read(new[name+'.render']['renders'])['results']
        assert len(ar)==len(br)
        images=[]
        for x,y in zip(ar,br):
            assert x['viewpoint_id']==y['viewpoint_id'] and x['reference_from_camera']==y['reference_from_camera']
            images.append(dict(viewpoint_id=x['viewpoint_id'],bytes_equal=sha(x['image_ref'])==sha(y['image_ref'])))
        comparisons.append(dict(capture=name,meshes=meshes,plan_fields_equal=keys,
            keyframes=len(b['viewpoints']),planning_status=b['planning_status'],renders=images))
    stage_times={}
    for stage in state['stages']:
        bp=args.baseline/'stages'/stage/'complete.json';npth=args.run/'stages'/stage/'complete.json'
        if bp.exists():stage_times[stage]=dict(v5_seconds=read(bp)['seconds'],v51_seconds=read(npth)['seconds'])
    save(args.output,dict(status='passed',baseline=str(args.baseline),run=str(args.run),
         scenes=comparisons,stage_compute_seconds=stage_times,v51_wall_seconds=state['wall_seconds'],
         comparison_scope='Scientific plan/mesh content; source paths and provenance excluded',
         cold_run=False,training='Imported existing 30000-step checkpoint, not retrained'))
    print(args.output)


if __name__=='__main__':main()
