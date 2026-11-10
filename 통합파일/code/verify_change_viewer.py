#!/usr/bin/env python3
"""Validate the current attached comparison against the completed detector outputs."""
import argparse
from pathlib import Path
import numpy as np
from spacewatch_pipeline.io import read, save, sha
from spacewatch_pipeline.geometry import transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    deps=read(args.run/'results.json');service=read(args.run/'viewer-service.json')
    bundle=Path(service['bundle']);manifest=read(bundle/'manifest.json')
    data=read(bundle/'surface-comparison/comparison.json');source=read(deps['pair.changes']['changes'])
    assert data['source_changes_sha256']==sha(deps['pair.changes']['changes'])
    assert data['alignment']=={k:v for k,v in source['alignment'].items() if k!='alignment'}
    assert data['counts']==dict(candidates=deps['pair.changes']['candidates'],deferred=deps['pair.changes']['deferred'])
    by_id={e['id']:e for e in source['events']};assert len(by_id)==len(data['events'])
    for event in data['events']:
        expected=by_id[event['id']]
        for key,value in event.items():
            if key in ['own_views','opposite_views']:
                assert value==[{k:v for k,v in row.items() if k!='source_image'} for row in expected[key]]
            else:assert value==expected[key],(event['id'],key)
    rows=[]
    for model in data['maps']:
        name=model['id'];path=Path(deps[name+'.semantic']['arrays'])
        assert sha(path)==model['source_arrays_sha256']
        with np.load(path) as z:
            matrix=np.eye(4) if model['epoch']==0 else np.array(data['alignment']['baseline_from_current'])
            expected=transform(z['vertices'],matrix)[z['faces']].astype('<f4')
            binary=bundle/model['mesh'].removeprefix('/pipeline-assets/')
            raw=binary.read_bytes();count=len(z['faces'])*3
            assert len(raw)==16+count*30
            actual=np.frombuffer(raw,'<f4',count=count*3,offset=16).reshape(-1,3,3)
            np.testing.assert_array_equal(actual,expected)
            actual_rgb=np.frombuffer(raw,np.uint8,count=count*3,offset=16+count*24).reshape(-1,3,3)
            np.testing.assert_array_equal(actual_rgb,z['rgb'][z['faces']])
            rows.append(dict(capture=name,faces=len(z['faces']),exact_float32_geometry=True,exact_face_order=True,exact_rgb=True))
    assert manifest['changes']['viewer_kind']=='surface' and manifest['changes']['url']=='/pipeline-changes.html'
    for asset in ['surface-comparison/comparison.json']+[m['mesh'].removeprefix('/pipeline-assets/') for m in data['maps']]:
        assert asset in manifest['assets'] and (bundle/asset).is_file()
    save(args.report,dict(status='passed',bundle_id=manifest['bundle_id'],source_run=str(args.run),
                         original_detection_sha256=data['source_changes_sha256'],original_events_preserved=len(data['events']),
                         counts=data['counts'],alignment=data['alignment']['status'],meshes=rows,
                         processing='viewer export only; no reconstruction, detection, planning or rendering rerun'))
    print(args.report)


if __name__=='__main__':main()
