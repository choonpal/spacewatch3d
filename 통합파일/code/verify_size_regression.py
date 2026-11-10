#!/usr/bin/env python3
"""Compare v5's label adapter + solver with a saved full-mesh test NPZ."""
import argparse,json,time
from pathlib import Path
import numpy as np
from spacewatch_pipeline.size_instances import assumptions,canonical_labels
from spacewatch_pipeline.size_geometry import prepare,solve
from spacewatch_pipeline.io import read,ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--vocabulary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    catalog=read(a.vocabulary);_,spec=assumptions(read(ROOT/'configs/dorm2.json'));rows=[]
    for name in ['dorm2','dorm3']:
        start=time.monotonic();data=np.load(a.fixtures/name/'whole_mesh.npz')
        labels,inv,mapping=canonical_labels(data['face_class_id'],catalog,spec['priors'],{})
        r=solve(data['vertices'],data['faces'],spec['priors'],spec['settings'],semantic=labels,prepared=prepare(data['vertices'],data['faces']))
        np.testing.assert_array_equal(r['ids'],data['semantic_size_region_id']);np.testing.assert_array_equal(r['kinds'],data['semantic_size_kind'])
        assert r['counts']['object_candidate']==87
        rows.append(dict(capture=name,faces=len(data['faces']),counts=r['counts'],exact_region_ids=True,exact_region_kinds=True,seconds=time.monotonic()-start));print(name,rows[-1],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='passed',scenes=rows),indent=2)+'\n')

if __name__=='__main__':main()
