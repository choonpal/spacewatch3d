"""Refine whole-layout proposals using geometric floor and tall-wall support."""
import argparse
import json
from pathlib import Path
import numpy as np
from align_dorm_mesh_structure import smooth_structural_samples,metrics
from dorm_mesh_geometry import refine_similarity


def run(root,previous_folder,output):
    output.mkdir(exist_ok=False)
    previous=json.loads((previous_folder/'alignment.json').read_text())
    clouds=np.load(root/'inspection/alignment-samples.npz');h=previous['reference_height']
    before,nb,kb=smooth_structural_samples(clouds['before'],h)
    after,na,ka=smooth_structural_samples(clouds['after'],h)
    np.savez_compressed(output/'structure.npz',before=before,after=after,nb=nb,na=na,kb=kb,ka=ka)
    results=[]
    for hypothesis in previous['hypotheses'][:4]:
        matrix=np.array(hypothesis['matrix']);initial=metrics(after,before,na,nb,matrix,h);trace=[]
        refined=refine_similarity(after,before,na,nb,matrix,h,iterations=40,trace=trace)
        final=metrics(after,before,na,nb,refined,h)
        print('SMOOTH WALL / FLOOR',initial,'->',final,flush=True)
        results.append({'matrix':refined.tolist(),'initial':initial,'final':final,'trace':trace})
    results.sort(key=lambda row:row['final']['cost'])
    record={**previous,'method':'whole footprint proposals; local PCA floor patches and full-height wall columns; guarded robust Sim3 refinement',
        'source_counts':[len(before),len(after)],'hypotheses':results,'after_to_before':results[0]['matrix']}
    (output/'alignment.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('previous_folder',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();run(args.root,args.previous_folder,args.output)
