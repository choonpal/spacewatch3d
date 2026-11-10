#!/usr/bin/env python3
"""Audit exact full-mesh OBJ export and native/reference ray agreement (CPU)."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import trimesh
from spacewatch_pipeline.io import ROOT,read,save
sys.path[:0]=[str(ROOT/'vendor/spacewatch3d_src'),str(ROOT/'vendor/native')]
from spacewatch3d.tasks.task_5b_keyframes.visibility import BVH
import native_executor

def fingerprint(triangles):
    rows=np.asarray(triangles,dtype='<f8')
    order=np.lexsort((rows[:,:,2],rows[:,:,1],rows[:,:,0]),axis=1)
    rows=np.take_along_axis(rows,order[:,:,None],axis=1)
    return np.sort(np.ascontiguousarray(rows).reshape(-1,9).view('V72').reshape(-1))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=read(a.run/'results.json');cfg=read(a.run/'config.json');rows=[]
    for capture in cfg['captures']:
        prefix=capture['id'];start=time.monotonic();source=np.load(results[prefix+'.instances']['arrays']);original=source['vertices'][source['faces']].astype(float)
        path=Path(results[prefix+'.scene']['scene']);scene=read(path);rebuilt=[]
        for item in scene['objects']+scene['structures']:
            mesh=trimesh.load_scene(path.parent/item['mesh_uri'],process=False).to_mesh();rebuilt.append(mesh.triangles)
        rebuilt=np.concatenate(rebuilt);np.testing.assert_array_equal(fingerprint(original),fingerprint(rebuilt))
        bvh=BVH(original);rng=np.random.default_rng(5812);tri=original[rng.choice(len(original),48,replace=False)]
        targets=np.einsum('ij,ijk->ik',rng.dirichlet([1,1,1],len(tri)),tri);centers=np.asarray(scene['camera']['positions_m']);mismatch=0
        for origin in centers[np.linspace(0,len(centers)-1,4,dtype=int)]:
            mismatch+=int(np.count_nonzero(native_executor.ORIGINAL(bvh,origin,targets)!=native_executor.blocked_many(bvh,origin,targets)))
        assert mismatch==0
        rows.append(dict(capture=prefix,exported_faces=len(rebuilt),candidate_objects=len(scene['objects']),all_face_coordinates_exact=True,native_reference_rays=192,native_reference_disagreements=mismatch,seconds=time.monotonic()-start));print(rows[-1],flush=True)
    save(a.output,dict(status='passed',scenes=rows))

if __name__=='__main__':main()
