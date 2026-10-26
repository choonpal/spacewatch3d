"""Prepare an explicitly provisional preview when independent alignment checks fail.

Never upgrades the hypothesis to verified. The published viewer must retain its
registration_status warning; original geometry and camera inputs stay untouched.
"""
import argparse
import shutil
import numpy as np
from scipy.spatial import cKDTree
from dorm_mesh_geometry import plane_samples,transform
from run_dorm_mesh_comparison import load,save


def prepare(geometry,photo_audit,photo_check,output):
    if output.exists():raise ValueError('Use a new empty output path')
    output.mkdir(parents=True)
    meta,seed,check=load(geometry/'alignment.json'),load(photo_audit),load(photo_check)
    saved=np.load(geometry/'alignment-samples.npz')
    a,na=plane_samples(saved['before'],meta['planes'][0]);b,nb=plane_samples(saved['after'],meta['planes'][1])
    matrix=np.array(seed['after_to_before']);h=meta['quality']['reference_height_units']
    moved=transform(b,matrix);distance,nearest=cKDTree(a).query(moved);reverse=cKDTree(moved).query(a)[0]
    scale=float(np.cbrt(np.linalg.det(matrix[:3,:3])));rotation=matrix[:3,:3]/scale
    good=(distance<h*.05)&(np.abs(np.einsum('ij,ij->i',nb@rotation.T,na[nearest]))>.9)
    residual=np.abs(np.einsum('ij,ij->i',na[nearest],moved-a[nearest]))
    meta['geometry_only_initial_alignment']=meta['after_to_before'];meta['after_to_before']=matrix.tolist()
    meta['registration_status']='provisional_corridor_ambiguity';meta['independent_photo_check']=check
    meta['independent_photo_check_passed']=False
    meta['quality']={'structural_overlap':float((good.mean()+(reverse<h*.05).mean())/2),
        'normal_residual_median_units':float(np.median(residual[good])),'relative_scale_after_to_before':scale,
        'reference_height_units':h,'structural_photo_feature_inliers':seed['inliers'],
        'structural_photo_feature_candidates':seed['candidates'],
        'method':'Provisional floor/wall photo-guided similarity. Neither hypothesis passed the independent-photo acceptance gate.'}
    save(output/'alignment.json',meta);shutil.copy2(geometry/'alignment-samples.npz',output/'alignment-samples.npz')
    protocol=load(geometry/'protocol.json');protocol.update(registration_status='provisional_corridor_ambiguity',
        registration_selection='Photo-supported hypothesis for provisional visualization only; independent acceptance gate failed.')
    save(output/'protocol.json',protocol)


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    for key in ('geometry','photo-audit','photo-check','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--allow-provisional-preview',action='store_true',required=True)
    args=parser.parse_args();prepare(args.geometry,args.photo_audit,args.photo_check,args.output)
