"""Explicit existing-label input. No inference, old instance IDs, or label guessing."""
from pathlib import Path
import numpy as np
from .io import ROOT,read,save,result,sha,verify
from .geometry import context,export_semantic

def checked_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        required=['vertices','faces','rgb','face_class_id']
        if any(k not in z.files for k in required):raise ValueError('Labeled NPZ requires '+', '.join(required))
        # Preserve original observation attributes, but never import previous segmentation decisions.
        keys=required+[k for k in ['vote_agreement','mean_winning_support','observations','geometric_observations','source_face_index'] if k in z.files]
        d={k:z[k] for k in keys}
    v,f,rgb,c=[d[k] for k in required]
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError('Invalid vertices')
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid faces')
    if rgb.shape!=v.shape or rgb.dtype!=np.uint8:raise ValueError('RGB must be uint8 Nx3')
    if c.shape!=(len(f),) or c.dtype.kind not in 'iu' or c.min()<0 or c.max()>255:raise ValueError('Class IDs must be integer face labels 0..255')
    for key in set(d)-set(required):
        if d[key].shape!=(len(f),) or not np.isfinite(d[key]).all():raise ValueError('Invalid per-face attribute: '+key)
    return d

def mesh(job):
    c,out,deps,prefix=context(job);path=Path(c['semantic_arrays']).resolve()
    external=[str(path)]
    if c.get('semantic_receipt'):
        receipt=read(c['semantic_receipt'])
        rec=next((r for r in receipt['artifacts'] if Path(r['path']).resolve()==path),None)
        if rec is None:raise ValueError('semantic_receipt does not certify the supplied arrays')
        verify(rec);external.append(c['semantic_receipt'])
    coords=read(c['coordinates'])
    if c['semantic_frame_id']!=prefix+'_mesh':raise ValueError('Supplied mesh frame does not match the capture')
    if c['semantic_coordinates_sha256']!=sha(c['coordinates']):raise ValueError('Supplied mesh coordinate digest mismatch')
    d=checked_arrays(path);np.savez_compressed(out/'geometry.npz',**{k:d[k] for k in ['vertices','faces','rgb']})
    result(out,geometry=str(out/'geometry.npz'),coordinates=deps[prefix+'.reconstruct']['coordinates'],
        vertices=len(d['vertices']),faces=len(d['faces']),geometry_sha256=sha(out/'geometry.npz'),
        semantic_labels_used_for_meshing=False,provided_geometry=True,source_label_arrays=str(path),external_artifacts=external)

def discovery(job):
    out=Path(job['output']);cfg=job['config'];path=Path(cfg['segmentation']['vocabulary'])
    catalog=read(path)
    ids=[r['id'] for r in catalog];names=[r['name'] for r in catalog]
    if ids!=list(range(len(ids))) or len(ids)>255 or len(set(names))!=len(names):raise ValueError('Vocabulary needs contiguous unique IDs 0..254 and unique names')
    for row in catalog:
        if not isinstance(row['name'],str) or not row['name'] or len(row['rgb'])!=3 or any(not isinstance(x,int) or not 0<=x<=255 for x in row['rgb']):raise ValueError('Invalid vocabulary entry')
    save(out/'classes.json',catalog)
    assumption=Path(cfg['size_segmentation']['assumptions']);assumption=assumption if assumption.is_absolute() else ROOT/assumption
    spec=read(assumption);save(out/'size_priors.json',spec)
    result(out,vocabulary=str(out/'classes.json'),size_priors=str(out/'size_priors.json'),classes=names,
        size_prior_classes=[p['name'] for p in spec['priors']],evidence_sources=[],source_mode='provided',
        inference_performed=False,external_artifacts=[str(path.resolve()),str(assumption.resolve())])

def semantic(job):
    c,out,deps,prefix=context(job);d=checked_arrays(c['semantic_arrays']);catalog=read(deps['discovery']['vocabulary'])
    valid={r['id'] for r in catalog}|{255}
    if not set(np.unique(d['face_class_id'])).issubset(valid):raise ValueError('Label not present in supplied vocabulary')
    original=np.load(deps[prefix+'.mesh']['geometry'])
    for key in ['vertices','faces','rgb']:
        if not np.array_equal(d[key],original[key]):raise ValueError('Provided semantic geometry changed: '+key)
    np.savez_compressed(out/'mesh_arrays.npz',**d)
    result(out,arrays=str(out/'mesh_arrays.npz'),vocabulary=deps['discovery']['vocabulary'],
        coordinates=deps[prefix+'.mesh']['coordinates'],inference_performed=False,
        source_mode='provided',**export_semantic(out,d,catalog),external_artifacts=[str(Path(c['semantic_arrays']).resolve())])
