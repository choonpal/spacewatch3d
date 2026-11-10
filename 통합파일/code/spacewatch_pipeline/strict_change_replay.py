"""Reaggregate exact saved model probabilities after strict original-mesh rays.

The requested producer remains unchanged as a comparison. This implements the
user's earlier no-through-wall requirement without another model invocation.
"""
from pathlib import Path
import os
import time
import numpy as np
from PIL import Image
from .io import read,save,sha

def atomic_npz(path,**values):
    path=Path(path);temporary=path.with_name(path.name+'.strict-tmp.npz')
    np.savez_compressed(temporary,**values);os.replace(temporary,path)

def replay(root):
    from tools.mesh_simsac.run import Evidence,qualified_evidence
    from tools.mesh_simsac.core import evidence_masks
    from tools.mesh_simsac.observation_quality import reference_consensus
    from tools.mesh_simsac.rgb_visibility import MeshVisibility
    root=Path(root);start=time.monotonic();m=read(root/'manifest.json');settings=m['settings']
    assert not m.get('strict_original_mesh_visibility'), 'Do not replay twice'
    oldchanges=read(root/'results/changes.json');oldaudit={r['id']:r for r in read(root/'results/pair-audit.json')}
    gate=MeshVisibility(root,m,settings['visibility_tolerance_u'])
    views={}
    for meta in m['views']:
        p=root/'views'/(meta['id']+'.npz')
        if sha(p)!=meta['geometry_sha256']:raise ValueError('Prepared geometry changed')
        with np.load(p) as z:views[meta['id']]=dict(meta,**dict(z))
    evidence=Evidence(settings);audits=[];view_results=[];removed=0;probabilities=[]
    for pair in read(root/'pairs.json'):
        q=views[pair['query']];support=np.zeros(q['valid'].shape,'u4');positive=np.zeros_like(support);probs=[]
        for rec in pair['refs']:
            ref=views[rec['id']];key=q['id']+'--'+ref['id'];path=root/'results/pairs'/(key+'.npz')
            if sha(path)!=oldaudit[key]['sha256']:raise ValueError('Original inference pair changed')
            with np.load(path) as z:prob=z['probability'].copy();oldcandidate=z['candidate'].copy()
            masks=evidence_masks(q,ref,prob,q['person'],ref['person'],settings['visibility_tolerance_u'],settings['probability_threshold'])
            masks,quality=qualified_evidence(q,ref,masks,settings)
            eligible=masks['observable'];locations=np.flatnonzero(eligible);visible=np.zeros(eligible.size,bool)
            if len(locations):
                points=q['points'].reshape(-1,3)[locations];regions=q['point_region'].ravel()[locations]
                _,own=gate.test(points,q,regions,q['epoch']);_,opposite=gate.test(points,ref,regions,ref['epoch'])
                visible[locations]=own&opposite
            visible=visible.reshape(eligible.shape);reject=eligible&~visible
            masks['unobserved_counterpart'] |= masks['candidate']&reject
            masks['candidate'] &= visible;masks['observable'] &= visible
            assert not (masks['candidate']&~oldcandidate).any(), 'Visibility filter cannot add positives'
            assert np.array_equal(sum(masks[k].astype(int) for k in ['candidate','excluded_person','unsupported_geometry','occluded','unobserved_counterpart']),masks['raw'].astype(int))
            removed+=int((oldcandidate&~masks['candidate']).sum())
            probability_sha=__import__('hashlib').sha256(prob.tobytes()).hexdigest()
            atomic_npz(path,probability=prob,**masks)
            with np.load(path) as z:assert __import__('hashlib').sha256(z['probability'].tobytes()).hexdigest()==probability_sha
            probabilities.append(dict(pair=key,probability_sha256=probability_sha))
            audits.append(dict(oldaudit[key],sha256=sha(path),object_observation_quality=quality,
                               pixels={k:int(v.sum()) for k,v in masks.items()}))
            support+=masks['observable'];positive+=masks['candidate'];probs.append(prob)
        observable=support>0;changed=reference_consensus(support,positive,settings['changed_frame_ratio'])
        evidence.add(q,observable,changed)
        mean=np.mean(probs,axis=0).astype('f4') if probs else np.zeros(support.shape,'f4')
        atomic_npz(root/'results/views'/(q['id']+'.npz'),mean_probability=mean,observable=observable,candidate=changed,
                   person=q['person'],reference_count=support)
        rgb=q['rgb'].copy();rgb[changed]=(.35*rgb[changed]+.65*np.array([255,75,30])).astype('u1')
        rgb[q['person']]=(.5*rgb[q['person']]+.5*np.array([60,120,255])).astype('u1')
        # JPEG result paths were hardlinked from immutable inference. Replace,
        # never overwrite the linked file in place.
        jpg=root/'results/views'/(q['id']+'.jpg');temp=jpg.with_name(jpg.stem+'.strict-tmp.jpg')
        Image.fromarray(rgb).save(temp,quality=92);os.replace(temp,jpg)
        view_results.append(dict(id=q['id'],epoch=q['epoch'],frame_id=q['frame_id'],references=len(pair['refs']),
            observed_pixels=int(observable.sum()),candidate_pixels=int(changed.sum()),state=pair['state']))
    objects,regions,voxels=evidence.finish(read(root/'objects.json'))
    m['strict_original_mesh_visibility']=True
    m['visibility_policy']='Exact original full-mesh first ray in both cameras before votes; hidden/unknown excluded, not no-change'
    save(root/'manifest.json',m)
    save(root/'results/changes.json',dict(oldchanges,objects=objects,regions=regions,candidate_voxels=voxels,views=view_results,
        source_manifest_sha256=sha(root/'manifest.json'),strict_original_mesh_visibility=True))
    # Descriptors depend only on unchanged source RGB/geometry/person masks,
    # not on changed-vote maps. Bind their metadata to the new visibility receipt.
    appearance=read(root/'results/object-appearance.json');appearance['source_manifest_sha256']=sha(root/'manifest.json')
    appearance['descriptor_reuse']='Exact source RGB/geometry/person masks unchanged; visibility changes votes only'
    save(root/'results/object-appearance.json',appearance)
    save(root/'results/pair-audit.json',audits)
    save(root/'strict-visibility.json',dict(seconds=time.monotonic()-start,removed_candidate_pixels=removed,
        source_pair_count=len(audits),model_probabilities_bit_exact=True,probabilities=probabilities,
        stats=dict(gate.stats),candidate_pixels_only_removed=True,unknown_is_no_change=False,
        models_rerun=False,original_source_method_preserved_as_separate_result=True))
