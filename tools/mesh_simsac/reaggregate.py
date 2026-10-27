"""Re-evaluate an immutable run's actual probabilities without model inference."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .core import read, write, sha, evidence_masks
from .observation_quality import reference_consensus,quality_settings
from .run import Evidence, qualified_evidence


def execute(source, manifest, output, runtime):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    root=output/'detection';root.mkdir(exist_ok=False)
    started=time.time()
    write(output/'status.json',dict(state='running',stage='classification'))
    try:
        if sha(source/'viewer-data.json')!=runtime['reprocess_parent_viewer_sha256']:
            raise ValueError('Parent result changed before reaggregation')
        if read(source/'validation.json')['state']!='passed':
            raise ValueError('Parent inference did not pass validation')
        for path in source.iterdir():
            if path.name=='views':shutil.copytree(path,root/path.name)
            elif path.suffix=='.npz' or path.name in ['manifest.json','objects.json','pairs.json','registration.json'] or path.name.startswith('pointcloud-'):
                if path.is_file():shutil.copyfile(path,root/path.name)
        results=root/'results';results.mkdir();(results/'pairs').mkdir();(results/'views').mkdir()
        for name in ['model.json','object-appearance.json','object-appearance.npz']:
            shutil.copyfile(source/'results'/name,results/name)
        for path in source.glob('results/self-*.npz'):shutil.copyfile(path,results/path.name)
        prepared=read(root/'manifest.json');settings=prepared['settings'];metadata={v['id']:v for v in prepared['views']}
        pair_audits={p['id']:p for p in read(source/'results/pair-audit.json')}
        evidence=Evidence(settings);audits=[];view_rows=[];cache={}
        def load(vid):
            if vid not in cache:cache[vid]={**metadata[vid],**dict(np.load(root/'views'/(vid+'.npz')))}
            return cache[vid]
        persons={v['id']:np.load(source/'results/views'/(v['id']+'.npz'))['person'] for v in prepared['views']}
        for index,pair in enumerate(read(root/'pairs.json')):
            query=load(pair['query']);support=np.zeros(query['valid'].shape,'u4');positive=np.zeros_like(support);probabilities=[]
            for record in pair['refs']:
                key=query['id']+'--'+record['id'];previous=pair_audits[key];path=source/previous['path']
                if sha(path)!=previous['sha256']:raise ValueError('Recorded probability file changed')
                probability=np.load(path)['probability']
                masks=evidence_masks(query,load(record['id']),probability,persons[query['id']],persons[record['id']],
                    settings['visibility_tolerance_u'],settings['probability_threshold'])
                masks,quality=qualified_evidence(query,load(record['id']),masks,settings)
                support+=masks['observable'];positive+=masks['candidate'];probabilities.append(probability)
                target=results/'pairs'/(key+'.npz');np.savez_compressed(target,probability=probability,**masks)
                audits.append({**previous,'path':str(target.relative_to(root)),'sha256':sha(target),
                    'object_observation_quality':quality,'pixels':{k:int(v.sum()) for k,v in masks.items()}})
            observable=support>0;changed=reference_consensus(support,positive,settings['changed_frame_ratio'])
            evidence.add(query,observable,changed)
            mean=np.mean(probabilities,axis=0).astype('f4') if probabilities else np.zeros(support.shape,'f4')
            person=persons[query['id']]
            np.savez_compressed(results/'views'/(query['id']+'.npz'),mean_probability=mean,observable=observable,
                candidate=changed,person=person,reference_count=support)
            overlay=query['rgb'].copy();overlay[changed]=(.35*overlay[changed]+.65*np.array([255,75,30])).astype('u1')
            Image.fromarray(overlay).save(results/'views'/(query['id']+'.jpg'))
            view_rows.append(dict(id=query['id'],epoch=query['epoch'],frame_id=query['frame_id'],references=len(pair['refs']),
                observed_pixels=int(observable.sum()),candidate_pixels=int(changed.sum()),state=pair['state']))
            if index%30==0:print('REAGGREGATE',index+1,len(prepared['views']),flush=True)
        objects,regions,voxels=evidence.finish(read(root/'objects.json'))
        previous=read(source/'results/changes.json')
        write(results/'changes.json',{**previous,'objects':objects,'regions':regions,'candidate_voxels':voxels,'views':view_rows})
        write(results/'pair-audit.json',audits)
        write(root/'inference-status.json',dict(state='complete',pairs=len(audits),reused_probabilities=True,seconds=time.time()-started))
        from .classify import classify
        from .validate import validate
        from .preview import export
        from .build_event_viewer import build
        classify(root,manifest)
        validation=validate(root)
        write(root/'validation.json',{**validation,'fresh_inference_pairs':0,'evaluated_probability_pairs':len(audits),
            'reused_probability_pairs':len(audits),'new_model_predictions':0})
        export(root,manifest,runtime['three_package']);build(root,manifest)
        result=read(root/'viewer-data.json');original=read(manifest)
        result['labels']=[original[p]['id'] for p in ['before','after']]
        result['job_provenance']=dict(fresh_inference=False,reused_probabilities=True,
            parent_viewer_sha256=sha(source/'viewer-data.json'),parent_pair_audit_sha256=sha(source/'results/pair-audit.json'),
            new_model_predictions=0,object_quality=quality_settings(settings),
            reference_vote='Strict majority; one positive and one negative do not make a change',gaussian_inputs_used=False)
        write(root/'viewer-data.json',result)
        write(output/'status.json',dict(state='complete',stage='complete',counts=result['counts'],inference_pairs=len(audits),
            reused_probabilities=True,new_model_predictions=0,seconds=time.time()-started))
    except Exception as error:
        write(output/'status.json',dict(state='failed',stage='failed',message=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['source','manifest','output','runtime']:parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();execute(args.source,args.manifest,args.output,read(args.runtime))
