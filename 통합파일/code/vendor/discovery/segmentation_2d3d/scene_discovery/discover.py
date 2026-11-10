"""Unprompted candidate discovery; no input class list or hand-edited output."""
import os
from pathlib import Path
from .common import read,save,sha,digest,canonicalize


def reusable(config,output):
    output=Path(output)
    if not (output/'prepared.json').exists():return False
    rows=[r for s in read(output/'prepared.json')['scenes'] for r in s['views']]
    for row in rows:
        target=output/'discovery'/f'{row["key"]}.json'
        identity=digest([row['sha256'],config['models']['florence'],config.get('discovery_tasks')])
        try:
            record=read(target)
            if record.get('identity')!=identity or not isinstance(record.get('results'),dict):return False
        except (OSError,ValueError):return False
    return bool(rows)


def run(config,output):
    if reusable(config,output):
        rows=[r for s in read(Path(output)/'prepared.json')['scenes'] for r in s['views']]
        save(Path(output)/'discovery_complete.json',dict(views=len(rows),model=config['models']['florence'],
                                                       reused_matching_image_model_task_cache=True,new_inference=False))
        return
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForCausalLM
    if torch.cuda.device_count()!=1: raise RuntimeError('Exactly one leased GPU is required')
    torch.set_num_threads(2);torch.manual_seed(5201)
    model_path=config['models']['florence']['path']
    processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,
                                             torch_dtype=torch.float32,attn_implementation='eager').eval().cuda()
    output=Path(output)
    rows=[r for scene in read(output/'prepared.json')['scenes'] for r in scene['views']]
    for j,row in enumerate(rows):
        target=output/'discovery'/f'{row["key"]}.json'
        identity=digest([row['sha256'],config['models']['florence'],config.get('discovery_tasks')])
        if target.exists() and read(target).get('identity')==identity: continue
        if sha(row['path'])!=row['sha256']: raise ValueError('Prepared frame was modified')
        im=Image.open(row['path']).convert('RGB');results={}; candidates=[]
        def generate(task,prompt=None):
            inputs=processor(text=prompt or task,images=im,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                ids=model.generate(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],
                                   max_new_tokens=512,num_beams=1,do_sample=False,use_cache=True)
            text=processor.batch_decode(ids,skip_special_tokens=False)[0]
            return text,processor.post_process_generation(text,task=task,image_size=im.size)[task]
        for task in config.get('discovery_tasks',['<OD>','<MORE_DETAILED_CAPTION>']):
            text,parsed=generate(task)
            results[task]=dict(text=text,parsed=parsed)
            if isinstance(parsed,str):
                ground_task='<CAPTION_TO_PHRASE_GROUNDING>'
                ground_text,parsed=generate(ground_task,ground_task+parsed)
                results[ground_task]=dict(text=ground_text,parsed=parsed)
            if isinstance(parsed,dict):
                for label,box in zip(parsed.get('labels',[]),parsed.get('bboxes',[])):
                    name=canonicalize(label)
                    if name:
                        candidates.append(dict(name=name,phrase=label,box=box,task=task))
        save(target,dict(identity=identity,view=row,results=results,candidates=candidates))
        print('DISCOVER',j+1,len(rows),row['key'],sorted(set(c['name'] for c in candidates)),flush=True)
    save(output/'discovery_complete.json',dict(views=len(rows),model=config['models']['florence'],
                                              peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    import sys
    run(read(sys.argv[1]),sys.argv[2])
