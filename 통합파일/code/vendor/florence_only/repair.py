"""Retry length-limited polygon generations with the SAME Florence checkpoint.

No second segmentation model or box-as-mask fallback is used. Initial outputs
are kept verbatim for auditing. A retry still hitting the limit is rejected.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from florence_core import read, save, sha, rasterize, mask_box_metrics


def run(a):
    root=Path(a.output);summary=read(root/'inference_summary.json')
    pending=[]
    for key in summary['prediction_keys']:
        path=root/'predictions'/(key+'.json');r=read(path)
        if r['result']['output_tokens'] >= summary['max_tokens']+1 and 'retry' not in r:
            pending.append((path,r))
    if not pending:
        if not (root/'repair_summary.json').exists():save(root/'repair_summary.json',dict(attempted=0,recovered=0))
        print('NO_RETRY_REQUIRED');return
    import torch
    from transformers import AutoProcessor,AutoModelForCausalLM
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(5201)
    processor=AutoProcessor.from_pretrained(summary['model'],trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(summary['model'],trust_remote_code=True,local_files_only=True,
                  torch_dtype=torch.float32,attn_implementation='eager').eval().cuda()
    attempts=[]
    for path,r in pending:
        image=Image.open(r['view']['path']).convert('RGB')
        if sha(r['view']['path'])!=r['view']['sha256']:raise ValueError('Input changed')
        original=dict(result=r['result'],quality=r['quality'],mask_sha256=r['mask_sha256'])
        inputs=processor(text=r['result']['prompt'],images=image,return_tensors='pt').to('cuda')
        started=time.perf_counter()
        with torch.inference_mode():
            ids=model.generate(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],
                 max_new_tokens=summary['max_tokens'],num_beams=3,no_repeat_ngram_size=12,
                 do_sample=False,use_cache=True)
        raw=processor.batch_decode(ids,skip_special_tokens=False)[0]
        parsed=processor.post_process_generation(raw,task=r['result']['task'],image_size=image.size)[r['result']['task']]
        mask,q=rasterize(parsed,image.size);q.update(mask_box_metrics(mask,r['box']))
        limited=int(ids.shape[-1])>=summary['max_tokens']+1
        q['generation_limit_reached']=limited
        if limited:q.update(usable=False,rejection_reason='generation_limit_reached_after_retry')
        r['retry']=dict(reason='initial_generation_reached_token_limit',initial=original,
                        beams=3,no_repeat_ngram_size=12,model=summary['model'])
        r['result']=dict(task=r['result']['task'],prompt=r['result']['prompt'],raw=raw,parsed=parsed,
                        seconds=time.perf_counter()-started,output_tokens=int(ids.shape[-1]),beams=3,no_repeat_ngram_size=12)
        r['quality']=q
        np.savez_compressed(path.with_suffix('.npz'),mask=mask)
        r['mask_sha256']=sha(path.with_suffix('.npz'));save(path,r)
        attempts.append(dict(key=r['key'],label=r['label'],recovered=q['usable'],pixels=q['pixels'],
                             seconds=r['result']['seconds'],output_tokens=r['result']['output_tokens']))
        print('RETRY',attempts[-1],flush=True)
    save(root/'repair_summary.json',dict(attempted=len(attempts),recovered=sum(r['recovered'] for r in attempts),
         same_model=True,attempts=attempts,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run(p.parse_args())
