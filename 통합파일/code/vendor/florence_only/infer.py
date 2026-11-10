"""Florence-only OD -> per-object polygon segmentation, with auditable raw output."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from florence_core import read, save, sha, region_prompt, rasterize, mask_box_metrics
from florence_image import PerImageGeneration


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Run on the server through a single-GPU lease.')
    torch.set_num_threads(2)
    torch.manual_seed(5201)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prepared = read(Path(args.discovery) / 'prepared.json')
    rows = [r for s in prepared['scenes'] if s['id'] == args.scene for r in s['views']]
    if args.views:
        rows = [r for r in rows if r['key'].split('/')[-1] in args.views.split(',')]
    if not rows:
        raise ValueError('No selected views')
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation='eager').eval().cuda()
    print('MODEL_LOADED', round(time.perf_counter() - started, 2), flush=True)
    records = []
    new_mask_calls = 0
    image_cache_statistics = []
    def generate(image, task, prompt):
        ts = time.perf_counter()
        if image is not image_generation.image:
            raise ValueError('Image cache scope differs from current view')
        ids = image_generation.generate(prompt, max_new_tokens=args.max_tokens,
                    num_beams=args.beams, do_sample=False, use_cache=True)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=task, image_size=image.size)[task]
        torch.cuda.synchronize()
        return dict(task=task, prompt=prompt, raw=raw, parsed=parsed,
                    seconds=time.perf_counter()-ts, output_tokens=int(ids.shape[-1]))
    for view in rows:
        if sha(view['path']) != view['sha256']:
            raise ValueError('Prepared image hash differs: ' + view['path'])
        image = Image.open(view['path']).convert('RGB')
        image_generation = PerImageGeneration(model, processor, image)
        if args.fresh_od:
            od = generate(image, '<OD>', '<OD>')
        else:
            source_model = read(Path(args.discovery) / 'run.json')['config']['models']['florence']['path']
            if Path(source_model).resolve() != Path(args.model).resolve():
                raise ValueError('Cached detection model differs from requested model')
            cached = read(Path(args.discovery) / 'discovery' / (view['key'] + '.json'))
            if cached['view']['sha256'] != view['sha256']:
                raise ValueError('OD cache image differs')
            od = dict(cached['results']['<OD>'], reused=True, cache_path=str(Path(args.discovery) / 'discovery' / (view['key'] + '.json')))
        detections = list(zip(od['parsed']['labels'], od['parsed']['bboxes']))
        if args.max_objects:
            detections = detections[:args.max_objects]
        for index, (label, box) in enumerate(detections):
            for method in args.methods.split(','):
                task = '<REGION_TO_SEGMENTATION>' if method == 'region' else '<REFERRING_EXPRESSION_SEGMENTATION>'
                prompt = region_prompt(box, image.size) if method == 'region' else task + label
                key = f"{view['key']}/{index:03d}_{method}"
                path = out / 'predictions' / (key + '.json')
                identity = dict(image_sha256=view['sha256'], model=str(args.model), prompt=prompt,
                                beams=args.beams, max_tokens=args.max_tokens, float_dtype='float32',
                                infer_sha256=sha(__file__), core_sha256=sha(Path(__file__).with_name('florence_core.py')))
                identity['image_cache_sha256'] = sha(Path(__file__).with_name('florence_image.py'))
                if path.exists() and read(path).get('identity') == identity and path.with_suffix('.npz').exists() and read(path).get('mask_sha256') == sha(path.with_suffix('.npz')):
                    record = read(path)
                else:
                    result = generate(image, task, prompt)
                    new_mask_calls += 1
                    mask, quality = rasterize(result['parsed'], image.size)
                    quality.update(mask_box_metrics(mask, box))
                    record = dict(key=key, identity=identity, view=view, detection_index=index, label=label,
                                  box=box, method=method, result=result, quality=quality)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path.with_suffix('.npz'), mask=mask)
                    record['mask_sha256'] = sha(path.with_suffix('.npz'))
                    save(path, record)
                records.append(record)
                print('MASK', key, label, 'pixels', record['quality']['pixels'],
                      'seconds', round(record['result']['seconds'], 2),
                      'raw', record['result']['raw'][:140], flush=True)
        save(out / 'od' / (view['key'] + '.json'), dict(view=view, result=od))
        print('VIEW', view['key'], 'detections', len(detections), flush=True)
        image_cache_statistics.append(dict(key=view['key'], **image_generation.statistics()))
        del image_generation
    from collections import Counter
    summary = dict(model=str(args.model), segmentation_models=['Florence-2'],
        other_segmentation_models_used=False, scene=args.scene, views=len(rows),
        detections=sum(len(read(out/'od'/(r['key']+'.json'))['result']['parsed']['labels']) for r in rows),
        mask_calls=len(records), usable_masks=sum(r['quality']['usable'] for r in records),
        methods=dict(Counter(r['method'] for r in records)), max_tokens=args.max_tokens, beams=args.beams,
        od_reused=not args.fresh_od, new_segmentation_inference=bool(new_mask_calls),
        new_mask_calls=new_mask_calls, prediction_keys=[r['key'] for r in records],
        per_image_cache=image_cache_statistics,
        source_prepared_sha256=sha(Path(args.discovery)/'prepared.json'),
        wall_seconds=time.perf_counter()-started, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    save(out / 'inference_summary.json', summary)
    print('SUMMARY', summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--discovery', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scene', default='dorm2')
    p.add_argument('--views', default='')
    p.add_argument('--max-objects', type=int, default=0)
    p.add_argument('--methods', default='region')
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--beams', type=int, default=1)
    p.add_argument('--fresh-od', action='store_true')
    run(p.parse_args())
