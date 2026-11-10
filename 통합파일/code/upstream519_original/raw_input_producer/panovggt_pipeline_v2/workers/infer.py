"""Fresh dorm3 inference, restricted to one physical GPU; all caches stay here."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, time, hashlib, subprocess, fcntl, traceback
from runtime import R,C,gpu_guard,replay_files
os.environ['CUDA_VISIBLE_DEVICES'] = str(C['physical_gpu'])
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = str(C['execution']['cpu_threads'])

def save(p, v):
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(p)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def main():
    lock = (R / 'inference.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names=[f'window_{i:03d}{ext}' for i in range(len(C['window_indices'])) for ext in ['.npz','.json']]
    if replay_files('windows',names):
        for wi,ids in enumerate(C['window_indices']):
            p=R/'windows'/f'window_{wi:03d}.npz';meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['indices']==ids and digest(p)==meta['sha256']
        save(R/'inference_status.json',dict(status='complete',mode='verified_cache_replay',windows=len(C['window_indices']),gpu_used=False))
        return
    gpu_guard()
    revision=subprocess.check_output(['git','-C',C['model']['repository'],'rev-parse','HEAD'],text=True).strip()
    assert revision==C['model']['revision'], 'PanoVGGT source revision mismatch'
    import numpy as np
    import torch, engine
    assert torch.cuda.device_count() == 1
    manifest = engine.manifest
    masks = json.loads((R / 'reviewed_mask_manifest.json').read_text())
    for row, mask in zip(manifest['frames'], engine.mask_paths):
        assert digest(R / 'input' / row['file']) == row['sha256']
        assert digest(mask) == masks[mask.name]
    assert json.loads((R / 'mask_review.json').read_text())['reviewed']
    assert digest(engine.CHECKPOINT) == C['checkpoint_sha256']
    torch.manual_seed(C['seed']); np.random.seed(C['seed'])
    start = time.perf_counter(); engine.load_model()
    save(R / 'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
         physical_gpu=C['physical_gpu'], visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
         precision=str(engine.AMP_DTYPE), checkpoint_sha256=C['checkpoint_sha256'], model_load_s=time.perf_counter()-start))
    out = R / 'windows'; out.mkdir(exist_ok=True)
    timings = {}; started = datetime.now(timezone.utc).isoformat()
    for wi, ids in enumerate(C['window_indices']):
        p = out / f'window_{wi:03d}.npz'; meta = p.with_suffix('.json')
        if p.exists() and meta.exists():
            record = json.loads(meta.read_text())
            assert record['indices'] == ids and record['sha256'] == digest(p)
            timings[str(wi)] = record; continue
        save(R / 'inference_status.json', dict(status='inference', pid=os.getpid(), started_at=started,
             completed_windows=len(timings), total_windows=len(C['window_indices']), active_window=wi, physical_gpu=C['physical_gpu']))
        data, info = engine.predict_precise(ids, 1)
        t = time.perf_counter(); tmp = p.with_suffix('.partial.npz')
        np.savez(tmp, indices=np.array(ids), **data); tmp.replace(p)
        info.update(window_write_s=time.perf_counter()-t, sha256=digest(p), indices=ids, physical_gpu=C['physical_gpu'])
        save(meta, info); timings[str(wi)] = info; save(R / 'inference_timing.json', timings)
        print('WINDOW_COMPLETE', wi+1, '/', len(C['window_indices']), info['forward_gpu_s'], flush=True)
        del data
    del engine.model
    torch.cuda.empty_cache()
    save(R / 'inference_status.json', dict(status='complete', pid=os.getpid(), windows=len(timings),
         started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter()-start,
         gpu_forward_s=sum(v['forward_gpu_s'] for v in timings.values()), physical_gpu=C['physical_gpu']))

if __name__ == '__main__':
    try: main()
    except BaseException as e:
        save(R / 'inference_status.json', dict(status='failed', error=str(e), traceback=traceback.format_exc()))
        raise
