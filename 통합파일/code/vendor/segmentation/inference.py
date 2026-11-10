"""MTPano semantic inference; always launched in a separate single-GPU process."""
import ast
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
from common import read_json, save_json, progress, valid_stage, complete_stage

def gpu_guard(config):
    e = config['execution']
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    found = False
    for row in rows.strip().splitlines():
        uuid, memory, utilization = [v.strip() for v in row.split(',')]
        if uuid != e['gpu_uuid']:
            continue
        found = True
        if int(memory) > e['max_gpu_memory_mib'] or int(utilization) > e['max_gpu_utilization_percent']:
            raise RuntimeError(f'Configured GPU busy: {memory} MiB, {utilization}%. Retry later; no other GPU is used.')
    if not found:
        raise RuntimeError('Configured GPU UUID is absent')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    if any(row.split(',')[0].strip() == e['gpu_uuid'] for row in processes.splitlines()):
        raise RuntimeError('Configured GPU already has a compute process; retry later.')

def run(config, plan):
    import fcntl
    gpu_lock = open('/tmp/segmentation-2d3d-' + config['execution']['gpu_uuid'] + '.lock', 'a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = Path(config['output'])
    target = out / 'frames'
    target.mkdir(exist_ok=True)
    identity = plan['identity']
    pending = [row for row in plan['frames'] if not valid_stage(target / f"{row['index']:05d}.json", identity)]
    if not pending:
        progress(out, 'inference_cached', completed=len(plan['frames']))
        return
    gpu_guard(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['execution']['gpu_uuid']
    os.environ['HF_HOME'] = config['model']['hf_home']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, config['model']['repository'])
    import numpy as np
    import torch
    from PIL import Image
    from easydict import EasyDict as edict
    from utils.common_config import get_model
    torch.set_num_threads(config['execution']['cpu_threads'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    tree = ast.parse((Path(config['model']['repository']) / 'inference.py').read_text())
    nodes = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name == 'get_inference_config') or
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ADE20K_PALETTE' for t in node.targets))]
    namespace = {'edict': edict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'upstream_config', 'exec'), namespace)
    palette = np.asarray(namespace['ADE20K_PALETTE'], dtype=np.uint8)
    names = read_json(config['model']['class_names'])['id2label']
    classes = [{'id': i, 'name': names[str(i)].strip(), 'rgb': palette[i].tolist()} for i in range(150)]
    classes.append({'id': 255, 'name': 'unclassified', 'rgb': [90, 96, 107]})
    save_json(out / 'classes.json', classes)
    started = time.perf_counter()
    progress(out, 'loading_model', remaining_frames=len(pending))
    checkpoint = torch.load(config['model']['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    state = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
    model = get_model(namespace['get_inference_config']())
    model.load_state_dict(state, strict=True)
    assert any(k.startswith('backbone.') for k in state)
    parameter_count = sum(p.numel() for p in model.parameters())
    del checkpoint, state
    gc.collect()
    assert torch.cuda.device_count() == 1, 'Exactly one visible GPU required'
    model.eval().cuda()
    mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1).cuda()
    std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1).cuda()
    load_seconds = time.perf_counter() - started
    forward_total = 0
    for done, row in enumerate(pending):
        rgb = Image.open(row['path']).convert('RGB').resize((1024, 512), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255).permute(2, 0, 1)[None].cuda()
        tensor = (tensor - mean) / std
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            predictions = model(tensor)
            logits = predictions['semseg']
            assert logits.shape == (1, 150, 512, 1024) and torch.isfinite(logits).all()
            labels = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
            confidence = torch.exp(logits.amax(1) - torch.logsumexp(logits, dim=1)).squeeze().cpu().numpy().astype(np.float32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        forward_total += elapsed
        p = target / f"{row['index']:05d}.npz"
        temp = p.with_suffix('.partial.npz')
        np.savez_compressed(temp, labels=labels, confidence=confidence)
        temp.replace(p)
        overlay = target / f"{row['index']:05d}_overlay.jpg"
        Image.fromarray(np.rint(np.asarray(rgb) * .45 + palette[labels] * .55).astype(np.uint8)).save(overlay, quality=90)
        complete_stage(target / f"{row['index']:05d}.json", identity, [p, overlay, out / 'classes.json'], frame=row,
                       forward_seconds=elapsed, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        progress(out, 'inference', completed=len(plan['frames']) - len(pending) + done + 1,
                 total=len(plan['frames']), timestamp_s=row['timestamp_s'])
        del predictions, logits, tensor
    save_json(out / 'inference_run.json', dict(strict_load=True, parameters=parameter_count,
              gpu=torch.cuda.get_device_name(), gpu_uuid=config['execution']['gpu_uuid'],
              load_seconds=load_seconds, forward_seconds_this_run=forward_total,
              generated_frames=len(pending), resolution=[1024, 512], precision='float32'))

if __name__ == '__main__':
    run(read_json(sys.argv[1]), read_json(sys.argv[2]))
