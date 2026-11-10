from pathlib import Path
import os,sys,json,gc,time
import numpy as np
import torch
from PIL import Image
from runtime import R,C
REPO=Path(C['model']['repository'])
CHECKPOINT=Path(C['model']['checkpoint'])
manifest=json.loads((R/'input/manifest.json').read_text())
frame_paths=[R/'input'/x['file'] for x in manifest['frames']]
mask_paths=[R/'masks'/(p.stem+'.png') for p in frame_paths]

def load_model():
    global model, AMP_DTYPE
    # Strict checkpoint loading; inference remains FP32.
    import gc
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf

    os.environ['XFORMERS_DISABLED'] = '1'
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from panovggt.models.aggregator import Aggregator
    from panovggt.models.panovggt_model import PanoVGGTModel

    # 공식 checkpoint에 DINOv2 가중치도 있으므로 중복 다운로드를 생략합니다.
    # Meta initialization avoids holding duplicate CPU weight copies.
    cfg = OmegaConf.load(REPO / 'training/config/default.yaml')
    OmegaConf.resolve(cfg)
    load_backbone = Aggregator._try_load_dinov2
    Aggregator._try_load_dinov2 = lambda *args, **kwargs: None
    original_linspace = torch.linspace
    def cpu_initialization_constants(*args, **kwargs):
        # DINO constructor reads stochastic-depth constants with .item().
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)
    torch.linspace = cpu_initialization_constants
    try:
        with torch.device('meta'):
            model = PanoVGGTModel(
                img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
                enable_camera=cfg.model.enable_camera, enable_depth=cfg.model.enable_depth,
                enable_point=cfg.model.enable_point,
                aggregator=OmegaConf.to_container(cfg.model.aggregator, resolve=True))
    finally:
        Aggregator._try_load_dinov2 = load_backbone
        torch.linspace = original_linspace
    state = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    for key in ('model_state_dict', 'model', 'state_dict'):
        if key in state:
            state = state[key]
            break
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # persistent=False인 상수 버퍼는 checkpoint에 없으므로 원본 소스의 값으로 복원합니다.
    model.aggregator._resnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    model.aggregator._resnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    for adapter in model.pos_adapters.values():
        adapter._res_scale = torch.tensor(0.0)
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    model.eval().requires_grad_(False).to('cuda')
    # This pipeline uses world_points from local depth + camera poses, never global_points.
    # The independent final global_points branch cannot affect those earlier outputs.
    model.enable_global_points = False
    if C.get('execution', {}).get('dino_feature_cache', True):
        from feature_cache import FrameFeatureCache
        model.aggregator.patch_embed = FrameFeatureCache(model.aggregator.patch_embed,
            int(C['execution'].get('dino_cache_mib', 512)) * 1024**2).eval()
    del state
    gc.collect()
    torch.set_num_threads(2)
    AMP_DTYPE = torch.float32
    print('가중치 전체 일치 확인. GPU:', torch.cuda.get_device_name(0), 'AMP:', AMP_DTYPE)
    print('모델 GPU 메모리 GB:', round(torch.cuda.memory_allocated() / 1e9, 3))

def predict_precise(indices, pixel_step):
    t0=time.perf_counter()
    frames=np.stack([np.asarray(Image.open(frame_paths[i]).convert('RGB')) for i in indices])
    masks=np.stack([np.asarray(Image.open(mask_paths[i]))>0 for i in indices])
    assert frames.shape[1:3]==(C['height'],C['width'])
    images=torch.from_numpy(frames.copy()).permute(0,3,1,2).float().div_(255).unsqueeze(0).cuda()
    torch.cuda.synchronize()
    prep_s=time.perf_counter()-t0
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    feature_cache = model.aggregator.patch_embed
    if hasattr(feature_cache, 'set_frames'):
        feature_cache.set_frames([('rgb-fp32-resnet-v1', manifest['frames'][i]['sha256']) for i in indices])
    with torch.inference_mode(),torch.autocast('cuda',dtype=AMP_DTYPE,enabled=AMP_DTYPE!=torch.float32):
        pred=model(images)
    torch.cuda.synchronize()
    forward_s=time.perf_counter()-started
    copy_start=time.perf_counter()
    points=pred['world_points'][0].float().cpu().numpy()
    depths=pred['depth'][0,...,0].float().cpu().numpy()
    poses=pred['camera_poses'][0].float().cpu().numpy()
    valid=np.isfinite(points).all(-1)&np.isfinite(depths)&(depths>0)&~masks
    valid[:,np.arange(C['height'])/C['height']<C['top_fraction_excluded'],:]=False
    for i in range(len(indices)):
        if not valid[i].any():raise ValueError('Frame has no valid unmasked depth pixels')
        lo,hi=np.percentile(depths[i][valid[i]],C['depth_percentiles'])
        valid[i]&=(depths[i]>=lo)&(depths[i]<=hi)
    assert np.isfinite(poses).all() and valid.any()
    result=dict(points=points[:,::pixel_step,::pixel_step].copy(),
                depths=depths[:,::pixel_step,::pixel_step].copy(),poses=poses,
                colors=frames[:,::pixel_step,::pixel_step].copy(),
                valid=valid[:,::pixel_step,::pixel_step].copy())
    info=dict(seconds=time.perf_counter()-started,preprocess_s=prep_s,forward_gpu_s=forward_s,
              copy_and_filter_s=time.perf_counter()-copy_start,
              peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,height=C['height'],pixel_step=pixel_step)
    if hasattr(feature_cache, 'statistics'):
        info['dino_feature_cache'] = feature_cache.statistics()
    del pred,images,points,depths,valid
    # Retain CUDA allocator blocks and overlap features between windows.
    # Optional diagnostic compatibility switch, never necessary for correctness.
    if C['execution'].get('release_cuda_cache_each_window', False):
        gc.collect(); torch.cuda.empty_cache()
    return result,info
