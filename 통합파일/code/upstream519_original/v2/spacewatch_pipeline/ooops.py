"""OOOPS inference isolated in its existing mmseg/PyTorch environment."""
import gc
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from .io import sha,save

def predict(job,images,output,names):
    runtime=job['config']['runtime'];base=Path(runtime['ooops_root'])
    sys.path.insert(0,str(base/'OPS'));os.chdir(base/'OPS')
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    register_all_modules()
    import ops
    torch.set_num_threads(2);torch.manual_seed(5201);torch.backends.cudnn.benchmark=False
    if torch.cuda.device_count()!=1: raise RuntimeError('Exactly one GPU must be visible')
    checkpoint=base/'weights/OOOPS_with_REPR.pt'
    if sha(checkpoint)!='896838b23a59725d54484de8fa87fad13dca09269fe52fd2b04fd61493caa2cf':
        raise ValueError('Unexpected OOOPS checkpoint')
    cfg=Config.fromfile(str(base/'OPS/configs/ops/ops-vit-b16_s2d3d-640x640.py'))
    cfg.model.pretrained=None;cfg.model.text_encoder.dataset_name=None
    cfg.model.text_encoder.vocabulary=names;cfg.model.decode_head.num_classes=len(names)
    cfg.model.decode_head.loss_decode[0].class_weight=[1.0]*len(names)+[.1]
    cfg.test_pipeline=[s for s in cfg.test_pipeline if s['type']!='LoadAnnotations']
    cfg.dump(str(output/'resolved_config.py'))
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state={k.removeprefix('module.'):v for k,v in checkpoint_data.get('state_dict',checkpoint_data).items()}
    model=MODELS.build(cfg.model);model.load_state_dict(state,strict=True)
    del state,checkpoint_data;gc.collect();model.eval().cuda()
    pipeline=Compose(cfg.test_pipeline)
    for frame,path in enumerate(images):
        resized=output/'input.png'
        Image.open(path).convert('RGB').resize((1024,512),Image.Resampling.BILINEAR).save(resized)
        with torch.inference_mode():
            data=pipeline(dict(img_path=str(resized)))
            data=model.data_preprocessor(dict(inputs=[data['inputs']],data_samples=[data['data_samples']]),False)
            scores=model.inference(data['inputs'],[s.metainfo for s in data['data_samples']])
            res=model.postprocess_result(scores,data['data_samples'])[0]
            score=res.seg_logits.data.float()
            if not torch.isfinite(score).all() or (score<0).any(): raise ValueError('Invalid OOOPS scores')
            top=score.topk(2,dim=0).values;denom=score.sum(0).clamp_min(1e-12)
            label=res.pred_sem_seg.data.squeeze().cpu().numpy().astype('u1')
            np.savez_compressed(output/f'{frame:05d}.npz',labels=label,
                support_fraction=(top[0]/denom).cpu().numpy().astype('f2'),
                margin_fraction=((top[0]-top[1])/denom).cpu().numpy().astype('f2'))
        if frame%10==0: print('OOOPS_FRAME',frame,len(images),flush=True)
    resized.unlink()
    save(output/'complete.json',dict(frames=len(images),vocabulary_sha256=sha(job['config']['vocabulary']),
                                    checkpoint_sha256=sha(checkpoint),strict_load=True))
