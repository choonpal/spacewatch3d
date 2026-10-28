"""Server-only model stages; called exclusively through launch.py."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

from pipeline import read, save, sha

ROOT=Path(__file__).resolve().parent


def main():
    output=Path(sys.argv[1]).resolve()
    cfg,plan=read(output/'effective_config.json'),read(output/'plan.json')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    clips=[clip for clip in plan['clips'] if not clip['reused']]
    sam=cfg['sam']
    tracker=build_sam2_video_predictor(sam['model_config'],sam['checkpoint'],device='cuda',
             apply_postprocessing=False,hydra_overrides_extra=['++model.fill_hole_area=0'])
    checkpoint_sha=sha(sam['checkpoint'])
    for clip in clips:
        started=time.perf_counter()
        folder=Path(clip['directory'])
        prompt=clip['prompt']
        state=tracker.init_state(video_path=str(folder/'sam_frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        tracker.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=1,
                points=np.asarray(prompt['points'],dtype='float32'),
                labels=np.asarray(prompt.get('labels',[1]*len(prompt['points'])),dtype='int32'),
                box=np.asarray(prompt['box'],dtype='float32'))
        rows=[]
        close=cfg['mask']['close_kernel_pixels']
        radius=cfg['mask']['dilation_radius_pixels']
        for index,ids,logits in tracker.propagate_in_video(state):
            raw=(logits[0,0]>0).cpu().numpy().astype('uint8')*255
            if not raw.any():
                raise RuntimeError(f'Empty person mask in {clip["id"]}:{index}.')
            mask=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close,close)))
            if radius:
                mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
            cv2.imwrite(str(folder/'raw_masks'/f'{index:05d}.png'),raw)
            cv2.imwrite(str(folder/'masks'/f'{index:05d}.png'),mask)
            rows.append(dict(frame=index,mask_touches_border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
                             expanded_fraction=float((mask>0).mean())))
        if {r['frame'] for r in rows} != set(range(clip['frame_count'])):
            raise RuntimeError('Tracker did not produce every frame.')
        selected={clip['video_indices'].index(t['native_video_frame']) for t in plan['targets'] if t['clip']==clip['id']}
        if any(row['mask_touches_border'] for row in rows if row['frame'] in selected):
            raise RuntimeError('Selected person mask touches crop edge; change projection and prompt.')
        save(folder/'tracking_report.json',dict(status='complete',method='SAM 2.1 Hiera Small video propagation',
             prompt=prompt,checkpoint_sha256=checkpoint_sha,rows=rows,dilation_radius_pixels=radius,
             close_kernel_pixels=close,elapsed_seconds=time.perf_counter()-started))
        del state
        torch.cuda.empty_cache()
    del tracker
    gc.collect()
    torch.cuda.empty_cache()
    for clip in clips:
        folder=Path(clip['directory'])
        with (folder/'propainter.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'propainter_stage.py'),str(output),clip['id']],
                           stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    main()
