"""Official ProPainter inference, with a read-only propagation diagnostic hook."""
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from pipeline import read,save


def main():
    output=Path(sys.argv[1]).resolve()
    clip_id=sys.argv[2]
    cfg,plan=read(output/'effective_config.json'),read(output/'plan.json')
    clip=next(item for item in plan['clips'] if item['id']==clip_id)
    folder=Path(clip['directory'])
    repo=Path(cfg['propainter']['repository'])
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=cfg['propainter']['official_commit']:
        raise RuntimeError('ProPainter source version differs from the configured version.')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    command=['inference_propainter.py','--video',str(folder/'frames'),'--mask',str(folder/'masks'),
             '--output',str(folder/'propainter_results'),'--width',str(clip['projection']['size'][0]),
             '--height',str(clip['projection']['size'][1])]
    for key in ('mask_dilation','ref_stride','neighbor_length','subvideo_length','raft_iter','save_fps'):
        command.extend(['--'+key,str(cfg['propainter'][key])])
    command.append('--save_frames')
    if cfg['propainter']['fp16']:
        command.append('--fp16')
    for name in ('propagation_frames','propagation_remaining_masks'):
        (folder/name).mkdir(exist_ok=True)

    def inspect_propagation(updated_frames,updated_masks,masks_dilated):
        rgb=((updated_frames[0].detach().float().cpu().permute(0,2,3,1).numpy()+1)/2*255).clip(0,255).astype('uint8')
        remain=updated_masks[0,:,0].detach().float().cpu().numpy()>.5
        original=masks_dilated[0,:,0].detach().float().cpu().numpy()>.5
        rows=[]
        for index,(image,mask,orig) in enumerate(zip(rgb,remain,original)):
            cv2.imwrite(str(folder/'propagation_frames'/f'{index:05d}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(folder/'propagation_remaining_masks'/f'{index:05d}.png'),mask.astype('uint8')*255)
            rows.append(dict(frame=index,masked_pixels=int(orig.sum()),unfilled_after_propagation=int((mask&orig).sum())))
        save(folder/'propagation_report.json',dict(rows=rows,note='Propagation support only; not a reconstruction accuracy score.'))

    source=(repo/'inference_propainter.py').read_text()
    anchor='    # ---- feature propagation + transformer ----'
    if source.count(anchor)!=1:
        raise RuntimeError('Official code hook location changed.')
    source=source.replace(anchor,'    inspect_propagation(updated_frames, updated_masks, masks_dilated)\n'+anchor)
    started=time.perf_counter()
    report=dict(status='running',command=command,official_commit=commit,trained_or_finetuned=False,started_unix=time.time())
    save(folder/'propainter_report.json',report)
    sys.argv=command
    try:
        exec(compile(source,str(repo/'inference_propainter.py'),'exec'),
             dict(__name__='__main__',__file__=str(repo/'inference_propainter.py'),inspect_propagation=inspect_propagation))
    except Exception as error:
        report.update(status='failed',error=str(error),elapsed_seconds=time.perf_counter()-started)
        save(folder/'propainter_report.json',report)
        raise
    report.update(status='complete',elapsed_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    save(folder/'propainter_report.json',report)


if __name__=='__main__':
    main()
