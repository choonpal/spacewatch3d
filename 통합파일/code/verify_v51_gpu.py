#!/usr/bin/env python3
"""Server-only, sequential verification under the chat's single-GPU lease."""
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
from spacewatch_pipeline.io import ROOT, read, save


def inputs(out):
    import torch
    from spacewatch_pipeline.gaussian_inputs import CudaViewLoader
    from spacewatch_pipeline.semantic_aggregation import aggregate
    from spacewatch_pipeline.florence import aggregate_votes_reference
    data=read('/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_20261025/results.json')
    dataset=Path(data['dorm2.gs_prepare']['dataset']); meta=read(dataset/'views.json')
    loader=CudaViewLoader(dataset,meta['frames'],meta['size'],1024**3)
    import cv2
    indices=np.random.default_rng(42).integers(0,len(meta['frames']),24).tolist()
    for i,index in enumerate(indices):
        target,valid=loader.get(index)
        if i+1<len(indices):loader.prefetch(indices[i+1])
        view=meta['frames'][index]
        expected=torch.tensor(cv2.cvtColor(cv2.imread(str(dataset/view['image'])),cv2.COLOR_BGR2RGB),device='cuda').float()/255
        mask=torch.tensor(cv2.imread(str(dataset/view['mask']),0)>254,device='cuda')
        torch.testing.assert_close(target,expected,rtol=0,atol=0)
        torch.testing.assert_close(valid,mask,rtol=0,atol=0)
        loader.stage_pending()
    loader.close();report=dict(input_pixel_parity=True,input_cache=loader.statistics(),aggregation=[])
    # Explicit winners, within-frame ties, across-frame ties, and large instance
    # IDs exercise cases that uniform random votes often reject as ambiguous.
    rows=np.array([(0,0,7)]*3+[(0,1,7)]*2+[(0,2,7)]+[(0,2,8)]+
                  [(1,0,3),(1,0,4),(1,1,3),(1,2,4)]+
                  [(2,0,350),(2,1,350),(2,2,351),(2,3,350)],dtype=np.int64)
    for unknown in [0,255]:
        for minimum in [1,2,3]:
            expected=aggregate_votes_reference(*rows.T,4,unknown,minimum)
            actual=aggregate(*rows.T,4,unknown,minimum,backend='cuda')
            for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
    report['aggregation_targeted_cases_passed']=6
    rng=np.random.default_rng(23)
    aggregate(np.array([0]),np.array([0]),np.array([0]),1,255,backend='cuda')
    for n in [1000,100000,1000000]:
        face=rng.integers(0,max(1,n//100),n);frame=rng.integers(0,30,n);label=rng.integers(0,17,n)
        row=dict(pixels=n);reference=None
        for name,fn in [('v5',aggregate_votes_reference),('numpy',aggregate),('cuda',aggregate)]:
            torch.cuda.synchronize();start=time.perf_counter()
            answer=fn(face,frame,label,max(1,n//100),255,**({'backend':name} if name!='v5' else {}))
            torch.cuda.synchronize();row[name+'_seconds']=time.perf_counter()-start
            if reference is None:reference=answer
            else:
                for a,b in zip(answer,reference):np.testing.assert_array_equal(a,b)
        row['exact_parity']=True;report['aggregation'].append(row);print('AGGREGATION',row,flush=True)
    save(out/'inputs.json',report)


def panovggt(out):
    import torch
    source=Path('/workspace/dorm2_pointcloud_v2_20261017');sandbox=out/'pano_inputs';sandbox.mkdir(exist_ok=True)
    cfg=read(source/'config.json');cfg['physical_gpu']=int(os.environ['V51_PHYSICAL_GPU'])
    cfg['execution']['physical_gpu']=cfg['physical_gpu'];save(sandbox/'config.json',cfg)
    for name in ['input','masks']:
        if not (sandbox/name).exists():(sandbox/name).symlink_to(source/name,target_is_directory=True)
    os.environ['PANOVGGT_RUN_ROOT']=str(sandbox)
    sys.path.insert(0,str(ROOT/'vendor/panovggt_pipeline_v2/workers'))
    import engine
    torch.manual_seed(cfg['seed']);engine.load_model()
    cached=engine.model.aggregator.patch_embed;encoder=cached.encoder
    engine.model.aggregator.patch_embed=encoder
    engine.predict_precise(cfg['window_indices'][0],1)  # Warm kernels before timing.
    reference=[];old=[]
    for ids in cfg['window_indices'][:2]:
        values,info=engine.predict_precise(ids,1);reference.append(values);old.append(info)
    engine.model.aggregator.patch_embed=cached;cached.clear();new=[];errors=[]
    for ids,expected in zip(cfg['window_indices'][:2],reference):
        values,info=engine.predict_precise(ids,1);new.append(info);row={}
        for name in ['points','depths','poses']:
            delta=np.abs(values[name]-expected[name]);row[name+'_max_abs']=float(delta.max())
            # FP32 batch kernels can differ slightly after removing cached frames.
            np.testing.assert_allclose(values[name],expected[name],rtol=3e-4,atol=3e-4)
        row['changed_valid_pixels']=int(np.count_nonzero(values['valid']!=expected['valid']))
        if row['changed_valid_pixels']:raise AssertionError('DINO cache changed retained depth pixels')
        errors.append(row)
    save(out/'panovggt.json',dict(reference=old,optimized=new,errors=errors,cache=cached.statistics(),
         reference_forward_seconds=sum(x['forward_gpu_s'] for x in old),
         optimized_forward_seconds=sum(x['forward_gpu_s'] for x in new)))
    print('PANO_PARITY',errors,flush=True)


def gs(out):
    from spacewatch_pipeline.gaussian import gs_train
    data=read('/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_20261025/results.json')
    config=read(ROOT/'configs/dorm2.json');config['gaussian'].update(steps=100,decoded_cache_mib=1024)
    target=out/'gs_smoke/attempt-test';target.mkdir(parents=True,exist_ok=True)
    gs_train(dict(config=config,capture=dict(id='dorm2'),dependencies={'dorm2.gs_prepare':data['dorm2.gs_prepare']},output=str(target)))
    save(out/'gs.json',read(target/'result.json'))


def gs_reference(out):
    import torch
    spec=importlib.util.spec_from_file_location('spacewatch_pipeline.gaussian_reference',
        '/workspace/spacewatch_pipeline_v5/spacewatch_pipeline/gaussian.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=read('/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_20261025/results.json')
    config=read(ROOT/'configs/dorm2.json');config['gaussian'].update(steps=100,decoded_cache_mib=1024)
    target=out/'gs_reference/attempt-test';target.mkdir(parents=True,exist_ok=True)
    if not (target/'result.json').exists():
        module.gs_train(dict(config=config,capture=dict(id='dorm2'),dependencies={'dorm2.gs_prepare':data['dorm2.gs_prepare']},output=str(target)))
    old=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
    new=torch.load(out/'gs_smoke/attempt-test/best.pt',map_location='cpu',weights_only=False)
    repeat=out/'gs_reference_repeat/attempt-test';repeat.mkdir(parents=True,exist_ok=True)
    if not (repeat/'result.json').exists():
        module.gs_train(dict(config=config,capture=dict(id='dorm2'),dependencies={'dorm2.gs_prepare':data['dorm2.gs_prepare']},output=str(repeat)))
    again=torch.load(repeat/'best.pt',map_location='cpu',weights_only=False)
    def difference(checkpoint):
        return {k:dict(max_abs=float((checkpoint['splats'][k]-v).abs().max()),
                       rms=float(((checkpoint['splats'][k]-v).double().square().mean()).sqrt())) for k,v in old['splats'].items()}
    changed=difference(new);repeat_error=difference(again)
    # gsplat backward uses floating-point atomic reductions. Quantify baseline
    # repeatability instead of claiming bitwise-identical trained parameters.
    within_repeatability=all(changed[k]['rms'] <= max(1e-5,5*repeat_error[k]['rms']) for k in changed)
    delta=abs(old['best_psnr']-new['best_psnr'])
    report=dict(reference=read(target/'result.json'),parameter_difference=changed,
         baseline_repeat_difference=repeat_error,psnr_difference=delta,
         strict_parameter_comparison_passed=False,within_baseline_repeatability=within_repeatability,
         acceptance='PSNR difference <= 0.02 dB and per-parameter RMS <= max(1e-5, 5x repeated-v5 RMS)')
    save(out/'gs_reference.json',report)
    if delta>.02 or not within_repeatability:raise AssertionError('3DGS exceeds measured baseline repeatability')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=['inputs','panovggt','gs','gs_reference']);parser.add_argument('--only',nargs='+',default=['inputs','panovggt','gs'])
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'configs/dorm2.json')
    if socket.gethostname()!=config['runtime']['allowed_hostname']:raise RuntimeError('GPU server only')
    if args.worker:
        globals()[args.worker](out);return
    from spacewatch_pipeline.gpu import lease
    from spacewatch_pipeline.engine import worker_environment
    for kind in args.only:
        with lease(thread_id=config['runtime']['gpu_thread_id']) as gpu:
            python,env=worker_environment(config,'gs_train',gpu)
            # Benchmark uses real training environment, not the reuse worker.
            python=config['runtime']['reconstruction_python']
            env['V51_PHYSICAL_GPU']=str(gpu['index'])
            save(out/(kind+'_gpu.json'),gpu)
            print('VERIFY',kind,'GPU',gpu['index'],flush=True)
            subprocess.run([python,str(Path(__file__).resolve()),'--output',str(out),'--worker',kind],env=env,check=True)


if __name__=='__main__':main()
