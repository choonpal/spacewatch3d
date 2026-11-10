"""Sequential, restartable server workflow with a single-GPU lease."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import read,save,sha,digest
from .gpu import lease

STAGES=['prepare','discover','ground','refine','measure','segment','project','validate']
GPU_STAGES={'discover','ground','refine','segment'}


def source_identity(config):
    records={}
    base=Path(__file__).resolve().parents[1]
    for path in [base/name for name in ('geometry.py','core.py','common.py','filtered_source.py','chair_size_prior.py','mesh_instances.py')]+[Path(config['runtime']['pipeline_root'])/'spacewatch_pipeline/ooops.py']:
        records[str(path)]=sha(path)
    for capture in config['captures']:
        source=Path(capture['source_run'])
        paths=[source/name for name in ('config.json','prepared.npz','optimized.npz','input/manifest.json','result/camera_poses.json')]
        paths += [Path(capture['coordinates']),Path(capture['mesh'])]
        paths += sorted((source/'windows').glob('*.json'))
        for path in paths:records[str(path)]=sha(path)
        manifest=read(source/'input/manifest.json')
        for frame in manifest['frames']:
            path=source/'input'/frame['file']
            if sha(path)!=frame['sha256']:raise ValueError('Reconstruction image was modified: '+str(path))
    return records


def stage_files(stage,output):
    folders={'prepare':['prepared'],'discover':['discovery'],'ground':['grounded'],'refine':['prepared_refinement','grounded'],
             'measure':['instance_points'],'segment':['segmentation'],'project':['meshes'],'validate':[]}
    names={'prepare':['prepared.json'],'discover':['discovery_complete.json'],
           'ground':['classes.json','candidate_classes.json','grounding_summary.json','normalized_phrases.json'],
           'refine':['refined.json','refinement_summary.json'],
           'measure':['observations.json','instances.json','size_priors.json'],
           'segment':['segmentation_summary.json'],'project':['projection_summary.json'],
           'validate':['validation.json','spacewatch_config.json']}
    files=[p for name in folders[stage] for p in (output/name).rglob('*') if p.is_file()]
    files += [output/name for name in names[stage] if (output/name).exists()]
    return sorted(set(files))


def receipt_valid(receipt,identity,output):
    if receipt.get('identity')!=identity or not receipt.get('artifacts'):return False
    return all((output/path).is_file() and sha(output/path)==expected for path,expected in receipt['artifacts'].items())


def execute(config_path,output,stop_after=None):
    config_path=Path(config_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    config=read(config_path)
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parent
    source_hashes={p.name:sha(p) for p in source.glob('*.py')}
    sources=source_identity(config)
    identity=digest([config,source_hashes,sources])
    old=read(output/'run.json') if (output/'run.json').exists() else None
    if old and old['config_sha256']!=sha(config_path):
        raise ValueError('Output belongs to a different config; choose a new output directory')
    save(output/'run.json',dict(config_sha256=sha(config_path),config=config,source_hashes=source_hashes,input_hashes=sources,identity=identity))
    env=dict(os.environ,PYTHONPATH=str(source.parents[1]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MKL_NUM_THREADS='2',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1')
    upstream_changed=False
    for stage in STAGES:
        started=time.time(); receipt=output/f'{stage}_receipt.json'
        previous=read(receipt) if receipt.exists() else {}
        # Stage modules validate reusable frame artifacts independently.
        if upstream_changed or not receipt_valid(previous,identity,output):
            upstream_changed=True
            python=config['runtime']['ooops_python'] if stage=='segment' else config['runtime']['python']
            if stage in {'measure','project','validate'}:python=config['runtime']['geometry_python']
            command=[python,'-m',f'segmentation_2d3d.scene_discovery.{stage}',str(config_path),str(output)]
            log=output/f'{stage}.log'
            save(output/'status.json',dict(stage=stage,status='running',started=started))
            try:
                with log.open('a') as stream:
                    if stage in GPU_STAGES:
                        from .discover import reusable
                        if stage=='discover' and reusable(config,output):
                            subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
                        else:
                            save(output/'status.json',dict(stage=stage,status='waiting_for_gpu',started=started))
                            with lease(config) as gpu:
                                save(output/f'{stage}_gpu.json',gpu)
                                save(output/'status.json',dict(stage=stage,status='running',started=started))
                                subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=stream,stderr=subprocess.STDOUT,check=True)
                    else:
                        subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
            except BaseException:
                save(output/'status.json',dict(stage=stage,status='failed',log=str(log)))
                raise
            artifacts={str(p.relative_to(output)):sha(p) for p in stage_files(stage,output)}
            save(receipt,dict(identity=identity,seconds=time.time()-started,stage=stage,status='complete',artifacts=artifacts))
            print('STAGE_COMPLETE',stage,round(time.time()-started,2),flush=True)
        if stage==stop_after: break
    save(output/'status.json',dict(status='complete' if not stop_after or stop_after==STAGES[-1] else 'stopped',
                                  last_stage=stage))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stop-after',choices=STAGES)
    a=p.parse_args();execute(a.config,a.output,a.stop_after)


if __name__=='__main__':main()
