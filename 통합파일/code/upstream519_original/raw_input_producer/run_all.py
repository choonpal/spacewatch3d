"""Raw dorm4/dorm5 adapter: fresh OOOPS/PanoVGGT inputs, then unchanged v5."""
import os,sys,json,time,subprocess,contextlib,traceback,hashlib,fcntl
from pathlib import Path
BASE=Path('/workspace/dorm4_dorm5_v5_20261025');V5=BASE/'code/v5';V2=BASE/'code/v2';THREAD='01a1734c-0468-70f1-a15d-d03ed287abea'
sys.path.insert(0,str(V5))
from spacewatch_pipeline.gpu import lease
from spacewatch_pipeline.io import read,save,sha,record
PYREC='/workspace/panovggt_env/bin/python';PYGEO='/workspace/mesh_generator_v1/.venv/bin/python';PYOPS='/workspace/mtpano_ooops_20261023/env/bin/python'
state=dict(status='running',pid=os.getpid(),completed=[],started_at=time.time())
def status(stage):
 state.update(stage=stage,updated_at=time.time());save(BASE/'status.json',state);print('STAGE',stage,flush=True)
def execute(name,command,env=None,gpu=False):
 status(name);log=BASE/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
 with lease(86400,thread_id=THREAD) if gpu else contextlib.nullcontext(None) as dev:
  if dev:state['gpu']=dev;save(BASE/'status.json',state)
  with log.open('a') as stream:subprocess.run(command(dev) if callable(command) else command,env=env(dev) if callable(env) else env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 state.pop('gpu',None);state['completed'].append(name);save(BASE/'status.json',state)
def baseenv(root,dev=None):
 e=dict(os.environ);e.update(PYTHONPATH=str(root)+':'+str(root/'vendor/spacewatch3d_src'),CUDA_VISIBLE_DEVICES='' if dev is None else dev['uuid'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',CAPSTONE_GPU_THREAD_ID=THREAD)
 e['LD_LIBRARY_PATH']='/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu:'+e.get('LD_LIBRARY_PATH','');return e
def worker(cid,action,config,deps,root=V2,gpu=False):
 path=BASE/'prepared'/cid/action;receipt=path/'receipt.json'
 if receipt.exists():
  rec=read(receipt)
  for item in rec['artifacts']:assert sha(item['path'])==item['sha256'],item['path']
  return read(path/'result.json')
 path.mkdir(parents=True,exist_ok=True)
 capture=dict(id=cid,source_run=str(BASE/'reconstruction'/cid),coordinates=str(BASE/'coordinates'/f'{cid}.json'))
 job=dict(action=action,config=config,capture=capture,dependencies=deps,output=str(path),run_root=str(BASE/'prepared'))
 def cmd(dev):
  job['gpu']=dev;save(path/'job.json',job);return [PYOPS if action=='infer' else PYGEO,'-m','spacewatch_pipeline.worker',str(path/'job.json')]
 execute(cid+'.'+action,cmd,lambda dev:baseenv(root,dev),gpu)
 result=read(path/'result.json');paths=[path/'result.json']
 for key in ['arrays','geometry','coordinates','vocabulary']:
  if result.get(key):paths.append(Path(result[key]))
 if result.get('predictions'):paths.extend(sorted(Path(result['predictions']).glob('*.npz')));paths.append(Path(result['predictions'])/'complete.json')
 save(receipt,dict(artifacts=[record(p) for p in set(paths)],completed_at=time.time()))
 return result

def main():
 lock=(BASE/'.driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 configs=BASE/'configs';configs.mkdir(exist_ok=True)
 catalog=read(configs/'classes.json')
 for r in catalog:r.pop('evidence',None);r['selection']='Assistant-selected fixed vocabulary for new dorm4/dorm5 run; not evidence of presence'
 save(configs/'classes.json',catalog)
 spec=read(V5/'configs/dorm_size_assumptions.json');dims={3:[.15,.95,2.1],4:[.2,1.2,2.2],5:[.12,1.2,1.3],6:[.55,.6,.9],7:[.8,1.4,.75],8:[.4,.45,1.2],9:[.3,.35,1.0],10:[.2,.7,1.2],11:[.2,.25,.55],12:[.25,.25,.08],13:[.08,.4,.4],14:[.04,.3,.2]}
 priors=[dict(class_id=r['id'],name=r['name'],name_ko=r['name_ko'],dimensions=dims[r['id']],references=[],known_per_physical_instance=False,evidence='user-authorized hypothetical dimensions, not measurements',assumption_ko='새 영상용 임의 대표 크기. 순서: 수평 짧은 변, 수평 긴 변, 높이.') for r in catalog if r['id'] in dims]
 spec.update(priors=priors,units='estimated meters based on assumed 2.7m ceiling height; unverified metric scale',metric_scale_verified=False,per_instance_size_measured=False,prior_type='Assistant-selected hypotheses for dorm4 and dorm5; same dimensions in both captures');save(configs/'sizes.json',spec)
 cfg=read(V2/'configs/dorm2_dorm3.json');cfg['runtime']['gpu_thread_id']=THREAD;cfg['vocabulary']=str(configs/'classes.json');cfg['mesh'].pop('protected_voxel_ratio',None)
 captures=[]
 for cid in ['dorm4','dorm5']:
  inp=BASE/'inputs'/cid
  if not (inp/'manifest.json').exists():execute(cid+'.extract',[PYREC,str(BASE/'prepare_new.py'),'extract',cid],baseenv(V5))
  m=read(inp/'manifest.json');deps={cid+'.inputs':dict(manifest=str(inp/'manifest.json'),images=[str(inp/r['file']) for r in m['frames']])}
  infer=worker(cid,'infer',cfg,deps,gpu=True);deps[cid+'.infer']=infer
  if not (inp/'assets.json').exists():execute(cid+'.masks',[PYGEO,str(BASE/'prepare_new.py'),'masks',cid,infer['predictions']],baseenv(V5))
  src=BASE/'reconstruction'/cid
  if not (src/'status.json').exists() or read(src/'status.json')['status']!='complete':
   def command(dev):
    rc=read(configs/f'{cid}_reconstruction.json');rc['execution']['physical_gpu']=dev['index'];save(configs/f'{cid}_reconstruction.json',rc)
    return [PYREC,str(V5/'vendor/panovggt_pipeline_v2/run.py'),'--config',str(configs/f'{cid}_reconstruction.json'),'--output',str(src)]
   execute(cid+'.reconstruction',command,lambda dev:baseenv(V5,dev),gpu=True)
  coord=BASE/'coordinates'/f'{cid}.json'
  if not coord.exists():execute(cid+'.coordinates',[PYGEO,str(BASE/'prepare_new.py'),'coordinates',cid,str(src),infer['predictions']],baseenv(V2))
  for action in ['reconstruct','mesh','semantic']:deps[cid+'.'+action]=worker(cid,action,cfg,deps)
  captures.append(dict(id=cid,source_run=str(src),coordinates=str(coord),video=str(BASE/'source'/f'{cid}.MP4'),semantic_arrays=deps[cid+'.semantic']['arrays'],semantic_frame_id=cid+'_mesh',semantic_coordinates_sha256=sha(coord)))
 final=read(V5/'configs/dorm2_dorm3.json');final['captures']=captures;final['runtime']['gpu_thread_id']=THREAD;final['runtime']['gpu_wait_seconds']=86400;final.pop('reuse_gaussian_run',None);final['segmentation']['vocabulary']=str(configs/'classes.json');final['size_segmentation']['assumptions']=str(configs/'sizes.json');final['scene_id']='dorm4_dorm5';save(configs/'dorm4_dorm5_v5.json',final)
 run=Path('/workspace/spacewatch_runs/dorm4_dorm5_pipeline_v5_20261025')
 execute('v5.doctor',[str(V5/'run.sh'),'doctor','--config',str(configs/'dorm4_dorm5_v5.json')],baseenv(V5))
 execute('v5.pipeline',[str(V5/'run.sh'),'run','--config',str(configs/'dorm4_dorm5_v5.json'),'--output',str(run)],baseenv(V5))
 execute('v5.export',['python3',str(V5/'export_results.py'),'--run',str(run),'--output',str(BASE/'export'),'--zip'],baseenv(V5))
 state.update(status='complete',run=str(run),export=str(BASE/'export'),completed_at=time.time());save(BASE/'status.json',state)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  state.update(status='failed',error=str(e),traceback=traceback.format_exc(),updated_at=time.time());save(BASE/'status.json',state);raise
