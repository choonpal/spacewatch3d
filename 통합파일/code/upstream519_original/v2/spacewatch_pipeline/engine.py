"""Receipted sequential pipeline, safe restart and subprocess isolation."""
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .io import ROOT, read, save, digest, record, verify, code_identity, now
from .gpu import lease

def validate(config):
    if config.get('pipeline_version')!='2.0.0':
        raise ValueError('Use a v2 config with pipeline_version 2.0.0')
    if config.get('schema_version') != 1:
        raise ValueError('Expected schema_version 1')
    captures = config.get('captures', [])
    if not 1 <= len(captures) <= 2:
        raise ValueError('Supply one capture, or before/after captures in that order')
    import re
    ids = [c['id'] for c in captures]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('Capture IDs must be unique safe names')
    for c in captures:
        if bool(c.get('source_run')) == bool(c.get('reconstruction_config')):
            raise ValueError('Each capture needs exactly one source_run or reconstruction_config')
        if not c.get('coordinates'):
            raise ValueError('Explicit source-to-metric coordinates JSON is required; scale is never guessed')
    if config.get('alignment',{}).get('prepared_source'):
        raise ValueError('v1 regional alignment is bound to old geometry; v2 requires fresh geometry validation')
    for c in captures:
        if any(c.get(k) for k in ['mesh_arrays','semantic_arrays','planning_scene']):
            raise ValueError('v2 rebuilds geometry and planning scenes after semantic filtering; remove v1 mesh/scene caches')
    gs = config.get('gaussian', {})
    if gs.get('steps', 30000) < 1:
        raise ValueError('gaussian.steps must be positive')

def stages(config):
    rows = []
    for c in config['captures']:
        prefix = c['id']
        previous = []
        for action in ['inputs', 'infer', 'masks', 'reconstruct', 'filter', 'mesh', 'semantic', 'scene']:
            name = prefix + '.' + action
            rows.append(dict(name=name, action=action, capture=prefix, dependencies=previous.copy(),
                             gpu=(action == 'reconstruct' and not c.get('source_run')) or
                                 (action == 'infer' and not c.get('predictions')) or action == 'gs_train'))
            previous.append(name)
    if len(config['captures']) == 2:
        deps = [f'{c["id"]}.{a}' for c in config['captures'] for a in ['reconstruct','filter','mesh','semantic','scene']]
        rows.append(dict(name='pair.align',action='align',dependencies=deps,gpu=False))
        rows.append(dict(name='pair.changes',action='changes',dependencies=deps+['pair.align'],gpu=False))
    for c in config['captures']:
        prefix = c['id']
        deps = [f'{prefix}.{a}' for a in ['reconstruct','filter','semantic','scene']]
        if len(config['captures'])==2:
            rows.append(dict(name=prefix+'.priority',action='priority',capture=prefix,dependencies=deps+['pair.changes'],gpu=False))
            deps.append(prefix+'.priority')
        rows.append(dict(name=prefix+'.plan',action='plan',capture=prefix,dependencies=deps.copy(),gpu=False))
        rows.append(dict(name=prefix+'.gs_prepare',action='gs_prepare',capture=prefix,dependencies=deps.copy(),gpu=False))
        rows.append(dict(name=prefix+'.gs_train',action='gs_train',capture=prefix,
                         dependencies=deps+[prefix+'.gs_prepare'],gpu=True))
        rows.append(dict(name=prefix+'.render',action='render',capture=prefix,
                         dependencies=deps+[prefix+'.plan',prefix+'.gs_prepare',prefix+'.gs_train'],gpu=True))
    rows.append(dict(name='publish', action='publish', dependencies=[r['name'] for r in rows], gpu=False))
    if config.get('viewer',{}).get('enabled',True):
        rows.append(dict(name='viewer',action='viewer',dependencies=[r['name'] for r in rows],gpu=False))
    return rows

def worker_environment(config, action, gpu=None):
    runtime = config['runtime']
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor/spacewatch3d_src'),
               CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'],
               OMP_NUM_THREADS=str(runtime.get('cpu_threads',4)), OPENBLAS_NUM_THREADS=str(runtime.get('cpu_threads',4)),
               MKL_NUM_THREADS=str(runtime.get('cpu_threads',4)), PYTHONUNBUFFERED='1')
    env['LD_LIBRARY_PATH']=runtime.get('geometry_library_path','/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    python = runtime['geometry_python']
    if action in ['reconstruct','gs_prepare','gs_train','render']:
        python = runtime['reconstruction_python']
    if action == 'infer':
        python = runtime['ooops_python']
    if action == 'plan':
        python = runtime['planning_python']
    if action in ['gs_train','render']:
        base = Path(runtime['gsplat_root'])
        env.update(CUDA_HOME=runtime.get('cuda_home','/usr/local/cuda-11.8'),
                   TORCH_CUDA_ARCH_LIST='6.1', MAX_JOBS='2',
                   TORCH_EXTENSIONS_DIR=str(base/'cache/torch_extensions'))
        env['PYTHONPATH'] += ':'+str(base/'python_deps')+':'+str(base/'gsplat')
        env['PATH'] = str(base/'python_deps/bin')+':'+env['CUDA_HOME']+'/bin:'+env['PATH']
        env['LD_LIBRARY_PATH'] = env['CUDA_HOME']+'/lib64:'+env.get('LD_LIBRARY_PATH','')
    return python, env

def run_worker(command, env, log, timeout):
    with Path(log).open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'Worker exited {code}; see {log}')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise

def execute(config_path, output, stop_after=None):
    config = read(config_path)
    validate(config)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = dict(config=config,code=code_identity())
        signature = digest(identity)
        identity_path = output/'identity.json'
        if identity_path.exists():
            if read(identity_path)['signature'] != signature:
                raise ValueError('Code/config changed. Select a new output directory; previous results are preserved.')
        elif any(p.name not in ['.run.lock','launch.log'] for p in output.iterdir()):
            raise ValueError('Nonempty unowned output directory')
        else:
            save(identity_path,dict(signature=signature,**identity))
        save(output/'config.json',config)
        state = dict(status='running',pid=os.getpid(),started_at=now(),stages={})
        results, receipts = {}, {}
        live_viewer=None
        try:
            for spec in stages(config):
                name = spec['name']
                base = output/'stages'/name
                base.mkdir(parents=True,exist_ok=True)
                token = digest([signature,name,{k:receipts[k]['receipt_digest'] for k in spec['dependencies']}])
                complete = base/'complete.json'
                if complete.exists():
                    receipt = read(complete)
                    if receipt['identity'] != token:
                        raise ValueError(f'Dependency mismatch: {name}')
                    for artifact in receipt['artifacts']:
                        verify(artifact)
                    result = read(receipt['result_path'])
                    print('RESUMED',name,flush=True)
                    state['stages'][name] = dict(status='complete',resumed=True)
                else:
                    attempt = base/f'attempt-{time.time_ns()}'
                    attempt.mkdir()
                    state.update(stage=name,status='waiting_gpu' if spec['gpu'] else 'running',updated_at=now())
                    save(output/'status.json',state)
                    capture = next((c for c in config['captures'] if c['id']==spec.get('capture')),None)
                    with lease(config['runtime'].get('gpu_wait_seconds',3600)) if spec['gpu'] else contextlib.nullcontext(None) as gpu:
                        reuse_action=spec['action'] if config.get('reuse_geometry_run') and spec['action'] in ['inputs','infer','masks','reconstruct','filter','mesh','semantic','scene'] else None
                        payload = dict(config=config,capture=capture,action='reuse' if reuse_action else spec['action'],reuse_action=reuse_action,dependencies=results,
                                       output=str(attempt),run_root=str(output),gpu=gpu)
                        save(attempt/'job.json',payload)
                        python, env = worker_environment(config,spec['action'],gpu)
                        command = [python,'-m','spacewatch_pipeline.worker',str(attempt/'job.json')]
                        state.update(status='running',gpu=gpu,log=str(attempt/'worker.log'))
                        save(output/'status.json',state)
                        started=time.monotonic()
                        print('RUNNING',name,flush=True)
                        run_worker(command,env,attempt/'worker.log',config['runtime'].get('stage_timeout_seconds',86400))
                    result = read(attempt/'result.json')
                    paths = {p.resolve() for p in attempt.rglob('*') if p.is_file() and p.name!='worker.log'}
                    paths.update(Path(p).resolve() for p in result.get('external_artifacts',[]))
                    receipt=dict(identity=token,result_path=str(attempt/'result.json'),seconds=time.monotonic()-started,
                                 artifacts=[record(p) for p in sorted(paths)],completed_at=now())
                    receipt['receipt_digest']=digest(receipt)
                    save(complete,receipt)
                    state['stages'][name]=dict(status='complete',seconds=receipt['seconds'],resumed=False)
                receipts[name]=receipt
                results[name]=result
                if name=='viewer':
                    from .viewer_service import launch
                    live_viewer=launch(output,result['bundle'],config.get('viewer',{}).get('port',0))
                save(output/'results.json',results)
                save(output/'status.json',state)
                if stop_after==name:
                    state.update(status='stopped_after_stage',stage=name)
                    save(output/'status.json',state)
                    return
            state.update(status='complete',completed_at=now(),gpu=None,summary=results['publish'],
                         viewer={**results['viewer'],'service':live_viewer} if live_viewer else None)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         error=str(exc),traceback=traceback.format_exc(),updated_at=now(),gpu=None)
            raise
        finally:
            save(output/'status.json',state)
