"""Receipted DAG execution: one CPU worker plus one leased GPU worker."""
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from .io import ROOT, read, save, digest, record, verify, code_identity, now
from .gpu import lease
from . import __version__

_worker_control = threading.local()

def validate(config):
    from .planning_policy import effective_planning_config
    effective_planning_config(config)
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
        if c.get('planning_profile'):
            profile=c['planning_profile']
            if config.get('segmentation',{}).get('source')!='provided' or not c.get('source_run'):
                raise ValueError('A frozen planning profile requires provided labels and a bound source run')
            if not isinstance(profile,dict) or set(profile)!={'path','sha256'} or not profile['path']:
                raise ValueError('planning_profile requires path and sha256')
            if not re.fullmatch(r'[0-9a-f]{64}',profile['sha256']):raise ValueError('Invalid planning profile digest')
    gs = config.get('gaussian', {})
    if gs.get('steps', 30000) < 1:
        raise ValueError('gaussian.steps must be positive')
    if config.get('pipeline_version') != __version__:
        raise ValueError('Expected pipeline_version ' + __version__)
    if not config.get('size_segmentation',{}).get('assumptions'):
        raise ValueError('v5 needs an explicit size_segmentation.assumptions JSON')
    if not config['runtime'].get('gpu_thread_id'):
        raise ValueError('runtime.gpu_thread_id is required for one-GPU coordination')
    if config.get('semantic', {}).get('aggregation_backend', 'numpy') not in ('numpy','cuda'):
        raise ValueError('semantic.aggregation_backend must be numpy or cuda')
    if not isinstance(config['runtime'].get('overlap_cpu_gpu', True), bool):
        raise ValueError('runtime.overlap_cpu_gpu must be boolean')
    if gs.get('decoded_cache_mib', 2048) < 0:
        raise ValueError('gaussian.decoded_cache_mib must be nonnegative')
    if not config.get('planning',{}).get('independent_coverage_validation',True) or config.get('planning',{}).get('validation_runs',2)<2:
        raise ValueError('v5 requires at least two independent coverage validation runs')
    mode=config.get('segmentation',{}).get('source')
    if mode=='florence':
        if not config.get('discovery',{}).get('enabled') or set(config['discovery'].get('models',{})) != {'florence'}:
            raise ValueError('Florence mode requires exactly the Florence-2 discovery model')
        if any(c.get(k) for c in captures for k in ['predictions','semantic_arrays','planning_scene']):
            raise ValueError('Florence mode cannot silently use old semantic predictions/scenes')
    elif mode=='provided':
        if not config['segmentation'].get('vocabulary') or config.get('discovery',{}).get('enabled'):
            raise ValueError('Provided mode needs an explicit vocabulary and discovery disabled')
        for c in captures:
            if not all(c.get(k) for k in ['semantic_arrays','semantic_frame_id','semantic_coordinates_sha256']):
                raise ValueError('Provided arrays need semantic_arrays, semantic_frame_id and semantic_coordinates_sha256')
            if c.get('planning_scene'):
                raise ValueError('v5 builds a scene from the new candidate IDs; old planning_scene is not accepted')
    else:
        raise ValueError('segmentation.source must be provided or florence')
    if config.get('paper_integration',{}).get('enabled'):
        from .paper_pipeline import validate as validate_paper
        validate_paper(config)

def stages(config):
    rows=[]
    for c in config['captures']:
        prefix=c['id'];previous=[]
        for action in (['reconstruct','mesh'] if config['segmentation']['source']=='provided' else ['reconstruct','mesh','prepare','infer','observe','refine_prepare','refine_infer','objects']):
            name=prefix+'.'+action
            rows.append(dict(name=name,action=action,capture=prefix,dependencies=previous.copy(),
                gpu=(action=='reconstruct' and not c.get('source_run')) or action in ['infer','refine_infer']))
            previous.append(name)
    rows.append(dict(name='discovery',action='discovery',dependencies=[r['name'] for r in rows],gpu=False))
    for c in config['captures']:
        prefix=c['id'];previous=[prefix+'.reconstruct',prefix+'.mesh','discovery']
        if config['segmentation']['source']=='florence':previous.append(prefix+'.objects')
        for action in ['semantic','instances','scene']:
            name=prefix+'.'+action
            gpu = (action == 'semantic' and config['segmentation']['source'] == 'florence'
                   and config.get('semantic', {}).get('aggregation_backend', 'numpy') == 'cuda')
            rows.append(dict(name=name,action=action,capture=prefix,dependencies=previous.copy(),gpu=gpu))
            previous.append(name)
    if len(config['captures'])==2 and config.get('alignment',{}).get('enabled',True):
        deps=[c['id']+'.'+a for c in config['captures'] for a in ['reconstruct','mesh','semantic','scene']]
        rows.append(dict(name='pair.align',action='align',dependencies=deps,gpu=False))
        rows.append(dict(name='pair.changes',action='changes',dependencies=deps+['pair.align'],gpu=False))
    for c in config['captures']:
        prefix=c['id'];deps=[prefix+'.'+a for a in ['reconstruct','semantic','scene']]
        for action,extra,gpu in [('plan',[],False),('gs_prepare',[],False),
                ('gs_train',['gs_prepare'],not bool(config.get('reuse_gaussian_run'))),
                ('render',['plan','gs_prepare','gs_train'],True)]:
            rows.append(dict(name=prefix+'.'+action,action=action,capture=prefix,
                dependencies=deps+[prefix+'.'+a for a in extra],gpu=gpu))
    for action in ['validate','publish']+(['viewer'] if config.get('viewer',{}).get('enabled',True) else []):
        rows.append(dict(name=action,action=action,dependencies=[r['name'] for r in rows],gpu=False))
    if config.get('media', {}).get('mode') == 'original_erp':
        from .capture519 import adapt_stages
        rows=adapt_stages(rows, config)
        if config.get('paper_integration',{}).get('enabled'):
            from .paper_pipeline import adapt_stages as paper_stages
            rows=paper_stages(rows,config)
        return rows
    return rows

def worker_environment(config, action, gpu=None):
    runtime = config['runtime']
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor/spacewatch3d_src'),
               CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'],
               OMP_NUM_THREADS=str(runtime.get('cpu_threads',4)), OPENBLAS_NUM_THREADS=str(runtime.get('cpu_threads',4)),
               MKL_NUM_THREADS=str(runtime.get('cpu_threads',4)), PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] += ':'+str(ROOT/'vendor/discovery')
    if config.get('paper_integration',{}).get('enabled'):
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                   WANDB_DISABLED='true')
    env['CAPSTONE_GPU_THREAD_ID']=runtime.get('gpu_thread_id') or env.get('CAPSTONE_GPU_THREAD_ID') or env.get('CODEX_THREAD_ID','')
    env['LD_LIBRARY_PATH']=runtime.get('geometry_library_path','/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    python = runtime['geometry_python']
    if action in ['reconstruct','gs_prepare','gs_train','render']:
        python = runtime['reconstruction_python']
    if action in ['prepare','refine_prepare','infer','refine_infer']:
        python = runtime['florence_python']
    if action == 'plan':
        python = runtime['planning_python']
    if action == 'trajectory':
        python = runtime['geometry_python']
    if action == 'quality':
        python = runtime['quality_python']
    if action == 'render' and config.get('media', {}).get('mode') == 'original_erp':
        python = runtime['reconstruction_python']
    if action == 'rgb_infer':
        python=runtime['change_python']
        env['TORCH_HOME']=runtime['change_torch_home']
    if action in ['gs_train','render'] and config.get('media', {}).get('mode') != 'original_erp':
        base = Path(runtime['gsplat_root'])
        env.update(CUDA_HOME=runtime.get('cuda_home','/usr/local/cuda-11.8'),
                   TORCH_CUDA_ARCH_LIST='6.1', MAX_JOBS='2',
                   TORCH_EXTENSIONS_DIR=str(base/'cache/torch_extensions'))
        env['PYTHONPATH'] += ':'+str(base/'python_deps')+':'+str(base/'gsplat')
        env['PATH'] = str(base/'python_deps/bin')+':'+env['CUDA_HOME']+'/bin:'+env['PATH']
        env['LD_LIBRARY_PATH'] = env['CUDA_HOME']+'/lib64:'+env.get('LD_LIBRARY_PATH','')
    if action in ['gs_prepare','gs_train'] and config.get('reuse_gaussian_run'):
        python=runtime['geometry_python']
    return python, env

def run_worker(command, env, log, timeout):
    with Path(log).open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while True:
                cancel = getattr(_worker_control, 'cancel', None)
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('Cancelled after another pipeline stage failed')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = child.wait(timeout=min(1., remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
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


def stage_task(spec, config, output, signature, results, receipts, cancel, events):
    """Only this task writes its attempt/receipt; the coordinator owns run state."""
    _worker_control.cancel = cancel
    name = spec['name']; base = output/'stages'/name
    base.mkdir(parents=True, exist_ok=True)
    token = digest([signature, name, {k: receipts[k]['receipt_digest'] for k in spec['dependencies']}])
    complete = base/'complete.json'
    if complete.exists():
        receipt = read(complete)
        if receipt['identity'] != token:
            raise ValueError(f'Dependency mismatch: {name}')
        for artifact in receipt['artifacts']:
            verify(artifact)
        print('RESUMED', name, flush=True)
        return receipt, read(receipt['result_path']), True
    attempt = base/f'attempt-{time.time_ns()}'; attempt.mkdir()
    capture = next((c for c in config['captures'] if c['id'] == spec.get('capture')), None)
    started = time.monotonic(); started_at = now()
    if config.get('paper_integration',{}).get('enabled'):
        from .paper_pipeline import resource_guard
        resource_guard(config, spec['action'], attempt/'resource_start.json')
    with lease(config['runtime'].get('gpu_wait_seconds', 3600),
               thread_id=config['runtime'].get('gpu_thread_id'), cancel=cancel) if spec['gpu'] else contextlib.nullcontext(None) as gpu:
        if cancel.is_set():
            raise RuntimeError('Pipeline cancelled')
        effective = dict(config)
        if 'discovery' in results:
            effective['vocabulary'] = results['discovery']['vocabulary']
        payload = dict(config=effective, capture=capture, action=spec['action'], dependencies=results,
                       output=str(attempt), run_root=str(output), gpu=gpu)
        save(attempt/'job.json', payload)
        python, env = worker_environment(config, spec['action'], gpu)
        events.put((name, dict(status='running', gpu=gpu, log=str(attempt/'worker.log'))))
        compute_start = time.monotonic()
        print('RUNNING', name, flush=True)
        run_worker([python, '-m', 'spacewatch_pipeline.worker', str(attempt/'job.json')],
                   env, attempt/'worker.log', config['runtime'].get('stage_timeout_seconds', 86400))
        worker_seconds = time.monotonic() - compute_start
    result = read(attempt/'result.json')
    paths = {p.resolve() for p in attempt.rglob('*') if p.is_file() and p.name != 'worker.log'}
    paths.update(Path(p).resolve() for p in result.get('external_artifacts', []))
    hash_start = time.monotonic()
    artifacts = [record(p) for p in sorted(paths)]
    receipt = dict(identity=token, result_path=str(attempt/'result.json'), seconds=worker_seconds,
                   wall_seconds=time.monotonic()-started, hash_seconds=time.monotonic()-hash_start,
                   wait_seconds=compute_start-started, started_at=started_at,
                   artifacts=artifacts, completed_at=now())
    receipt['receipt_digest'] = digest(receipt)
    save(complete, receipt)
    return receipt, result, False


def run_dag(specs, task, commit, progress, *, overlap=True):
    """Bounded scheduler. Cancellation drains children before releasing run lock."""
    pending = list(specs); completed = set(); running = {}
    cancel = threading.Event(); events = queue.SimpleQueue()
    with ThreadPoolExecutor(max_workers=2 if overlap else 1) as pool:
        try:
            while pending or running:
                busy = {spec['gpu'] for spec in running.values()}
                ready = [s for s in pending if set(s['dependencies']) <= completed]
                # Prepare training inputs first so training overlaps CPU planning.
                ready.sort(key=lambda s: s['action'] != 'gs_prepare')
                for spec in ready:
                    if (spec['gpu'] in busy) or (running and not overlap):
                        continue
                    progress(spec['name'], dict(status='waiting_gpu' if spec['gpu'] else 'running'))
                    running[pool.submit(task, spec, cancel, events)] = spec
                    busy.add(spec['gpu']); pending.remove(spec)
                if not running:
                    raise ValueError('Stage graph has missing dependencies or a cycle')
                done, _ = wait(running, timeout=.2, return_when=FIRST_COMPLETED)
                while not events.empty():
                    name, update = events.get(); progress(name, update)
                # Stable commit order, including simultaneous completions.
                for future in sorted(done, key=lambda f: next(i for i,s in enumerate(specs) if s['name'] == running[f]['name'])):
                    spec = running.pop(future)
                    value = future.result()
                    commit(spec, value); completed.add(spec['name'])
        except BaseException:
            cancel.set()
            for future in running:
                future.cancel()
            raise

def execute(config_path, output, stop_after=None):
    config = read(config_path)
    validate(config)
    if stop_after and stop_after not in {s['name'] for s in stages(config)}:
        raise ValueError('Unknown --stop-after stage: '+stop_after)
    import socket
    if socket.gethostname()!=config['runtime']['allowed_hostname']:
        raise RuntimeError('Run the pipeline on the configured GPU server')
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
        started = time.monotonic()
        specs = stages(config)
        if stop_after:
            specs = specs[:next(i for i,s in enumerate(specs) if s['name'] == stop_after)+1]
        def progress(name, update):
            state['stages'].setdefault(name, {}).update(update)
            state.update(stage=name, updated_at=now())
            state['active_stages'] = [n for n,r in state['stages'].items() if r['status'] != 'complete']
            save(output/'status.json', state)
        def task(spec, cancel, events):
            # Dependency-only payload makes cache identities independent of timing.
            deps = {k: results[k] for k in spec['dependencies']}
            recs = {k: receipts[k] for k in spec['dependencies']}
            # Vocabulary is an implicit transitive input for later capture stages.
            if 'discovery' in results:
                deps['discovery'] = results['discovery']
            return stage_task(spec, config, output, signature, deps, recs, cancel, events)
        def commit(spec, value):
            nonlocal live_viewer
            name = spec['name']; receipt, result, resumed = value
            receipts[name] = receipt; results[name] = result
            state['stages'][name] = dict(status='complete', resumed=resumed,
                seconds=receipt['seconds'], wall_seconds=receipt.get('wall_seconds'),
                started_at=receipt.get('started_at'), completed_at=receipt.get('completed_at'))
            if name == 'viewer':
                from .viewer_service import launch
                live_viewer = launch(output, result['bundle'], config.get('viewer', {}).get('port', 0))
            save(output/'results.json', results)
            progress(name, {})
        try:
            run_dag(specs, task, commit, progress, overlap=config['runtime'].get('overlap_cpu_gpu', True))
            if stop_after:
                state.update(status='stopped_after_stage', stage=stop_after)
                return
            state.update(status='complete',completed_at=now(),gpu=None,summary=results['publish'],
                         viewer={**results['viewer'],'service':live_viewer} if live_viewer else None)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         error=str(exc),traceback=traceback.format_exc(),updated_at=now(),gpu=None)
            raise
        finally:
            state['wall_seconds'] = time.monotonic() - started
            save(output/'status.json',state)
