"""Start the bundled apps/autotour360 server without a desktop or an agent."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from .io import ROOT,read,save,digest,code_identity

_children={}

def health(url,bundle_id):
    try:
        with urllib.request.urlopen(url+'/api/pipeline',timeout=2) as stream:
            return json.load(stream).get('bundle_id')==bundle_id
    except (OSError,ValueError):return False

def launch(run,bundle,port=0):
    if not 0 <= port <= 65535:raise ValueError('Viewer port must be between 0 and 65535')
    run,bundle=Path(run).resolve(),Path(bundle).resolve()
    manifest=read(bundle/'manifest.json');state_path=run/'viewer-service.json'
    with (run/'.viewer-service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if state_path.exists():
            old=read(state_path)
            if old.get('bundle_id')==manifest['bundle_id'] and (not port or port==old.get('port')) and health(old['url'],manifest['bundle_id']):
                return old
            if health(old['url'],old.get('bundle_id')):
                stopped=stop(run)
                if stopped['termination_requested']:
                    deadline=time.monotonic()+5
                    while health(old['url'],old.get('bundle_id')) and time.monotonic()<deadline:time.sleep(.05)
        state_dir=run/'viewer-service';state_dir.mkdir(exist_ok=True)
        ready=state_dir/('ready-'+manifest['bundle_id'][:16]+'.json')
        if ready.exists():ready.unlink()
        command=[sys.executable,str(ROOT/'apps/autotour360/server.py'),'--host','127.0.0.1',
                 '--port',str(port),'--pipeline-dir',str(bundle),'--ready-file',str(ready),
                 '--data-dir',str(state_dir/'data'),'--output-dir',str(state_dir/'output')]
        if (manifest.get('changes') or {}).get('url')=='/index.html?mode=changes&source=dorm':
            command+=['--dorm-tour-dir',str(bundle/'changes')]
        with (state_dir/'server.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        _children[child.pid]=child
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if child.poll() is not None:
                raise RuntimeError('Viewer server exited; see '+str(state_dir/'server.log'))
            if ready.exists():
                info=read(ready)
                if health(info['url'],manifest['bundle_id']):
                    state=dict(pid=child.pid,url=info['url'],viewer_url=info['url']+'/pipeline.html',
                               port=info['port'],bundle_id=manifest['bundle_id'],bundle=str(bundle),
                               log=str(state_dir/'server.log'),run_root=str(run))
                    save(state_path,state);return state
            time.sleep(.1)
        child.terminate();child.wait(timeout=10)
        raise RuntimeError('Viewer did not become ready; see '+str(state_dir/'server.log'))

def stop(run):
    path=Path(run).resolve()/'viewer-service.json';state=read(path)
    proc=Path('/proc')/str(state['pid'])/'cmdline'
    command=proc.read_bytes().split(b'\0') if proc.exists() else []
    owned=(str(ROOT/'apps/autotour360/server.py').encode() in command and state['bundle'].encode() in command)
    if owned:
        os.kill(state['pid'],signal.SIGTERM)
        child=_children.pop(state['pid'],None)
        if child:child.wait(timeout=10)
    return dict(termination_requested=owned,pid=state['pid'])

def attach(run,port=0):
    from .engine import worker_environment,run_worker,validate
    run=Path(run).resolve();config=read(run/'config.json');validate(config)
    deps=read(run/'results.json')
    if 'publish' not in deps:raise ValueError('Finish publish before attaching a viewer')
    identifier=digest(dict(results=deps,code=code_identity()))
    out=run/'viewers'/identifier[:20]
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'result.json').exists():
        job=dict(config=config,action='viewer',dependencies=deps,output=str(out),run_root=str(run),
                 gpu=None,viewer_launch=False,bundle_id=identifier)
        save(out/'job.json',job)
        python,env=worker_environment(config,'viewer')
        run_worker([python,'-m','spacewatch_pipeline.worker',str(out/'job.json')],env,out/'worker.log',600)
    return launch(run,read(out/'result.json')['bundle'],port)
