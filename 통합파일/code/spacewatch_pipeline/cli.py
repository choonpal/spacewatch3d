import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from .io import ROOT,read,save
from .engine import execute,validate,stages,worker_environment

def doctor(config):
    validate(config);rows=[]
    import socket
    rows.append(dict(check='GPU server hostname',actual=socket.gethostname(),expected=config['runtime']['allowed_hostname'],ok=socket.gethostname()==config['runtime']['allowed_hostname']))
    checks={'reconstruct':['numpy','torch','cv2','plyfile'], 'mesh':['numpy','scipy','trimesh','shapely','plyfile'],
            'plan':['numpy','scipy','skimage','shapely','trimesh'], 'gs_train':['torch','gsplat','cv2'], 'render':['torch','gsplat','cv2']}
    if config['segmentation']['source']=='florence':checks.update(prepare=['cv2','numpy','PIL'],infer=['torch','transformers','PIL']);checks['mesh'].append('open3d')
    if len(config['captures'])==2 and config.get('alignment',{}).get('enabled',True):checks['align']=['open3d','scipy']
    if config.get('reuse_gaussian_run'):checks['gs_train']=['numpy']
    if config.get('media',{}).get('mode')=='original_erp':
        checks.pop('gs_train',None);checks['render']=['numpy','cv2']
    if config.get('paper_integration',{}).get('enabled'):
        checks['quality']=['av','cv2','numpy']
        checks['rgb_infer']=['torch','cv2','numpy','PIL']
    for action,modules in checks.items():
        python,env=worker_environment(config,action)
        code='import '+','.join(modules)+'; print("OK")'
        if action=='plan' and config.get('native_visibility',True):code+='; import sys; sys.path.insert(0,"vendor/native"); import native_executor'
        try:
            proc=subprocess.run([python,'-c',code],env=env,cwd=ROOT,text=True,capture_output=True,timeout=120)
            rows.append(dict(action=action,python=python,ok=proc.returncode==0,detail=(proc.stdout+proc.stderr)[-1200:]))
        except (OSError,subprocess.TimeoutExpired) as exc:rows.append(dict(action=action,ok=False,detail=str(exc)))
    paths=[config['segmentation']['vocabulary']] if config['segmentation']['source']=='provided' else []
    if config.get('reuse_gaussian_run') and config.get('media',{}).get('mode')!='original_erp':paths.append(str(Path(config['reuse_gaussian_run'])/'results.json'))
    if config.get('discovery',{}).get('enabled'):
        paths += [r['path'] for r in config['discovery']['models'].values()]
    for c in config['captures']:
        paths+=[c[k] for k in ['source_run','reconstruction_config','coordinates','mesh_arrays','predictions','semantic_arrays'] if c.get(k)]
    for p in paths:rows.append(dict(path=p,ok=Path(p).exists()))
    python,env=worker_environment(config,'mesh')
    code='import json,sys; from spacewatch_pipeline.preflight import inputs; inputs(json.load(sys.stdin)); print("Inputs, coordinates and size assumptions OK")'
    try:
        proc=subprocess.run([python,'-c',code],input=json.dumps(config),env=env,cwd=ROOT,text=True,capture_output=True,timeout=120)
        rows.append(dict(check='input contracts',ok=proc.returncode==0,detail=(proc.stdout+proc.stderr)[-2000:]))
    except (OSError,subprocess.TimeoutExpired) as exc:rows.append(dict(check='input contracts',ok=False,detail=str(exc)))
    if config.get('paper_integration',{}).get('enabled'):
        for capture in config['captures']:
            python,env=worker_environment(config,'plan')
            try:
                proc=subprocess.run([python,'-m','spacewatch_pipeline.paper_preflight',capture['id']],
                    input=json.dumps(config),env=env,cwd=ROOT,text=True,capture_output=True,timeout=30)
                rows.append(dict(check='native planning config '+capture['id'],ok=proc.returncode==0,
                    detail=(proc.stdout+proc.stderr)[-1800:]))
            except (OSError,subprocess.TimeoutExpired) as exc:
                rows.append(dict(check='native planning config '+capture['id'],ok=False,detail=str(exc)))
    result=dict(ok=all(r['ok'] for r in rows),checks=rows)
    print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result['ok'] else 2

def main(argv=None):
    p=argparse.ArgumentParser(description='SpaceWatch: reconstruction/semantic mesh → captured-frame viewpoints → original-RGB changes → patrol viewer')
    from . import __version__
    p.add_argument('--version',action='version',version='SpaceWatch '+__version__)
    commands=p.add_subparsers(dest='command',required=True)
    for name in ['run','start','plan','doctor']:
        q=commands.add_parser(name);q.add_argument('--config',type=Path,required=True)
        if name in ['run','start']:
            q.add_argument('--output',type=Path,required=True);q.add_argument('--stop-after')
    for name in ['status','stop']:
        q=commands.add_parser(name);q.add_argument('--output',type=Path,required=True)
    for name in ['viewer','viewer-stop']:
        q=commands.add_parser(name);q.add_argument('--output',type=Path,required=True)
        if name=='viewer':q.add_argument('--port',type=int,default=0)
    args=p.parse_args(argv)
    if args.command in ['viewer','viewer-stop']:
        from .viewer_service import attach,stop
        print(json.dumps(attach(args.output,args.port) if args.command=='viewer' else stop(args.output),indent=2,ensure_ascii=False));return 0
    if args.command=='doctor':return doctor(read(args.config))
    if args.command=='plan':
        c=read(args.config);validate(c);print(json.dumps(stages(c),indent=2));return 0
    if args.command=='run':
        def terminate(signum,frame):raise KeyboardInterrupt('Pipeline termination requested')
        signal.signal(signal.SIGTERM,terminate)
        execute(args.config,args.output,args.stop_after);return 0
    if args.command=='start':
        config=read(args.config);validate(config)
        import socket
        if socket.gethostname()!=config['runtime']['allowed_hostname']:raise ValueError('Run start on the configured GPU server; use connect_viewer.py on your PC')
        if args.stop_after and args.stop_after not in {s['name'] for s in stages(config)}:raise ValueError('Unknown --stop-after stage')
        args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
        # Start logs are outside the immutable run; only one launcher per output path.
        lock_path=args.output.parent/(args.output.name+'.launch.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=args.output/'status.json'
            if state.exists():
                old=read(state);pid=old.get('pid')
                if pid and Path(f'/proc/{pid}/cmdline').exists() and str(args.output).encode() in Path(f'/proc/{pid}/cmdline').read_bytes():
                    raise RuntimeError('Pipeline already running')
            command=[sys.executable,'-m','spacewatch_pipeline.cli','run','--config',str(args.config.resolve()),'--output',str(args.output)]
            if args.stop_after:command+=['--stop-after',args.stop_after]
            with (args.output/'launch.log').open('a') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(json.dumps(dict(pid=proc.pid,output=str(args.output),log=str(args.output/'launch.log'))));return 0
    state=read(args.output/'status.json');pid=state.get('pid');cmd=Path(f'/proc/{pid}/cmdline')
    running=bool(pid and cmd.exists() and str(args.output.resolve()).encode() in cmd.read_bytes())
    if args.command=='stop':
        if running:os.kill(pid,signal.SIGTERM)
        print(json.dumps(dict(termination_requested=running,pid=pid)));return 0
    state['process_alive']=running
    if not running and state['status'] in ['running','waiting_gpu']:state['status']='interrupted'
    print(json.dumps(state,indent=2,ensure_ascii=False));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,BlockingIOError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(2)
