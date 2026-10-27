"""Transfer uploaded RGB captures to an existing authenticated GPU relay."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import time
import uuid
import zipfile

from tools.gaussian_change.remote import Relay
from .core import read, write


def execute(bundle, output, runtime):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    relay = Relay(runtime['remote_session'])
    remote_root = PurePosixPath(runtime['remote_root'])
    if not remote_root.is_absolute() or '..' in remote_root.parts:
        raise ValueError('remote_root must be absolute')
    python, quote = runtime['python'], shlex.quote
    receipt = output/'remote.local.json'
    if receipt.is_file():
        process = read(receipt); remote = process['directory']
    else:
        remote = str(remote_root / ('rgb-'+uuid.uuid4().hex))
        write(output/'status.json', dict(state='running', stage='transferring'))
        code = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(code/'__init__.py', 'tools/__init__.py')
            for name in ['__init__.py', 'registration.py']:
                z.write(code/'gaussian_change'/name, 'tools/gaussian_change/'+name)
            for file in (code/'mesh_simsac').iterdir():
                if file.suffix in ['.py', '.js', '.html', '.css']:
                    z.write(file, 'tools/mesh_simsac/'+file.name)
            package = Path(runtime['three_package'])
            for name in ['build/three.module.js', 'build/three.core.js', 'examples/jsm/controls/OrbitControls.js', 'LICENSE']:
                z.write(package/name, 'three/'+name)
        remote_runtime = {k: v for k, v in runtime.items() if k in ['simsac_source', 'simsac_checkpoint', 'person_checkpoint', 'dependency_paths', 'model_hashes', 'device']}
        remote_runtime['three_package'] = remote+'/code/three'
        write(output/'runtime.local.json', remote_runtime)
        relay.call(command='mkdir -p '+quote(remote+'/code'))
        relay.call(upload=[[str(output/'source.zip'), remote+'/source.zip'], [str(Path(bundle).resolve()), remote+'/input.zip'],
                           [str(output/'runtime.local.json'), remote+'/runtime.json']])
        command = [python, '-m', 'tools.mesh_simsac.worker', '--manifest', remote+'/inputs/manifest.json',
                   '--output', remote+'/result', '--runtime', remote+'/runtime.json']
        script = ('import zipfile,subprocess,os,json;'
                  f'zipfile.ZipFile({remote+"/source.zip"!r}).extractall({remote+"/code"!r});'
                  f'zipfile.ZipFile({remote+"/input.zip"!r}).extractall({remote+"/inputs"!r});'
                  f'env=os.environ.copy();env.update({runtime.get("environment", {})!r});'
                  f'log=open({remote+"/worker.log"!r},"w");'
                  f'p=subprocess.Popen({command!r},cwd={remote+"/code"!r},env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);'
                  'print(json.dumps({"pid":p.pid}))')
        process = json.loads(relay.call(command=quote(python)+' -c '+quote(script)))
        write(output/'remote.local.json', dict(directory=remote, **process))

    deadline = time.monotonic()+runtime.get('timeout_seconds', 14400)
    while True:
        script = (f'from pathlib import Path;import json;p=Path({remote+"/result/status.json"!r});'
                  f'proc=Path("/proc/{process["pid"]}/stat");'
                  'alive=proc.exists() and proc.read_text().split()[2]!="Z";'
                  's=json.loads(p.read_text()) if p.exists() else {"state":"running","stage":"starting"};'
                  f'log=Path({remote+"/worker.log"!r});'
                  's=s if alive or s.get("state") in ["complete","failed"] else {"state":"failed","message":log.read_text()[-1500:]};'
                  'print(json.dumps(s))')
        try:
            state = json.loads(relay.call(command=quote(python)+' -c '+quote(script)))
        except ConnectionError:
            if time.monotonic() > deadline:
                raise TimeoutError('서버 재연결 대기 시간이 초과됐습니다. remote.local.json에 기존 GPU 작업이 기록되어 있습니다.')
            write(output/'status.json', dict(state='running', stage='waiting_for_connection'))
            time.sleep(4)
            continue
        if state.get('state') == 'failed':
            write(output/'status.json', state)
            raise RuntimeError(state.get('message', 'GPU worker failed'))
        if state.get('state') == 'complete':
            break
        write(output/'status.json', state)
        if time.monotonic() > deadline:
            raise TimeoutError('GPU run timed out; remote.local.json records the worker')
        time.sleep(4)
    write(output/'status.json', dict(state='running', stage='receiving_results'))
    # Download this run's public viewer only; private inputs/geometry arrays stay on GPU.
    script = (f'from pathlib import Path;import zipfile;p=Path({remote+"/result/detection"!r});'
              f'z=zipfile.ZipFile({remote+"/viewer.zip"!r},"w",zipfile.ZIP_DEFLATED,compresslevel=1);'
              'files=[p/n for n in ["index.html","events-viewer.js","events-viewer.css","viewer-data.json","validation.json"]];'
              'files += [f for folder in ["events","preview"] for f in (p/folder).rglob("*") if f.is_file()];'
              'files += [p/"photo-presentation.js"] if (p/"photo-presentation.js").is_file() else [];'
              'files += [p/"results"/n for n in ["events.json","model.json"]];'
              '[z.write(f,str(f.relative_to(p))) for f in files];z.close()')
    relay.call(command=quote(python)+' -c '+quote(script))
    relay.call(get=[[remote+'/viewer.zip', str(output/'viewer.zip')]])
    target = output/'detection'; target.mkdir(exist_ok=True)
    with zipfile.ZipFile(output/'viewer.zip') as z:
        for entry in z.infolist():
            path = (target/entry.filename).resolve()
            if not path.is_relative_to(target) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe result archive')
        z.extractall(target)
    write(output/'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.bundle, args.output, read(args.runtime))
    except Exception as error:
        write(args.output/'status.json', dict(state='failed', stage='failed', message=str(error)))
        raise
