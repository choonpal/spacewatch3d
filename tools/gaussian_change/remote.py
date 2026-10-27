"""Run the same CLI through an existing authenticated SSH file-queue relay.

Credentials are managed by the host's relay, never by this processing package.
Both server-resident inputs and local input transfer are supported explicitly.
"""
from pathlib import Path, PurePosixPath
import json
import shlex
import time
import uuid
import zipfile
from tools.mesh_simsac.core import read, write


class Relay:
    def __init__(self, folder):
        self.folder = Path(folder)

    def call(self, **request):
        state = read(self.folder/'status.json')
        if state.get('state') != 'connected' or time.time()-state.get('heartbeat_unix', 0) > 90:
            raise ConnectionError('Authenticated SSH relay unavailable')
        key = 'mesh-gaussian-change-'+uuid.uuid4().hex+'.json'
        write(self.folder/'inbox'/key, request)
        result = self.folder/'outbox'/key
        deadline = time.monotonic()+600
        while not result.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('SSH relay request timed out')
            time.sleep(.5)
        result = read(result)
        if not result.get('ok') or result.get('exit_code', 0):
            raise RuntimeError(str(result.get('error', result.get('output')))[-3000:])
        return result.get('output', '')


def run_remote(manifest, output, runtime):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new/empty local output directory')
    output.mkdir(parents=True, exist_ok=True)
    root = PurePosixPath(runtime['remote_root'])
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('remote_root must be an absolute server directory')
    remote = str(root/('run-'+uuid.uuid4().hex))
    relay = Relay(runtime['remote_session'])
    python = runtime['python']
    quote = shlex.quote
    local_doc = read(manifest)
    uploads = []
    location = runtime.get('input_location', 'server')
    if location not in ['local', 'server']:
        raise ValueError('input_location must be local or server')
    if location == 'local':
        for phase in ['before', 'after']:
            for field in ['mesh', 'gaussian', 'cameras'] + (['instance_observations'] if local_doc[phase].get('instance_observations') else []):
                p = Path(local_doc[phase][field])
                p = p if p.is_absolute() else Path(manifest).resolve().parent/p
                dest = remote+'/inputs/'+phase+'-'+field+p.suffix
                uploads.append([str(p.resolve()), dest]); local_doc[phase][field] = dest
    else:
        for phase in ['before', 'after']:
            for field in ['mesh', 'gaussian', 'cameras'] + (['instance_observations'] if local_doc[phase].get('instance_observations') else []):
                p = PurePosixPath(local_doc[phase][field])
                if not p.is_absolute() or '..' in p.parts:
                    raise ValueError('Server inputs must use absolute POSIX paths')
    write(output/'submitted-input.json', local_doc)
    admin = {k: runtime[k] for k in ['simsac_source', 'simsac_checkpoint', 'person_checkpoint', 'person_deps'] if k in runtime}
    write(output/'submitted-runtime.local.json', admin)
    code = Path(__file__).resolve().parents[1]
    archive = output/'source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(code/'__init__.py', 'tools/__init__.py')
        for package in ['gaussian_change', 'mesh_simsac']:
            for file in (code/package).glob('*.py'):
                z.write(file, 'tools/'+package+'/'+file.name)
    relay.call(command='mkdir -p '+quote(remote+'/code')+' '+quote(remote+'/inputs'))
    uploads += [[str(archive), remote+'/source.zip'], [str(output/'submitted-input.json'), remote+'/input.json'],
                [str(output/'submitted-runtime.local.json'), remote+'/runtime.json']]
    relay.call(upload=uploads)
    environment = {str(k): str(v) for k, v in runtime.get('environment', {}).items()}
    script = ('import zipfile,subprocess,os,json;'
              f'zipfile.ZipFile({remote+"/source.zip"!r}).extractall({remote+"/code"!r});'
              f'env=os.environ.copy();env.update({environment!r});'
              f'log=open({remote+"/worker.log"!r},"w");'
              f'p=subprocess.Popen({[python,"-m","tools.gaussian_change.pipeline","--manifest",remote+"/input.json","--output",remote+"/result","--runtime",remote+"/runtime.json"]!r},'
              f'cwd={remote+"/code"!r},env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);'
              'print(json.dumps({"pid":p.pid}))')
    launch = json.loads(relay.call(command=quote(python)+' -c '+quote(script)))
    write(output/'remote-run.local.json', {'directory': remote, **launch})
    deadline = time.monotonic()+runtime.get('timeout_seconds', 14400)
    previous = None
    while True:
        script = (f'from pathlib import Path;import os,json;p=Path({remote+"/result/status.json"!r});'
                  f'proc=Path("/proc/{launch["pid"]}/stat");'
                  'alive=proc.exists() and proc.read_text().split()[2]!="Z";'
                  's=json.loads(p.read_text()) if p.exists() else {};'
                  f'log=Path({remote+"/worker.log"!r});'
                  's=s if alive or s.get("state") in ["complete","failed"] else '
                  '{"state":"failed","message":"Worker exited: "+(log.read_text()[-1500:] if log.exists() else "no log")};'
                  'print(json.dumps(s))')
        state = json.loads(relay.call(command=quote(python)+' -c '+quote(script)) or '{}')
        if state:
            write(output/'status.json', state)
            marker = (state.get('stage'), state.get('completed'))
            if marker != previous:
                print(state, flush=True); previous = marker
            if state.get('state') in ['complete', 'failed']:
                break
        if time.monotonic() > deadline:
            raise TimeoutError('GPU run still active; remote-run.local.json records its directory and PID')
        time.sleep(3)
    # Fetch partial diagnostics on failure as well as successful 3D results.
    script = (f'from pathlib import Path;import zipfile;p=Path({remote!r});'
              'z=zipfile.ZipFile(p/"result.zip","w",zipfile.ZIP_DEFLATED,compresslevel=1);'
              '[z.write(f,str(f.relative_to(p/"result"))) for f in (p/"result").rglob("*") if f.is_file()];'
              'z.write(p/"worker.log","worker.log");z.close()')
    relay.call(command=quote(python)+' -c '+quote(script))
    relay.call(get=[[remote+'/result.zip', str(output/'result.zip')]])
    with zipfile.ZipFile(output/'result.zip') as z:
        for entry in z.infolist():
            target = (output/entry.filename).resolve()
            if not target.is_relative_to(output) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe result archive path')
        z.extractall(output)
    if state['state'] == 'failed':
        raise RuntimeError(state.get('message', 'GPU worker failed; see worker.log'))
    return output/'changes.json'
