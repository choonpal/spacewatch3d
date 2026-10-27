"""Upload -> content-addressed first computation -> persistent viewer result."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
import zipfile

REPO = Path(__file__).resolve().parents[2]
PRESENTATION = {'index.html':'events-viewer.html', 'events-viewer.js':'events-viewer.js',
                'events-viewer.css':'events-viewer.css', 'photo-presentation.js':'photo-presentation.js'}
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.mesh_simsac.bundle import MAX_BYTES, digest, inspect_bundle, unpack


def read(path):
    for attempt in range(40):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.025)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(40):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.025)


def model_identity(runtime):
    """Hash actual model/source contents, including changes at an unchanged path."""
    fields = ['simsac_checkpoint'] + (['person_checkpoint'] if runtime.get('person_checkpoint') else [])
    if runtime.get('remote_session'):
        from tools.mesh_simsac.relay import Relay
        script = ('from pathlib import Path;import hashlib,json;'
                  'sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();'
                  f'files={ {k: runtime[k] for k in fields}!r};'
                  f'source=Path({runtime["simsac_source"]!r});'
                  'print(json.dumps({"models":{k:sha(p) for k,p in files.items()},'
                  '"source":{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*.py"))}}))')
        return json.loads(Relay(runtime['remote_session']).call(command=shlex.quote(runtime['python'])+' -c '+shlex.quote(script)))
    source = Path(runtime['simsac_source'])
    return {'models': {k: digest(runtime[k]) for k in fields},
            'source': {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))}}


class MeshChangeJobs:
    def __init__(self, root, runtime=None):
        self.root = Path(root).resolve()
        self.runtime = Path(runtime).resolve() if runtime else None
        self.lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='mesh-rgb')
        self.active = set()

    def configuration(self):
        configured = bool(self.runtime and self.runtime.is_file())
        connected = configured
        if configured:
            config = read(self.runtime)
            if config.get('remote_session'):
                from tools.mesh_simsac.relay import relay_connected
                connected = relay_connected(config['remote_session'])
        return dict(configured=configured, connected=connected, max_upload_bytes=MAX_BYTES,
                    input_schema='mesh-simsac-input-v1')

    def folder(self, key):
        if not re.fullmatch('[a-f0-9]{24}', key):
            raise KeyError('올바르지 않은 작업 ID입니다.')
        path = self.root/'jobs'/key
        if not (path/'job.json').is_file():
            raise KeyError('작업을 찾을 수 없습니다.')
        return path

    def submit(self, stream, length):
        if not self.configuration()['configured']:
            raise ValueError('변화탐지 실행 환경을 설정해 주세요.')
        if not 0 < length <= MAX_BYTES:
            raise ValueError('입력 ZIP 크기는 8GB 이하여야 합니다.')
        key = uuid.uuid4().hex[:24]
        folder = self.root/'jobs'/key; folder.mkdir(parents=True)
        write(folder/'job.json', dict(id=key, state='uploading', created_unix=time.time()))
        try:
            with (folder/'upload.zip').open('xb') as target:
                remaining = length
                while remaining:
                    block = stream.read(min(1024**2, remaining))
                    if not block:
                        raise ValueError('업로드가 중단되었습니다.')
                    target.write(block); remaining -= len(block)
            write(folder/'job.json', dict(id=key, state='queued', created_unix=time.time()))
            with self.lock:
                self.active.add(key)
            self.executor.submit(self._process, key)
            return self.status(key)
        except Exception as error:
            write(folder/'job.json', dict(id=key, state='failed', message=str(error)))
            raise

    def sources(self):
        from mesh_capture_sources import public_sources
        return public_sources(read(self.runtime)) if self.runtime else []

    def submit_sources(self, request):
        from mesh_capture_sources import select
        configuration = self.configuration()
        if not configuration['configured']:
            raise ValueError('변화탐지 실행 환경을 설정해 주세요.')
        if not configuration['connected']:
            raise ConnectionError('SSH 연결이 종료됐습니다. 서버에 다시 로그인해 주세요.')
        select(read(self.runtime), request)
        key = uuid.uuid4().hex[:24]
        folder = self.root/'jobs'/key; folder.mkdir(parents=True)
        write(folder/'job.json', dict(id=key, state='queued', created_unix=time.time(), source_selection=request))
        with self.lock:
            self.active.add(key)
        self.executor.submit(self._process, key)
        return self.status(key)

    def _process(self, key):
        folder = self.folder(key)
        data = read(folder/'job.json')
        try:
            if data.get('source_selection'):
                from mesh_capture_sources import prepare
                data.update(state='validating', stage='preparing_inputs'); write(folder/'job.json', data)
                prepare(read(self.runtime), data['source_selection'], folder)
            data.update(state='validating', stage='input_validation'); write(folder/'job.json', data)
            doc, hashes = unpack(folder/'upload.zip', folder/'inputs')
            runtime = read(self.runtime)
            identity = model_identity(runtime)
            if data.get('reprocess_parent'):
                parent=self.folder(data['reprocess_parent'])
                previous=read(parent/'attempt/identity.json')
                if identity != previous['models'] or digest(REPO/'tools/mesh_simsac/model.py') != previous['code']['model.py']:
                    raise ValueError('모델이 변경돼 기존 확률을 재사용할 수 없습니다. 새 비교가 필요합니다.')
                runtime['reprocess_source']=read(parent/'attempt/result/remote.local.json')['directory']+'/result/detection'
                runtime['reprocess_parent_viewer_sha256']=data['reprocess_parent_viewer_sha256']
            source = REPO/'tools/mesh_simsac'
            versions = {p.name: digest(p) for p in source.iterdir() if p.suffix in ['.py', '.js', '.html', '.css']}
            versions['jobs'] = digest(__file__)
            versions['structural_registration'] = digest(REPO/'tools/mesh_simsac/registration.py')
            # The manifest hash contains camera poses, thresholds and transforms.
            content = dict(files=hashes, models=identity, code=versions,
                           environment=runtime.get('environment', {}), device=runtime.get('device', 'cuda'))
            if data.get('reprocess_parent'):
                content['reprocess_parent_viewer_sha256']=data['reprocess_parent_viewer_sha256']
            fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            runtime['model_hashes'] = identity['models']
            cache = self.root/'cache'/fingerprint
            marker = cache/'complete.json'
            if marker.is_file():
                receipt = read(marker)
                valid = all((cache/receipt['generation']/name).is_file() and digest(cache/receipt['generation']/name) == value
                            for name, value in receipt['assets'].items())
                if valid:
                    data.update(state='complete', fingerprint=fingerprint, cached=True, counts=receipt['counts'], finished_unix=time.time())
                    write(folder/'job.json', data)
                    return
            # An incomplete/corrupt result is retained; the next attempt has a fresh directory.
            attempt = folder/'attempt'
            attempt.mkdir()
            write(attempt/'runtime.local.json', runtime)
            write(attempt/'identity.json', content)
            # Only declared, validated files are sent to the worker.
            with zipfile.ZipFile(attempt/'inputs.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for name in hashes:
                    z.write(folder/'inputs'/name, name)
            output = attempt/'result'
            python = runtime.get('local_python', sys.executable)
            if runtime.get('remote_session'):
                command = [python, '-m', 'tools.mesh_simsac.remote_worker', '--bundle', str(attempt/'inputs.zip')]
            else:
                command = [python, '-m', 'tools.mesh_simsac.worker', '--manifest', str(folder/'inputs/manifest.json')]
            command += ['--output', str(output), '--runtime', str(attempt/'runtime.local.json')]
            data.update(state='running', stage='starting', fingerprint=fingerprint, cached=False, started_unix=time.time())
            write(folder/'job.json', data)
            env = dict(os.environ)
            if not runtime.get('remote_session'):
                env.update(runtime.get('environment', {}))
            with (attempt/'worker.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            state = read(output/'status.json') if (output/'status.json').is_file() else {}
            if completed.returncode or state.get('state') != 'complete':
                raise RuntimeError(state.get('message') or (attempt/'worker.log').read_text(encoding='utf-8', errors='replace')[-1200:])
            self._publish(key, folder, output, data, content)
        except Exception as error:
            data.update(state='failed', message=str(error), finished_unix=time.time())
            write(folder/'job.json', data)
        finally:
            with self.lock:
                self.active.discard(key)

    def _publish(self, key, folder, output, data, content):
        public = output/'detection'
        result = read(public/'viewer-data.json')
        provenance=result.get('job_provenance', {})
        replay=(data.get('reprocess_parent') and provenance.get('reused_probabilities') and
                provenance.get('parent_viewer_sha256')==data.get('reprocess_parent_viewer_sha256'))
        if not provenance.get('fresh_inference') and not replay:
            raise ValueError('현재 입력의 새 추론 결과가 없습니다.')
        assets = {str(p.relative_to(public)).replace('\\', '/'): digest(p) for p in public.rglob('*')
                  if p.is_file() and self.public_asset(str(p.relative_to(public)).replace('\\', '/'))}
        cache = self.root/'cache'/data['fingerprint']
        marker = cache/'complete.json'
        cache.mkdir(parents=True, exist_ok=True)
        # Use a unique generation, then atomically publish its receipt. Never
        # expose a half-downloaded worker directory as a completed result.
        import shutil
        generation = cache/('generation-'+key+'-'+uuid.uuid4().hex[:8])
        shutil.copytree(public, generation, ignore=lambda path, names: [n for n in names if (Path(path)/n).is_file()
                        and not self.public_asset(str((Path(path)/n).relative_to(public)).replace('\\', '/'))])
        write(marker, dict(assets=assets, counts=result['counts'], generation=generation.name, identity=content))
        data.update(state='complete', counts=result['counts'], finished_unix=time.time())
        write(folder/'job.json', data)

    def submit_reprocess(self, parent_key):
        """New auditable job using the parent's exact probabilities and inputs."""
        import shutil
        parent=self.folder(parent_key)
        if read(parent/'job.json')['state']!='complete':
            raise ValueError('완료된 비교만 재집계할 수 있습니다.')
        if not (parent/'attempt/result/remote.local.json').is_file():
            raise ValueError('서버 원본 확률 기록이 필요합니다.')
        key=uuid.uuid4().hex[:24];folder=self.root/'jobs'/key;folder.mkdir()
        shutil.copyfile(parent/'upload.zip',folder/'upload.zip')
        write(folder/'job.json',dict(id=key,state='queued',created_unix=time.time(),reprocess_parent=parent_key,
            reprocess_parent_viewer_sha256=digest(self.asset(parent_key,'viewer-data.json'))))
        with self.lock:self.active.add(key)
        self.executor.submit(self._process,key)
        return self.status(key)

    def retry(self, key):
        with self.lock:
            folder = self.folder(key); data = read(folder/'job.json')
            if data['state'] == 'complete' or key in self.active:
                return self.status(key)
            if not (folder/'attempt/result/remote.local.json').is_file():
                raise ValueError('재연결할 GPU 작업이 없습니다. 입력을 다시 제출해 주세요.')
            data.update(state='queued', message=None); write(folder/'job.json', data)
            self.active.add(key)
            self.executor.submit(self._resume, key)
        return self.status(key)

    def _resume(self, key):
        folder = self.folder(key); data = read(folder/'job.json'); attempt = folder/'attempt'
        try:
            runtime = read(attempt/'runtime.local.json')
            data.update(state='running', message=None); write(folder/'job.json', data)
            command = [runtime.get('local_python', sys.executable), '-m', 'tools.mesh_simsac.remote_worker',
                       '--bundle', str(attempt/'inputs.zip'), '--output', str(attempt/'result'),
                       '--runtime', str(attempt/'runtime.local.json')]
            with (attempt/'resume.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            state = read(attempt/'result/status.json')
            if process.returncode or state.get('state') != 'complete':
                raise RuntimeError(state.get('message', 'GPU 결과 재연결에 실패했습니다.'))
            identity = read(attempt/'identity.json') if (attempt/'identity.json').is_file() else {'resumed_original_job': key}
            self._publish(key, folder, attempt/'result', data, identity)
        except Exception as error:
            data.update(state='failed', message=str(error)); write(folder/'job.json', data)
        finally:
            with self.lock:
                self.active.discard(key)

    def status(self, key):
        with self.lock:
            folder = self.folder(key); data = read(folder/'job.json')
            if data['state'] in ['queued', 'validating', 'running'] and key not in self.active:
                data.update(state='interrupted', message='서버가 재시작되었습니다. 같은 입력을 다시 넣어 실행해 주세요.')
        state = {}
        path = folder/'attempt/result/status.json'
        if data['state'] == 'running' and path.is_file():
            state = read(path)
        return {k: v for k, v in dict(id=key, state=data['state'], stage=state.get('stage', data.get('stage', data['state'])),
                source_selection=data.get('source_selection'),
                message=data.get('message') or (state.get('message') if state.get('state') != 'failed' else None),
                can_resume=(folder/'attempt/result/remote.local.json').is_file() and data['state'] in ['failed', 'interrupted'], cached=data.get('cached', False), counts=data.get('counts'),
                result_url=f'/api/mesh-change-jobs/{key}/files/index.html' if data['state'] == 'complete' else None).items()}

    @staticmethod
    def public_asset(name):
        return bool(name in ['index.html', 'rgb-viewer.js', 'rgb-viewer.css', 'events-viewer.js', 'events-viewer.css', 'photo-presentation.js', 'viewer-data.json', 'validation.json', 'results/events.json']
                    or re.fullmatch(r'events/(?:objects/E[01]-I\d+\.bin|photos/[\w-]+\.(?:jpg|png))', name)
                    or re.fullmatch(r'preview/(?:mesh-[01]\.bin|cloud-[01]\.bin|vendor/(?:three\.(?:module|core)\.js|OrbitControls\.js|LICENSE))', name))

    def asset(self, key, name):
        folder = self.folder(key); data = read(folder/'job.json')
        if data['state'] != 'complete' or not self.public_asset(name):
            raise KeyError('공개된 결과 파일이 없습니다.')
        cache = self.root/'cache'/data['fingerprint']
        receipt = read(cache/'complete.json')
        # Presentation updates apply to completed jobs without altering their
        # immutable inference data, geometry, classifications or cache receipt.
        if name in PRESENTATION and 'events-viewer.js' in receipt['assets']:
            return REPO/'tools/mesh_simsac'/PRESENTATION[name]
        if name not in receipt['assets']:
            raise KeyError('결과 파일이 없습니다.')
        return cache/receipt['generation']/name
