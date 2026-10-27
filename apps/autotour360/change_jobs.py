"""Browser input jobs invoking the same fresh mesh/Gaussian CLI as system users."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
from trajectory_jobs import write_json


FIELDS = [p+'-'+f for p in ['before', 'after'] for f in ['mesh', 'gaussian', 'cameras']]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class ChangeJobs:
    def __init__(self, root, runtime=None, preset=None):
        self.root = Path(root).resolve()
        self.runtime_path = Path(runtime).resolve() if runtime else None
        self.preset_path = Path(preset).resolve() if preset else None
        self.processes = {}
        self.lock = threading.RLock()

    def configuration(self):
        configured = bool(self.runtime_path and self.runtime_path.is_file())
        connected = configured
        if configured:
            config = read(self.runtime_path)
            if config.get('remote_session'):
                try:
                    s = read(Path(config['remote_session'])/'status.json')
                    connected = s.get('state') == 'connected' and time.time()-s.get('heartbeat_unix', 0) < 90
                except (OSError, ValueError):
                    connected = False
        return {'configured': configured, 'connected': connected,
                'server_inputs_available': bool(self.preset_path and self.preset_path.is_file()),
                'input_fields': FIELDS}

    def folder(self, key):
        if not re.fullmatch(r'[a-f0-9]{24}', key):
            raise KeyError('작업 ID가 올바르지 않습니다.')
        p = self.root/key
        if not (p/'job.json').is_file():
            raise KeyError('변화탐지 작업을 찾을 수 없습니다.')
        return p

    def create(self, source):
        if source not in ['upload', 'server']:
            raise ValueError('입력 방식을 선택해 주세요.')
        if source == 'server' and not self.configuration()['server_inputs_available']:
            raise ValueError('서버 입력 묶음이 설정되지 않았습니다.')
        key = uuid.uuid4().hex[:24]
        p = self.root/key; p.mkdir(parents=True)
        write_json(p/'job.json', {'id': key, 'source': source, 'state': 'created', 'uploads': {}})
        return self.status(key)

    def upload(self, key, field, stream, length, name):
        p = self.folder(key)
        if field not in FIELDS:
            raise ValueError('알 수 없는 입력입니다.')
        suffix = Path(name).suffix.lower()
        allowed = ['.npz', '.ply', '.obj'] if field.endswith('mesh') else ['.ply'] if field.endswith('gaussian') else ['.json']
        maximum = 16*1024**2 if field.endswith('cameras') else 2*1024**3
        if suffix not in allowed or not 0 < length <= maximum:
            raise ValueError('입력 형식 또는 파일 크기를 확인해 주세요.')
        with self.lock:
            data = read(p/'job.json')
            if data['source'] != 'upload' or data['state'] not in ['created', 'uploading'] or field in data['uploads']:
                raise ValueError('이미 업로드했거나 실행을 시작한 입력입니다.')
            data['state'] = 'uploading'; data['uploads'][field] = {'state': 'uploading'}
            write_json(p/'job.json', data)
        destination = p/'inputs'/(field+suffix); destination.parent.mkdir(exist_ok=True)
        try:
            digest=hashlib.sha256()
            with destination.open('xb') as target:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024**2, remaining))
                    if not chunk:
                        raise ValueError('입력 전송이 중단됐습니다. 새 작업으로 다시 시도해 주세요.')
                    target.write(chunk);digest.update(chunk); remaining -= len(chunk)
            with self.lock:
                data = read(p/'job.json')
                data['uploads'][field] = {'state': 'ready', 'file': str(destination), 'name': Path(name).name, 'bytes': length,'sha256':digest.hexdigest()}
                write_json(p/'job.json', data)
        except BaseException:
            with self.lock:
                data = read(p/'job.json'); data['state'] = 'failed'; write_json(p/'job.json', data)
            raise
        return self.status(key)

    def start(self, key, descriptor=None):
        with self.lock:
            p = self.folder(key); data = read(p/'job.json')
            if data['state'] not in ['created', 'uploading']:
                raise ValueError('이미 실행된 작업입니다. 새 입력 작업을 만들어 주세요.')
            if not self.configuration()['connected']:
                raise ValueError('GPU 실행 환경 또는 SSH 연결을 확인해 주세요.')
            if any(proc.poll() is None for proc in self.processes.values()):
                raise ValueError('다른 변화탐지가 실행 중입니다. 완료 후 다시 실행해 주세요.')
            if data['source'] == 'server':
                manifest = read(self.preset_path)
            else:
                if any(data['uploads'].get(f, {}).get('state') != 'ready' for f in FIELDS):
                    raise ValueError('전후 메시·Gaussian·카메라 여섯 파일을 모두 입력해 주세요.')
                if not isinstance(descriptor, dict):
                    raise ValueError('자료 사이의 좌표 연결 정보를 입력해 주세요.')
                manifest = {'schema': 'mesh-gaussian-change-input-v1', 'world_up': '+Y',
                            'units': descriptor.get('units'), 'settings': descriptor.get('settings', {})}
                for phase in ['before', 'after']:
                    spec = descriptor.get(phase, {})
                    manifest[phase] = {k: copy.deepcopy(spec[k]) for k in ['world_from_mesh', 'world_from_gaussian', 'world_from_camera', 'camera_convention'] if k in spec}
                    for field in ['mesh', 'gaussian', 'cameras']:
                        manifest[phase][field] = data['uploads'][phase+'-'+field]['file']
            if descriptor is not None and not isinstance(descriptor,dict):raise ValueError('분석 설정 객체가 필요합니다.')
            profile=(descriptor or {}).get('analysis_profile')
            if profile not in [None,'input','4k-perspective','1k-panorama']:
                raise ValueError('알 수 없는 분석 품질 설정입니다.')
            if profile in ['4k-perspective','1k-panorama']:
                settings=manifest.setdefault('settings',{})
                settings.update(width=4096 if profile=='4k-perspective' else 1024,height=2048 if profile=='4k-perspective' else 512,
                                cube_size=2048 if profile=='4k-perspective' else 384,
                                inference_mode='perspective_tiles' if profile=='4k-perspective' else 'panorama',
                                perspective_tile_size=1040,perspective_tile_overlap=32)
            runtime = read(self.runtime_path)
            runtime['input_location'] = 'server' if data['source'] == 'server' else 'local'
            bindings=[]
            if data['source']=='upload' and runtime.get('instance_observations_registry'):
                registry=read(runtime['instance_observations_registry'])
                for phase in ['before','after']:
                    camera_path=Path(manifest[phase]['cameras']);camera=read(camera_path)
                    mesh_path=Path(manifest[phase]['mesh']);mesh_sha=data['uploads'][phase+'-mesh'].get('sha256') or hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                    source=registry.get('meshes',{}).get(mesh_sha)
                    if not camera.get('instance_segmentation') and source:
                        segmentation=read(source['file'])
                        if segmentation.get('mesh_sha256')!=mesh_sha:
                            raise ValueError('등록된 인스턴스 관측과 입력 메시가 일치하지 않습니다.')
                        camera['instance_segmentation']=segmentation
                        prepared=p/'prepared'/(phase+'-cameras-with-instances.json');write_json(prepared,camera)
                        manifest[phase]['cameras']=str(prepared)
                        bindings.append({'capture':phase,'mesh_sha256':mesh_sha,'source_observations':len(segmentation.get('observations',[])),
                                         'source_cameras':len(segmentation.get('cameras',[])),'original_camera_sha256':hashlib.sha256(camera_path.read_bytes()).hexdigest()})
            data['instance_bindings']=bindings
            write_json(p/'input.local.json', manifest)
            write_json(p/'runtime.local.json', runtime)
            python = runtime.get('local_python', sys.executable)
            command = [python, '-m', 'tools.gaussian_change.pipeline', '--manifest', str(p/'input.local.json'),
                       '--runtime', str(p/'runtime.local.json'), '--output', str(p/'result')]
            # New output and CLI process for every click. No prior job result is read.
            with (p/'launcher.log').open('w', encoding='utf-8') as log:
                proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log,
                                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.processes[key] = proc
            data.update(state='running', pid=proc.pid, started_unix=time.time())
            write_json(p/'job.json', data)
            return self.status(key)

    def status(self, key):
        p = self.folder(key); data = read(p/'job.json'); state = data['state']
        details = {}
        if (p/'result/status.json').is_file():
            try:
                details = read(p/'result/status.json')
            except (OSError, ValueError):
                pass
        proc = self.processes.get(key)
        if state == 'running':
            if proc is not None and proc.poll() is None:
                # Remote completion precedes result download; keep the UI running.
                if details.get('state') == 'complete':
                    details = {'stage': 'receiving_results', 'message': '새 결과를 뷰어로 가져오는 중'}
            elif (p/'result/changes.json').is_file() and details.get('state') == 'complete':
                state = 'complete'
            elif proc is not None and proc.poll() is not None:
                state = 'failed'
                if not details.get('message'):
                    details['message'] = (p/'launcher.log').read_text(encoding='utf-8', errors='replace')[-1200:]
            elif proc is None:
                state = 'interrupted'; details['message'] = '뷰어 서버가 재시작됐습니다. 새 작업으로 실행해 주세요.'
        if state in ['complete', 'failed'] and data['state'] != state:
            data['state'] = state
            completed = p/'result/worker.log'
            data['finished_unix'] = completed.stat().st_mtime if state == 'complete' and completed.is_file() else time.time()
            if details.get('message'):
                data['message'] = details['message']
            write_json(p/'job.json', data)
        return {'id': key, 'source': data['source'], 'state': state, 'stage': details.get('stage', 'transferring_inputs' if state == 'running' else state),
                'message': details.get('message', data.get('message')), 'completed': details.get('completed'), 'total': details.get('total'),
                'elapsed_s': max(0, data.get('finished_unix', time.time())-data['started_unix']) if data.get('started_unix') else 0,
                'uploads': {k: {'name': v.get('name'), 'state': v['state']} for k, v in data['uploads'].items()},
                'instance_bindings':data.get('instance_bindings',[]),
                'result_url': f'/api/change-jobs/{key}/files/changes.json' if state == 'complete' else None}

    def asset(self, key, name):
        if self.status(key)['state'] != 'complete':
            raise KeyError('변화탐지가 아직 완료되지 않았습니다.')
        if not (name in ['changes.json', 'registration.json', 'cameras.json', 'inputs.json', 'self-control.json', 'model.json', 'object-events.json', 'comparison.json'] or
                re.fullmatch(r'renders/V\d{5}/(?:before|after)(?:-change)?\.png', name)):
            raise KeyError('공개할 수 없는 결과 파일입니다.')
        return self.folder(key)/'result'/name
