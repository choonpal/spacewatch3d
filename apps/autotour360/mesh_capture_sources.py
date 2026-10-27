"""Trusted server capture catalogue and automatic source bundle preparation."""
import json
from pathlib import Path, PurePosixPath
import shlex
import time

from tools.gaussian_change.remote import Relay


def catalog(runtime):
    path = runtime.get('capture_catalog')
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {'captures': []}


def select(runtime, request):
    if not isinstance(request, dict) or set(request) != {'before', 'after'} or not all(isinstance(v, str) for v in request.values()):
        raise ValueError('전후 촬영 ID를 선택해 주세요.')
    sources = catalog(runtime)
    allowed = {s['id']: s for s in sources['captures']}
    if request['before'] == request['after']:
        raise ValueError('서로 다른 전후 촬영을 선택해 주세요.')
    if any(v not in allowed for v in request.values()):
        raise ValueError('등록된 서버 촬영만 선택할 수 있습니다.')
    return {**sources, 'captures': [allowed[request[p]] for p in ['before', 'after']]}


def public_sources(runtime):
    return [dict(id=s['id'], label=s.get('label', s['id']), mesh_files=1,
                 candidate_regions=s.get('candidate_regions'), segmentation=s.get('segmentation_model', ''),
                 description='기존 영상·클래스 메쉬·객체 후보 영역·점군·카메라')
            for s in catalog(runtime)['captures']]


def prepare(runtime, request, folder):
    config = select(runtime, request)
    if not runtime.get('remote_session'):
        raise ValueError('서버 촬영 입력에는 연결된 분석 서버가 필요합니다.')
    folder = Path(folder)
    root = PurePosixPath(runtime['remote_root'])
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('Invalid configured server root')
    remote = str(root/('capture-input-'+folder.name))
    config_path = folder/'server-inputs.local.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
    script_path = Path(__file__).resolve().parents[2]/'tools/mesh_simsac/import_semantic_scene.py'
    relay = Relay(runtime['remote_session'])
    quote = shlex.quote
    relay.call(command='mkdir -p '+quote(remote))
    relay.call(upload=[[str(config_path), remote+'/config.json'],
                       [str(script_path), remote+'/prepare.py']])
    command = [runtime['python'], remote+'/prepare.py', '--config', remote+'/config.json', '--output', remote+'/inputs']
    # The viewer job owns input preparation and the subsequent fresh inference.
    # Redirect long frame progress to its own log; never substitute a saved result.
    code = ('import subprocess,os,json;'
            f'env=os.environ.copy();env.update({runtime.get("environment", {})!r});'
            f'log=open({remote+"/prepare.log"!r},"w");'
            f'p=subprocess.Popen({command!r},env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);'
            'print(json.dumps({"pid":p.pid}))')
    process = json.loads(relay.call(command=quote(runtime['python'])+' -c '+quote(code)))
    (folder/'server-preparation.local.json').write_text(json.dumps(dict(directory=remote, **process)))
    deadline = time.monotonic()+runtime.get('timeout_seconds', 14400)
    while True:
        code = ('from pathlib import Path;import json;'
                f'p=Path({remote+"/inputs.status.json"!r});proc=Path("/proc/{process["pid"]}/stat");'
                'alive=proc.exists() and proc.read_text().split()[2]!="Z";'
                's=json.loads(p.read_text()) if p.exists() else {"state":"running"};'
                's=s if alive or s["state"]!="running" else {"state":"failed","message":"Input preparation process stopped"};'
                'print(json.dumps(s))')
        try:
            state = json.loads(relay.call(command=quote(runtime['python'])+' -c '+quote(code)))
        except ConnectionError:
            if time.monotonic() > deadline:
                raise TimeoutError('서버 재연결 대기 시간이 초과됐습니다. 기존 입력 준비 작업은 서버 기록에 보존됩니다.')
            time.sleep(4)
            continue
        if state['state'] == 'complete':
            break
        if state['state'] == 'failed':
            (folder/'source-error.log').write_text(state.get('message', 'Preparation failed'), encoding='utf-8')
            raise ValueError('선택한 촬영의 입력 연결을 확인하지 못했습니다. 서버 입력 기록을 확인해 주세요.')
        if time.monotonic() > deadline:
            raise TimeoutError('입력 준비가 지연되고 있습니다. 서버 입력 작업 기록을 확인해 주세요.')
        time.sleep(4)
    relay.call(get=[[remote+'/inputs.zip', str(folder/'upload.zip')],
                    [remote+'/inputs/source-audit.json', str(folder/'source-audit.json')]])
