#!/usr/bin/env python3
"""Standalone commands: doctor, run, start, status, serve. No agent or service required."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from common import ROOT, NAME, VERSION, digest, read_json, save_json, progress

def prepare(config):
    import numpy as np
    from core import select_frames
    if socket.gethostname() != config['execution']['hostname']:
        raise RuntimeError('This configuration may execute only on the designated GPU server; no local GPU fallback.')
    assert config['name'] == NAME and config['version'] == VERSION
    source = Path(config['source_run'])
    assert read_json(source / 'run_identity.json')['sha256'] == config['source_identity']
    settings = read_json(source / 'config.json')
    map_path = source / 'result' / settings['output']['ply_name']
    assert digest(map_path) == config['source_map_sha256'], 'Source map checksum mismatch'
    assert digest(config['model']['checkpoint']) == config['model']['checkpoint_sha256'], 'Checkpoint checksum mismatch'
    revision = subprocess.check_output(['git', '-C', config['model']['repository'], 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == config['model']['revision']
    for file, checksum in config['model']['modified_files'].items():
        assert digest(Path(config['model']['repository']) / file) == checksum, f'Model file mismatch: {file}'
    manifest = read_json(source / 'input/manifest.json')
    assert manifest['projection'] == 'equirectangular'
    selected = select_frames(manifest, config['start_seconds'], config['end_seconds'])
    assert len(selected) <= 65535
    frames = []
    for i in selected:
        row = manifest['frames'][i]
        path = source / 'input' / row['file']
        assert digest(path) == row['sha256'], f'Input image changed: {path}'
        frames.append(dict(index=i, timestamp_s=row['timestamp_s'], path=str(path), sha256=row['sha256']))
    inputs = {str(p): digest(p) for p in [source / 'prepared.npz', source / 'optimized.npz',
              source / 'input/manifest.json', source / 'source_hashes.json', source / 'config.json']}
    from filtered_source import source_layout
    layout = source_layout(config, settings)
    if layout['keep_mask'] is not None:
        info = read_json(layout['report'])
        if info['sha256'] != config['source_map_sha256']:
            raise ValueError('Filtered target checksum differs from component filter provenance')
        for path in [layout['unfiltered'], layout['keep_mask'], layout['report']]:
            inputs[str(path)] = digest(path)
        if inputs[str(layout['unfiltered'])] != info['source_sha256']:
            raise ValueError('Unfiltered map checksum differs from component filter provenance')
    inputs['class_names'] = digest(config['model']['class_names'])
    model_code = {str(p.relative_to(config['model']['repository'])): digest(p)
                  for p in sorted(Path(config['model']['repository']).rglob('*.py'))}
    code = {p.name: digest(p) for p in sorted(ROOT.glob('*.py')) if p.name != 'export_results.py'}
    # Output formatting is tracked but does not invalidate expensive inference/provenance caches.
    identity = hashlib.sha256(json.dumps(dict(config=config, inputs=inputs, code=code, model_code=model_code), sort_keys=True).encode()).hexdigest()
    return dict(identity=identity, frames=frames, source_frames=len(manifest['frames']),
                source_map=str(map_path), source_inputs=inputs, code=code, model_code=model_code,
                source_generator_version=settings.get('generator_version', '1.0.0'),
                export_code_sha256=digest(ROOT / 'export_results.py'))

def run(config):
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    try:
        progress(out, 'verifying_inputs')
        plan = prepare(config)
        previous = out / 'plan.json'
        if previous.exists() and read_json(previous)['identity'] != plan['identity']:
            raise RuntimeError('Configuration, source, or algorithm changed. Use a new output directory to preserve existing results.')
        save_json(previous, plan)
        save_json(out / 'config.json', config)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = str(config['execution']['cpu_threads'])
        env['LC_ALL'] = 'C'
        subprocess.run([sys.executable, str(ROOT / 'inference.py'), str(out / 'config.json'), str(previous)], env=env, check=True)
        # The inference subprocess has exited here, releasing GPU memory before CPU fusion.
        from geometry import build_index, project
        from export_results import export, build_viewer
        build_index(config, plan)
        project(config, plan)
        export(config, plan)
        build_viewer(config, plan)
        elapsed = time.perf_counter() - started
        status = dict(status='complete', name=NAME, version=VERSION, pid=os.getpid(),
                      seconds_this_invocation=elapsed, identity=plan['identity'],
                      summary=str(out / 'result/summary.json'), viewer=str(out / 'viewer/index.html'))
        save_json(out / 'status.json', status)
        # Retain the first successful runtime when subsequent invocations validate cached output.
        if not (out / 'first_completion.json').exists():
            save_json(out / 'first_completion.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except BaseException as exc:
        save_json(out / 'status.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc(), pid=os.getpid()))
        raise

def main():
    parser = argparse.ArgumentParser(description=f'{NAME} ({VERSION})')
    parser.add_argument('command', choices=['doctor', 'run', 'start', 'status', 'serve'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dorm3_0_30.json')
    parser.add_argument('--port', type=int, default=45426)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    out = Path(config['output'])
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = str(config['execution']['cpu_threads'])
    if args.command == 'doctor':
        plan = prepare(config)
        print(json.dumps(dict(status='ready', frames=len(plan['frames']), identity=plan['identity']), ensure_ascii=False))
    elif args.command == 'run':
        run(config)
    elif args.command == 'start':
        if socket.gethostname() != config['execution']['hostname']:
            raise RuntimeError('Launch this command on the configured server')
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'job.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'cli.py'), 'run', '--config', str(args.config.resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(out / 'launch.json', dict(pid=process.pid, log=str(out / 'job.log')))
        print(f'PID {process.pid}; log: {out / "job.log"}. Execution survives SSH disconnection.')
    elif args.command == 'status':
        data = read_json(out / 'status.json') if (out / 'status.json').exists() else {'status': 'not_started'}
        if data.get('status') == 'running':
            try:
                os.kill(data['pid'], 0)
            except ProcessLookupError:
                data['status'] = 'interrupted; rerun start to resume'
        print(json.dumps(data, ensure_ascii=False, indent=2))
    elif args.command == 'serve':
        import http.server
        import functools
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out / 'viewer'))
        print(f'Viewer: http://127.0.0.1:{args.port}', flush=True)
        http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler).serve_forever()

if __name__ == '__main__':
    main()
