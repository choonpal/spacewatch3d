#!/usr/bin/env python3
"""Advance the patch release after accepting and validating an optimization."""
import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def read_version():
    tree = ast.parse((ROOT/'spacewatch_pipeline/__init__.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing package version')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:20]
    with (Path(tempfile.gettempdir())/('spacewatch-release-'+key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read_version()
        if old != args.expected_version:
            raise ValueError(f'Expected {args.expected_version}, found {old}; do not bump twice')
        if not re.fullmatch(r'\d+\.\d+\.\d+', old):
            raise ValueError('Expected a three-part numeric version')
        major, minor, patch = map(int, old.split('.'))
        version = f'{major}.{minor}.{patch+1}'
        state_path = ROOT/'docs/OPTIMIZATION_STATE.json'
        state = json.loads(state_path.read_text())
        if state.get('running_jobs'):
            raise ValueError('Finish existing jobs before changing code version')
        stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        run = f'/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v{version.replace(".", "_")}_{stamp[:10].replace("-", "")}'
        edits = {}
        edits[ROOT/'spacewatch_pipeline/__init__.py'] = f'__version__ = {version!r}\n'
        for path in sorted((ROOT/'configs').glob('*.json')):
            config = json.loads(path.read_text())
            if 'pipeline_version' not in config:
                continue
            if config['pipeline_version'] != old:
                raise ValueError(f'Inconsistent config version: {path}')
            config['pipeline_version'] = version
            edits[path] = json.dumps(config, ensure_ascii=False, indent=2)+'\n'
        # Only live entry-point defaults/instructions change. Historical evidence
        # under docs/ and completed run directories retain the producer version.
        old_runs = {state.get('next_run')}
        for name in ('start_v5.sh', 'start_v5_1.sh'):
            path = ROOT/name
            text = path.read_text()
            match = re.search(r'output_path="\$\{2:-(.*?)\}"', text)
            if not match:
                raise ValueError(f'Missing entry-point output default: {name}')
            old_runs.add(match.group(1))
            edits[path] = text[:match.start(1)]+run+text[match.end(1):]
        for name in ('README.txt', 'START_HERE.txt', 'connect_viewer.py'):
            path = ROOT/name
            text = path.read_text()
            for previous in sorted(p for p in old_runs if p):
                text = text.replace(previous, run)
            if name.endswith('.txt'):
                text = re.sub(r'^SpaceWatch v[\d.]+', 'SpaceWatch v'+version, text, count=1)
            edits[path] = text
        state.update(current_version=version, previous_version=old, release_created_at=stamp,
                     next_run=run, release_zip=f'spacewatch_pipeline_v{version.replace(".", "_")}.zip',
                     server_sync_confirmed=False, server_deployed_version=None,
                     status='release_validation_pending', release_tests=None,
                     server_tests_passed=None, server_doctor_passed=False,
                     full_pipeline_speedup_measured=False,
                     full_pipeline_wall_reduction_percent=None,
                     full_pipeline_measurement_scope='Current release pending; latest_validated fields retain prior release evidence',
                     full_regression_report=None)
        edits[state_path] = json.dumps(state, ensure_ascii=False, indent=2)+'\n'
        history_path = ROOT/'docs/RELEASE_HISTORY.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else {'releases': []}
        if not history['releases']:
            history['releases'].append(dict(version=old, status='historical baseline',
                                            validation_run=state.get('baseline_run')))
        history['releases'].append(dict(version=version, previous_version=old, created_at=stamp,
                                        reason=args.reason, optimization_iteration=state.get('iteration'),
                                        status='locally_validated', validation_run=run))
        edits[history_path] = json.dumps(history, ensure_ascii=False, indent=2)+'\n'
        lineage_path = ROOT/'docs/VERSION_LINEAGE.json'
        lineage = json.loads(lineage_path.read_text())
        lineage.update(version=version, previous_release=old,
                       baseline_server_run=state.get('baseline_run'), server_run=None,
                       server_run_status='pending', server_deployed_version=None,
                       latest_regression_report=None,
                       next_validation_run=run, release_history='docs/RELEASE_HISTORY.json')
        edits[lineage_path] = json.dumps(lineage, ensure_ascii=False, indent=2)+'\n'
        for path, text in edits.items():
            temp = path.with_name(path.name+'.release-tmp')
            temp.write_text(text)
            temp.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
            temp.replace(path)
        print(json.dumps(dict(previous_version=old, version=version, next_run=run,
                              release_zip=state['release_zip'], files_updated=len(edits))))


if __name__ == '__main__':
    main()
