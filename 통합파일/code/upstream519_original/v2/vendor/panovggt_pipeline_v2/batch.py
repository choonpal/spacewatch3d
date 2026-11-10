#!/usr/bin/env python3
"""Sequential fresh inference for configured videos on one physical GPU."""
from pathlib import Path
import argparse
import datetime
import fcntl
import subprocess
import sys
import time
from common import read, save, validate_config

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True, help='JSON list of config/output objects')
    parser.add_argument('--report-directory', required=True)
    args = parser.parse_args()
    jobs = read(args.jobs)
    if not jobs:
        raise ValueError('Empty job list')
    configs = [read(job['config']) for job in jobs]
    for c in configs:
        validate_config(c)
    if len({c['execution']['physical_gpu'] for c in configs}) != 1:
        raise ValueError('All videos must use the same single physical GPU')
    out = Path(args.report_directory); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.batch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    report = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  physical_gpu=configs[0]['execution']['physical_gpu'], sequential=True,
                  neural_prediction_cache_reused=False, jobs=[])
    save(out/'batch_report.json', report)
    for i, job in enumerate(jobs):
        begin = time.perf_counter()
        item = dict(**job, status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report['jobs'].append(item); save(out/'batch_report.json', report)
        command = [sys.executable, '-u', str(HERE/'run.py'), '--config', job['config'], '--output', job['output']]
        with (out/f'job_{i}.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        item.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode,
                    wall_seconds=time.perf_counter()-begin,
                    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report['wall_seconds'] = time.perf_counter()-started
        if result.returncode:
            report['status'] = 'failed'; save(out/'batch_report.json', report)
            raise SystemExit(result.returncode)
        item['run_report'] = str(Path(job['output'])/'run_report.json')
        save(out/'batch_report.json', report)
    report.update(status='complete', wall_seconds=time.perf_counter()-started,
                  ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'batch_report.json', report)
    print('BATCH_COMPLETE', report['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
