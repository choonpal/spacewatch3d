"""Cooperative single-GPU lease. Never terminates another workload."""
import contextlib
import fcntl
import subprocess
import time
from pathlib import Path

def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                        '--format=csv,noheader'], text=True)
    busy = set(processes.splitlines())
    return [dict(index=int(a.strip()), uuid=b.strip(), memory=int(c.strip()), utilization=int(d.strip()))
            for a,b,c,d in (row.split(',') for row in rows.splitlines()) if b.strip() not in busy]

@contextlib.contextmanager
def lease(timeout=3600, interval=10, directory='/tmp'):
    deadline = time.monotonic() + timeout
    while True:
        handles = []
        acquired = None
        try:
            h = open(Path(directory)/'capstone-single-gpu.lock', 'a')
            handles.append(h)
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for row in inventory():
                if row['memory'] >= 100 or row['utilization'] != 0:
                    continue
                h = open(Path(directory)/f'spacewatch-gpu-{row["uuid"]}.lock', 'a')
                try:
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    h.close()
                    continue
                handles.append(h)
                fresh = {r['uuid']:r for r in inventory()}.get(row['uuid'])
                if fresh and fresh['memory'] < 100 and fresh['utilization'] == 0:
                    acquired = row
                    break
            if acquired is None:
                raise BlockingIOError('No idle GPU')
        except BlockingIOError:
            for h in handles:
                h.close()
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU wait timed out. Resume the same command when a GPU is available.')
            time.sleep(min(interval, max(0, deadline-time.monotonic())))
        except BaseException:
            for h in handles:
                h.close()
            raise
        if acquired is not None:
            try:
                yield acquired
            finally:
                for h in handles:
                    h.close()
            return
