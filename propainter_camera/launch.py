"""Run SAM2 and ProPainter sequentially on one idle server GPU."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    busy={line.strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
    return [dict(index=int(a),uuid=b.strip(),name=c.strip(),memory_mib=int(d),utilization=int(e))
            for a,b,c,d,e in (line.split(',') for line in rows.splitlines()) if b.strip() not in busy]


def main():
    output=Path(sys.argv[1]).resolve()
    # Keep the same chat-wide lock used by earlier jobs in this chat.
    with open('/tmp/codex-panogrf-mask-01a17b96-gpu.lock','a') as thread_lock:
        fcntl.flock(thread_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handle=device=None
        for row in inventory():
            if row['memory_mib']>=100 or row['utilization']!=0:
                continue
            candidate=open('/tmp/spacewatch-gpu-'+row['uuid']+'.lock','a')
            try:
                fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            fresh={r['uuid']:r for r in inventory()}.get(row['uuid'])
            if fresh and fresh['memory_mib']<100 and fresh['utilization']==0:
                handle,device=candidate,fresh
                break
            candidate.close()
        if device is None:
            raise RuntimeError('No idle GPU. No existing workload was changed.')
        try:
            (output/'gpu_lease.json').write_text(json.dumps(dict(device=device,pid=os.getpid(),started_unix=time.time()),indent=2))
            environment=os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES']=device['uuid']
            with (output/'gpu.log').open('w') as log:
                run=subprocess.run(['bash',str(ROOT/'run_env.sh'),str(ROOT/'gpu_worker.py'),str(output)],
                                   env=environment,stdout=log,stderr=subprocess.STDOUT)
            (output/'gpu_exit.json').write_text(json.dumps(dict(exit_code=run.returncode,finished_unix=time.time()),indent=2))
            raise SystemExit(run.returncode)
        finally:
            handle.close()


if __name__=='__main__':
    main()
