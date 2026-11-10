"""One idle server GPU per conversation, with shared physical-GPU locks."""
import contextlib
import hashlib
import fcntl
import os
import socket
import subprocess
import time


@contextlib.contextmanager
def lease(config):
    allowed = config['runtime']['allowed_hostname']
    if socket.gethostname() != allowed:
        raise RuntimeError('Inference is restricted to the configured GPU server: '+allowed)
    thread_id=config['runtime'].get('gpu_thread_id') or os.environ.get('CAPSTONE_GPU_THREAD_ID') or os.environ.get('CODEX_THREAD_ID')
    if not thread_id:
        raise ValueError('Set runtime.gpu_thread_id or CAPSTONE_GPU_THREAD_ID to enforce one GPU per conversation')
    thread_key=hashlib.sha256(thread_id.encode()).hexdigest()[:24]
    wait_seconds=int(os.environ.get('SCENE_DISCOVERY_GPU_WAIT_SECONDS',config.get('gpu_wait_seconds',3600)))
    if wait_seconds<=0:raise ValueError('GPU wait must be positive')
    deadline = time.monotonic()+wait_seconds
    locks=[]
    try:
        while time.monotonic() < deadline:
            try:
                global_lock = open('/tmp/capstone-thread-'+thread_key+'-gpu.lock','a')
                fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locks=[global_lock]
                snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                                    '--format=csv,noheader,nounits'],text=True)
                busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
                for line in snapshot.splitlines():
                    index,uuid,mem,util = [x.strip() for x in line.split(',')]
                    if int(mem)>=100 or int(util)!=0 or uuid in busy:
                        continue
                    candidate=[]
                    try:
                        for name in [f'spacewatch-gpu-{uuid}',f'panovggt-pipeline-gpu-{index}',f'pointcloud-v2-gpu-{index}',f'segmentation-2d3d-{uuid}']:
                            lock=open('/tmp/'+name+'.lock','a'); candidate.append(lock)
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        fresh=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                        current_mem,current_util=[int(v.strip()) for v in fresh.split(',')]
                        if current_mem>=100 or current_util!=0:raise BlockingIOError('GPU became busy')
                        locks+=candidate
                        yield dict(index=index,uuid=uuid,before=snapshot,hostname=socket.gethostname(),thread_id=thread_id)
                        return
                    except BlockingIOError:
                        for lock in candidate: lock.close()
                for lock in locks: lock.close()
                locks=[]
            except BlockingIOError:
                global_lock.close()
            time.sleep(10)
        raise RuntimeError('No idle GPU became available within the configured wait time')
    finally:
        for lock in locks: lock.close()
