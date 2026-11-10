from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest, read, record, save, verify

R = Path(os.environ['PANOVGGT_RUN_ROOT']).resolve()
C = read(R / 'config.json')
os.environ['CUDA_VISIBLE_DEVICES'] = str(C['physical_gpu'])
os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(C['execution']['cpu_threads'])


def gpu_guard():
    import subprocess
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    hostname = subprocess.check_output(['hostname'], text=True).strip()
    if C['execution']['server_address'] not in addresses and hostname != C['execution'].get('server_container_hostname'):
        raise RuntimeError('GPU work is allowed only on the configured server; no local/Colab fallback')
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    stats = {int(s.split(',')[0]): [int(x) for x in s.split(',')[1:]] for s in result.strip().splitlines()}
    memory, utilization = stats[C['physical_gpu']]
    if memory >= C['execution']['gpu_max_memory_mb'] or utilization >= C['execution']['gpu_max_utilization_percent']:
        raise RuntimeError(f'Configured GPU is busy: {stats[C["physical_gpu"]]}; no automatic device switch')
    save(R / 'last_gpu_check.json', {'physical_gpu': C['physical_gpu'], 'statistics': stats})


def replay_files(folder, names):
    """A replay is explicit, checksummed, and cannot silently fall back to inference."""
    receipt = R / 'replay_inventory.json'
    if not receipt.exists():
        return False
    import shutil
    inventory = read(receipt)['files']
    for name in names:
        key = f'{folder}/{name}'
        if key not in inventory:
            raise RuntimeError(f'Replay cache missing {key}; GPU fallback is disabled')
        source = verify(inventory[key])
        dest = R / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if digest(dest) != inventory[key]['sha256']:
                raise RuntimeError(f'Replay destination differs: {dest}')
        else:
            # Symlinks retain server data in place. All workers treat imported arrays as read-only.
            dest.symlink_to(source)
    return True
