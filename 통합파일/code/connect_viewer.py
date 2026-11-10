#!/usr/bin/env python3
"""Run on your PC: start/reuse the server viewer and open an SSH-forwarded URL."""
import argparse
import json
import shlex
import socket
import subprocess
import tempfile
import urllib.request
import webbrowser
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='root@155.230.28.172')
    p.add_argument('--ssh-port',type=int,default=4444)
    p.add_argument('--server-root',default='/workspace/spacewatch_pipeline_v5_1')
    p.add_argument('--run',default='/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_9_20261027')
    p.add_argument('--local-port',type=int,help='Fixed local port. Default: 18786, or a free port if occupied.')
    p.add_argument('--no-open',action='store_true')
    p.add_argument('--control-path',type=Path,help='Reuse an already authenticated OpenSSH control socket')
    a=p.parse_args()
    requested_port=a.local_port
    if requested_port is not None and not 1<=requested_port<=65535:p.error('Invalid local port')
    a.local_port=requested_port or 18786
    directory=None if a.control_path else Path(tempfile.mkdtemp(prefix='spacewatch-viewer-'))
    control=a.control_path or directory/'ssh'
    common=['ssh','-S',str(control),'-p',str(a.ssh_port),a.host]
    connected=False
    try:
        # SSH prompts securely in the terminal. No password is written to a file.
        if directory:subprocess.run(['ssh','-M','-S',str(control),'-p',str(a.ssh_port),
                        '-o','ControlPersist=12h','-o','ServerAliveInterval=30',
                        '-o','ServerAliveCountMax=3','-fN',a.host],check=True)
        remote='cd '+shlex.quote(a.server_root)+' && '+shlex.join(['./run.sh','viewer','--output',a.run])
        info=json.loads(subprocess.check_output(common+[remote],text=True))
        base=f'http://127.0.0.1:{a.local_port}'
        reused=False
        try:
            with urllib.request.urlopen(base+'/api/pipeline',timeout=1) as response:
                reused=json.load(response).get('bundle_id')==info['bundle_id']
        except (OSError,ValueError):pass
        if not reused:
            # Never replace another viewer/service. A plain invocation remains usable
            # even when another project already occupies the preferred local port.
            with socket.socket() as probe:
                try:probe.bind(('127.0.0.1',a.local_port))
                except OSError:
                    if requested_port is not None:raise RuntimeError('Requested local port is occupied; choose another --local-port')
                    probe.bind(('127.0.0.1',0));a.local_port=probe.getsockname()[1]
                    base=f'http://127.0.0.1:{a.local_port}'
            subprocess.run(['ssh','-S',str(control),'-O','forward','-L',f'127.0.0.1:{a.local_port}:127.0.0.1:{info["port"]}',
                            '-p',str(a.ssh_port),a.host],check=True)
            connected=True
        url=base+'/pipeline.html'
        print(url)
        if connected:print('SSH tunnel stays running for up to 12 hours. Re-run this command to reconnect.')
        if not a.no_open:webbrowser.open(url)
    finally:
        if not connected and directory:
            subprocess.run(['ssh','-S',str(control),'-O','exit','-p',str(a.ssh_port),a.host],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:directory.rmdir()
            except OSError:pass

if __name__=='__main__':main()
