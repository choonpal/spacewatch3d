"""User-owned SSH queue process; passwords stay only in this process's memory."""
import argparse
import getpass
import json
import os
from pathlib import Path
import time
import uuid


def save(path,value):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value),encoding='utf-8')
    try:
        for attempt in range(100):
            try:temp.replace(path);return
            except PermissionError:
                if attempt==99:raise
                time.sleep(.05)
    finally:
        try:temp.unlink(missing_ok=True)
        except PermissionError:pass


def command(client,text,timeout):
    _,stdout,_=client.exec_command(text,get_pty=False,timeout=15)
    channel=stdout.channel;chunks=[];deadline=time.monotonic()+timeout
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if time.monotonic()>deadline:
            channel.close();raise TimeoutError('SSH command timed out; inspect the recorded remote worker before retrying')
        if channel.recv_ready():chunks.append(channel.recv(65536).decode(errors='replace'))
        if channel.recv_stderr_ready():chunks.append(channel.recv_stderr(65536).decode(errors='replace'))
        time.sleep(.05)
    return dict(exit_code=channel.recv_exit_status(),output=''.join(chunks))


def serve(args):
    import paramiko
    root=args.session.resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['inbox','outbox']:(root/name).mkdir(exist_ok=True)
    status=root/'status.json';client=None;secret=None
    def connect():
        ssh=paramiko.SSHClient();ssh.load_host_keys(str(args.known_hosts.expanduser()))
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            ssh.connect(args.host,port=args.port,username=args.user,password=secret,
                key_filename=str(args.key_file) if args.key_file else None,
                look_for_keys=not args.password,allow_agent=not args.password,timeout=12,auth_timeout=25,banner_timeout=15)
            ssh.get_transport().set_keepalive(15)
            return ssh
        except BaseException:
            ssh.close();raise
    try:
        if args.password:
            save(status,dict(state='awaiting_password',pid=os.getpid()))
            secret=getpass.getpass('SSH password (memory only): ')
        client=connect();heartbeat=0
        while True:
            transport=client.get_transport()
            if not transport or not transport.is_active():
                client.close();save(status,dict(state='reconnecting',pid=os.getpid()))
                for attempt in range(8):
                    try:client=connect();break
                    except (OSError,paramiko.SSHException):
                        if attempt==7:raise
                        time.sleep(5)
            if time.time()-heartbeat>8:
                save(status,dict(state='connected',pid=os.getpid(),heartbeat_unix=time.time()));heartbeat=time.time()
            for path in sorted((root/'inbox').glob('*.json')):
                target=root/'outbox'/path.name
                if target.exists():continue
                try:
                    request=json.loads(path.read_text(encoding='utf-8-sig'));result={}
                    if request.get('upload') or request.get('get'):
                        with client.open_sftp() as sftp:
                            sftp.get_channel().settimeout(90)
                            for local,remote in request.get('upload',[]):sftp.put(local,remote)
                            for remote,local in request.get('get',[]):
                                destination=Path(local);destination.parent.mkdir(parents=True,exist_ok=True)
                                if destination.exists():raise FileExistsError('Download destination already exists')
                                sftp.get(remote,str(destination))
                    if request.get('command'):
                        result.update(command(client,request['command'],min(600,request.get('timeout_seconds',90))))
                    result['ok']=True
                except Exception as error:
                    result=dict(ok=False,error=type(error).__name__+': '+str(error))
                save(target,result)
            time.sleep(.4)
    except BaseException as error:
        save(status,dict(state='error',pid=os.getpid(),error=type(error).__name__+': '+str(error)))
        raise
    finally:
        secret=None
        if client:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True);parser.add_argument('--port',type=int,default=22)
    parser.add_argument('--user',required=True);parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--known-hosts',type=Path,default=Path.home()/'.ssh/known_hosts')
    parser.add_argument('--key-file',type=Path);parser.add_argument('--password',action='store_true')
    serve(parser.parse_args())
