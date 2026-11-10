"""Run the same CLI through an existing authenticated SSH file-queue relay.

Credentials are managed by the host's relay, never by this processing package.
Both server-resident inputs and local input transfer are supported explicitly.
"""
from pathlib import Path
import os
import json
import time
import uuid
from tools.mesh_simsac.core import read, write


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None  # Older relay receipts did not include an owner PID.
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if ctypes.get_last_error() == 5 else False
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def relay_state(folder):
    for attempt in range(40):
        try:
            return read(Path(folder)/'status.json')
        except (PermissionError, json.JSONDecodeError):
            if attempt == 39:
                raise
            time.sleep(.025)


def relay_connected(folder):
    try:
        state = relay_state(folder)
        return (state.get('state') == 'connected' and time.time()-state.get('heartbeat_unix', 0) <= 90
                and process_alive(state.get('pid')) is not False)
    except (OSError, ValueError):
        return False


class Relay:
    def __init__(self, folder):
        self.folder = Path(folder)

    def call(self, **request):
        if not relay_connected(self.folder):
            raise ConnectionError('SSH 연결이 종료됐습니다. 서버에 다시 로그인해 주세요.')
        key = 'mesh-rgb-change-'+uuid.uuid4().hex+'.json'
        write(self.folder/'inbox'/key, request)
        result = self.folder/'outbox'/key
        deadline = time.monotonic()+600
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('SSH relay request timed out')
            try:
                # The host relay can create the outbox file before it has
                # finished writing/closed its Windows handle.
                response = read(result)
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                pass
            state = relay_state(self.folder)
            if state.get('state') != 'connected' or process_alive(state.get('pid')) is False:
                raise ConnectionError('입력 처리 중 SSH 연결이 종료됐습니다. 서버에 다시 로그인해 주세요.')
            if request.get('command') and time.time()-state.get('heartbeat_unix', 0) > max(90, request.get('timeout_seconds', 90)+15):
                raise ConnectionError('SSH 연결 응답이 멈췄습니다. 서버 연결을 확인해 주세요.')
            time.sleep(.5)
        result = response
        if not result.get('ok') or result.get('exit_code', 0):
            raise RuntimeError(str(result.get('error', result.get('output')))[-3000:])
        return result.get('output', '')
