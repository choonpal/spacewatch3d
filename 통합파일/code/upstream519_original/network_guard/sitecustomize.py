"""Inherited by original producer subprocesses; no Python network downloads."""
import socket
import sys
sys.dont_write_bytecode = True


def _deny_network(*args, **kwargs):
    raise PermissionError('Original upstream is offline; all required model files must already be present')


socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
socket.create_connection = _deny_network
