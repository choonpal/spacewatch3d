import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.gaussian_change.remote import Relay


class RelayCompletion(unittest.TestCase):
    def test_status_writer_lock_is_retried_before_declaring_connection_lost(self):
        status_reads, receipt_reads = 0, 0
        def read(path):
            nonlocal status_reads, receipt_reads
            if Path(path).name == 'status.json':
                status_reads+=1
                if status_reads in [1,3]:
                    raise PermissionError('temporary Windows status writer lock')
                return dict(state='connected',heartbeat_unix=time.time())
            receipt_reads+=1
            if receipt_reads==1:
                raise FileNotFoundError(path)
            return dict(ok=True,output='same job remained connected')
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.gaussian_change.remote.read',side_effect=read), \
             patch('tools.gaussian_change.remote.time.sleep'):
            self.assertEqual(Relay(temp).call(command='fixture'),'same job remained connected')

    def test_dead_local_connector_is_rejected_even_with_fresh_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'inbox').mkdir();(root/'outbox').mkdir()
            (root/'status.json').write_text(json.dumps(dict(state='connected',pid=123,heartbeat_unix=time.time())))
            with patch('tools.gaussian_change.remote.process_alive',return_value=False,create=True), \
                 patch('tools.gaussian_change.remote.time.monotonic',side_effect=[0,601]):
                with self.assertRaises(ConnectionError):
                    Relay(root).call(command='must not queue on a dead connector')
            self.assertEqual(list((root/'inbox').iterdir()), [])

    def test_connector_exit_during_request_is_reported_without_full_timeout(self):
        def read(path):
            if Path(path).name == 'status.json':
                return dict(state='connected',pid=123,heartbeat_unix=time.time())
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.gaussian_change.remote.read',side_effect=read), \
             patch('tools.gaussian_change.remote.process_alive',side_effect=[True,False]):
            with self.assertRaisesRegex(ConnectionError,'SSH'):
                Relay(temp).call(command='connector exits before the receipt arrives')

    def test_unresponsive_connector_does_not_wait_ten_minutes_for_a_short_command(self):
        def read(path):
            if Path(path).name == 'status.json':
                return dict(state='connected',pid=123,heartbeat_unix=0)
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.gaussian_change.remote.read',side_effect=read), \
             patch('tools.gaussian_change.remote.process_alive',return_value=True), \
             patch('tools.gaussian_change.remote.time.time',side_effect=[0,106]), \
             patch('tools.gaussian_change.remote.time.monotonic',side_effect=[0,0,601]), \
             patch('tools.gaussian_change.remote.time.sleep'):
            with self.assertRaises(ConnectionError):
                Relay(temp).call(command='short command on an unresponsive connector')

    def test_partial_or_temporarily_locked_receipt_is_retried(self):
        attempts = iter([PermissionError('writer open'), json.JSONDecodeError('partial', '{', 1),
                         {'ok': True, 'output': 'finished'}])
        def read(path):
            if Path(path).name == 'status.json':
                return {'state': 'connected', 'heartbeat_unix': time.time()}
            value = next(attempts)
            if isinstance(value, Exception):
                raise value
            return value
        with tempfile.TemporaryDirectory() as temp, patch('tools.gaussian_change.remote.read', side_effect=read), patch('tools.gaussian_change.remote.time.sleep'):
            self.assertEqual(Relay(temp).call(command='fixture'), 'finished')


if __name__ == '__main__':
    unittest.main()
