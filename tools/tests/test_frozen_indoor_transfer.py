"""Frozen-source and acquisition-selection guarantees for an unseen pair."""
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_frozen_indoor_transfer import verify_frozen,temporal_windows


class FrozenTransferTests(unittest.TestCase):
    def test_changed_detector_source_is_rejected_before_transfer(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'detector.py';source.write_text('fixed algorithm')
            protocol={'source_code_sha256':{'detector.py':hashlib.sha256(source.read_bytes()).hexdigest()},
                'frozen_rescan_config':{'radii_m':[.08,.16,.28],'minimum_displacement_m':.06}}
            self.assertIs(verify_frozen(protocol,root),protocol['frozen_rescan_config'])
            source.write_text('algorithm tuned on new output')
            with self.assertRaisesRegex(ValueError,'source changed'):verify_frozen(protocol,root)

    def test_capture_windows_follow_predeclared_time_rule(self):
        self.assertEqual(temporal_windows([(100.,'first.png'),(125.,'last.png')]),((.5,5.),(20.,24.5)))
        with self.assertRaises(ValueError):temporal_windows([(100.,'a'),(104.,'b')])


if __name__=='__main__':unittest.main()
