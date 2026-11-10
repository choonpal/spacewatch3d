import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from spacewatch_pipeline.io import ROOT


class StartV5Tests(unittest.TestCase):
    def invoke(self, state, config_matches=True):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shutil.copy2(ROOT/'start_v5.sh',root/'start_v5.sh')
            (root/'run.sh').write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
            (root/'run.sh').chmod(0o755)
            (root/'config.json').write_text(json.dumps({'value':1}))
            out=root/'output';out.mkdir()
            (out/'config.json').write_text(json.dumps({'value':1 if config_matches else 2}))
            if state:(out/'status.json').write_text(json.dumps({'status':state}))
            return subprocess.run(['bash',str(root/'start_v5.sh'),'config.json',str(out)],capture_output=True,text=True)

    def test_complete_run_attaches_viewer_without_rerunning_pipeline(self):
        result=self.invoke('complete')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(result.stdout.startswith('viewer --output'))
        self.assertNotIn('doctor',result.stdout)

    def test_changed_configuration_is_not_silently_ignored(self):
        result=self.invoke('complete',False)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('different settings',result.stderr)
        self.assertEqual(result.stdout,'')

    def test_incomplete_run_keeps_normal_resume_path(self):
        result=self.invoke('interrupted')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(result.stdout.startswith('doctor --config'))
        self.assertIn('start --config',result.stdout)
