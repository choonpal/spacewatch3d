"""Run unmodified native tests in their pre-existing compatible environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

CODE=Path(__file__).resolve().parent
PLANNING={'test_planning_clearance','test_routing_scores','test_paper_selector','test_binmax'}


def main():
    if len(sys.argv)>1:
        group=sys.argv[1]
        sys.path[:0]=[str(CODE),str(CODE/'tests'),str(CODE/'vendor/florence_only'),str(CODE/'vendor/spacewatch3d_src')]
        loader=unittest.TestLoader()
        if group in ('geometry','planning'):
            names=[p.stem for p in sorted((CODE/'tests').glob('test_*.py'))
                   if (p.stem in PLANNING)==(group=='planning')]
            suite=unittest.TestSuite(loader.loadTestsFromName(n) for n in names)
        elif group=='florence':suite=loader.discover(str(CODE/'vendor/florence_only'),pattern='test_core.py')
        elif group=='upstream':suite=loader.discover(str(CODE/'upstream519_original/adapter_tests'),pattern='test_*.py')
        else:suite=loader.discover(str(CODE/'vendor/panovggt_pipeline_v2/tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=1).run(suite)
        print(json.dumps(dict(group=group,tests=result.testsRun,passed=result.wasSuccessful(),
                              failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))))
        return 0 if result.wasSuccessful() else 1
    runtime=json.loads((CODE/'configs/dorm45-paper.json').read_text())['runtime']
    folder=CODE.parent/'evidence/tests';folder.mkdir(exist_ok=True,parents=True)
    rows=[]
    for group in ['geometry','planning','florence','reconstruction','upstream']:
        executable=runtime['geometry_python' if group in ('florence','upstream') else group+'_python']
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'LD_LIBRARY_PATH':'/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')}
        child=subprocess.run([executable,__file__,group],cwd=CODE,env=env,text=True,capture_output=True,timeout=240)
        (folder/f'{group}.log').write_text(child.stdout+child.stderr)
        summaries=[json.loads(x) for x in child.stdout.splitlines() if x.startswith('{"group":')]
        row=dict(group=group,python=executable,returncode=child.returncode,summary=summaries[-1] if summaries else None)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='passed' if all(r['returncode']==0 for r in rows) else 'failed',
                tests_modified_to_pass=False,groups=rows)
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
