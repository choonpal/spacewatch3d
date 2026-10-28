"""CPU-only integration checks against previously verified server outputs."""
from pathlib import Path
import copy
import json
import py_compile
import sys

import cv2
import numpy as np

ROOT=Path('/workspace/propainter_camera_20261028')
sys.path.insert(0,str(ROOT))
from pipeline import read,save,sha,build_plan

cfg=read(ROOT/'config.json')
plan=read(ROOT/'results/plan.json')
report=read(ROOT/'results/run_complete.json')
poses=np.load(cfg['camera_poses'])
checks=[]
for target in plan['targets']:
    name=target['id']+'.png'
    new=ROOT/'results/images'/name
    old=Path('/workspace/panogrf_adjacent_13_20261027/inputs')/name
    actual=cv2.imread(str(new))
    original=cv2.imread(str(ROOT/'results/originals'/name))
    mask=cv2.imread(str(ROOT/'results/masks'/name),0)>0
    assert sha(new)==sha(old),(name,'output differs from previous verified result')
    assert np.array_equal(actual[~mask],original[~mask])
    assert np.array_equal(np.asarray(target['pose']),poses[target['camera_frame_id']])
    assert 'pair' not in target
    checks.append(dict(camera_frame_id=target['camera_frame_id'],byte_identical_to_previous=True,
                       outside_mask_unchanged=True,pose_equals_captured_pose=True))
assert report['completed'] and report['new_sam_seconds']==0 and report['new_propainter_seconds']==0
assert not (ROOT/'results/gpu_lease.json').exists()
changed=copy.deepcopy(cfg)
changed['propainter']['ref_stride']=3
changed_plan=build_plan(changed,ROOT/'scratch_changed_parameters')
assert all(not clip['reused'] for clip in changed_plan['clips'])
invalid=copy.deepcopy(cfg)
invalid['selected_camera_ids']=[7,7]
try:
    build_plan(invalid,ROOT/'scratch_duplicate_ids')
except ValueError:
    pass
else:
    raise AssertionError('Duplicate camera IDs accepted')
for path in ROOT.glob('*.py'):
    py_compile.compile(str(path),doraise=True)
    if path.name=='validate.py':
        continue
    code=path.read_text()
    assert not any(forbidden in code for forbidden in ('network.renderer','install_query_resolution','general_renderer/model.pth'))
save(ROOT/'validation.json',dict(status='passed',items=checks,changed_parameters_invalidate_cache=True,
 duplicate_camera_ids_rejected=True,no_renderer_imports=True,no_gpu_used=True,
 fresh_gpu_worker_execution_tested=False,run=report))
print(json.dumps(read(ROOT/'validation.json'),ensure_ascii=False,indent=2))
