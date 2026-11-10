"""Copy only declared source/policy files, never geometry/images/models/runs."""
from pathlib import Path
import hashlib
import json
import re
import shutil

SOURCE = Path('/workspace/v519_capture_clone_20261028_tWZimf/original_producers')
TARGET = Path('/workspace/spacewatch_integrated_20261110_8QJn4J/code/upstream519_original')
RELEASE = Path('/workspace/spacewatch_pipeline_v5_1')
allowed = {'.py','.c','.so','.md','.txt','.html','.js','.css'}
files = []
modules = ('__init__.py','io.py','geometry.py','early_semantics.py','ooops.py','planning.py','reuse.py','worker.py','engine.py','gpu.py')
for name in modules:
    files.append((SOURCE/'v2/spacewatch_pipeline'/name,'v2/spacewatch_pipeline/'+name,'original_v2_source'))
for folder in ('vendor/segmentation','vendor/panovggt_pipeline_v2','vendor/native','vendor/spacewatch3d_src'):
    for p in sorted((SOURCE/'v2'/folder).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if p.suffix in allowed or (p.suffix=='.json' and 'schemas' in p.parts):
            files.append((p,'v2/'+p.relative_to(SOURCE/'v2').as_posix(),'original_v2_source'))
for name in ('prepare_map_profile.py','statistical_adapter.py'):
    files.append((SOURCE/name,name,'original_profile_producer'))
template=RELEASE/'profiles/dorm2_statistical_scored'
for folder in ('code','native','algorithms'):
    for p in sorted((template/folder).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and (p.suffix in allowed or (p.suffix=='.json' and 'schemas' in p.parts)):
            files.append((p,'statistical_policy_snapshot/template/'+p.relative_to(template).as_posix(),'accepted_statistical_policy'))
for name in ('profile_runner.py','integrated_runner.py','cache_optimization.py','fast_execution.py','manual.json','prompt.txt','config.json'):
    files.append((template/name,'statistical_policy_snapshot/template/'+name,'accepted_statistical_policy'))
files.extend([(RELEASE/'profiles/dorm3_statistical_scored/score_summary.json','statistical_policy_snapshot/class_score_summary.json','accepted_statistical_policy'),
              (RELEASE/'configs/dorm3.json','statistical_policy_snapshot/release_config.json','accepted_statistical_policy')])
raw=Path('/workspace/dorm4_dorm5_v5_20261025')
for name in ('prepare_new.py','run_all.py'):
    files.append((raw/name,'raw_input_producer/'+name,'accepted_raw_input_producer'))
files.append((raw/'code/v5/vendor/panovggt_pipeline_v2/configs/dorm2.json',
              'raw_input_producer/reconstruction_template.json','accepted_raw_input_producer'))
for p in sorted((raw/'code/v5/vendor/panovggt_pipeline_v2').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix in allowed:
        files.append((p,'raw_input_producer/panovggt_pipeline_v2/'+p.relative_to(raw/'code/v5/vendor/panovggt_pipeline_v2').as_posix(),'accepted_raw_input_producer'))
for name in ('offline_worker.py','network_guard/sitecustomize.py'):
    files.append((TARGET/name,name,'integrated_io_class_and_offline_adapter'))
pattern=re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[A-Z0-9]{16}|(?:api_key|password|secret|access_token)\s*[:=]\s*[\'\"][^\'\"]{8,}[\'\"]',re.I)
rows=[]
for src,rel,role in files:
    if src.stat().st_size>2*1024**2 or src.is_symlink():raise ValueError('Refuse oversized/symlink source '+str(src))
    if src.suffix!='.so' and pattern.search(src.read_text(errors='replace')):raise ValueError('Potential credential in source '+str(src))
    digest=hashlib.sha256(src.read_bytes()).hexdigest();dest=TARGET/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise ValueError('Changed target '+str(dest))
    if not dest.exists():shutil.copyfile(src,dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise ValueError('Source copy mismatch')
    rows.append(dict(source=str(src),relative_path=rel,sha256=digest,bytes=src.stat().st_size,role=role))
manifest=dict(files=rows,total_bytes=sum(r['bytes'] for r in rows),source_only=True,images_models_geometry_or_runs_copied=False,
              note='Original profile producer still reads accepted external release paths, verified against these frozen policy snapshots before execution.')
(TARGET/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=manifest['total_bytes'],all_source_hashes_match=True,images_models_geometry_or_runs_copied=False)))
